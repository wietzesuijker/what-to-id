"""Replay: rerun a past daily build from a saved pool snapshot and check it against the served log.

Runs the same code path as `what-to-id build`, into a scratch directory, and never touches the
real served log. Refuses to run at all when the pool, key, grid or arm kind on hand do not match
what the build record says the original build used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from what_to_id.artifacts import build_inputs, resolve_embeddings
from what_to_id.cli import main as cli_main
from what_to_id.labelfirst_adapter import labelfirst_provenance
from what_to_id.manifest import code_commit, grid_hash, key_fingerprint, key_from_env, sha256_file
from what_to_id.served_log import COLUMNS

log = logging.getLogger("what_to_id.replay")

# Env var name used only for the scratch rerun below, regardless of what env var (if any) the
# caller read the key from. Never logged; cleared as soon as the rerun finishes.
_RERUN_KEY_ENV = "_WHAT_TO_ID_REPLAY_KEY"


class ReplayError(ValueError):
    """A replay was refused: the pool, key, grid or arm kind on hand does not match the record."""


@dataclass
class ReplayResult:
    ok: bool
    expected_rows: int
    rebuilt_rows: int
    mismatches: pd.DataFrame


def _load_record(record_path: Path | str) -> dict:
    return json.loads(Path(record_path).read_text())


def _check_preconditions(
    record: dict, pool_path: Path, webapp_dir: Path, key: bytes | None
) -> None:
    pool_sha = sha256_file(pool_path)
    if pool_sha != record["pool_sha256"]:
        raise ReplayError(
            f"{pool_path}: sha256 {pool_sha} does not match the record's pool_sha256 "
            f"{record['pool_sha256']}"
        )
    if record.get("assignment") == "keyed":
        if key is None:
            raise ReplayError("record used a keyed assignment; pass --key-env")
        fp = key_fingerprint(key)
        if fp != record.get("key_fingerprint"):
            raise ReplayError(
                f"key fingerprint {fp} does not match the record's key_fingerprint "
                f"{record.get('key_fingerprint')!r}"
            )
    grid = grid_hash(webapp_dir)
    if grid != record.get("where_to_blitz_grid"):
        raise ReplayError(
            f"{webapp_dir}: grid hash {grid!r} does not match the record's where_to_blitz_grid "
            f"{record.get('where_to_blitz_grid')!r}"
        )


def _rerun(
    record: dict,
    pool_path: Path,
    webapp_dir: Path,
    key: bytes | None,
    out: Path,
    served_log: Path,
    inputs: dict[str, str | None],
    rest_log: Path | None = None,
) -> None:
    argv = [
        "build",
        "--pool",
        str(pool_path),
        "--freeze",
        record["freeze"],
        "--d1",
        record["d1"],
        "--seed",
        str(record["seed"]),
        "--batch-size",
        str(record["batch_size"]),
        "--arms",
        ",".join(record["arms"]),
        "--webapp-dir",
        str(webapp_dir),
        "--design",
        record["design"],
        "--created-at",
        record["created_at"],
        "--out",
        str(out),
        "--served-log",
        str(served_log),
    ]
    for name, spec in inputs.items():
        if spec is not None:
            argv += ["--" + name.replace("_", "-"), str(spec)]
    if record.get("max_batches") is not None:
        argv += ["--max-batches", str(record["max_batches"])]
    rest = record.get("rest")
    if rest is not None:
        if rest_log is None:
            raise ReplayError("the record rested records; pass the served log it read")
        # Rest reads only builds before this one, so the full log rests the same records.
        argv += ["--rest-log", str(rest_log)]
        argv += ["--rest-front", str(rest["front"]), "--rest-after", str(rest["after"])]
        argv += ["--rest-days", str(rest["days"])]
    if record.get("assignment") == "keyed":
        os.environ[_RERUN_KEY_ENV] = key.hex()
        argv += ["--key-env", _RERUN_KEY_ENV]
    try:
        cli_main(argv)
    finally:
        os.environ.pop(_RERUN_KEY_ENV, None)


def _compare(expected: pd.DataFrame, rebuilt: pd.DataFrame) -> pd.DataFrame:
    """Rows on one side only, on every served-log column (NaN matches NaN); empty on a match."""
    if expected.duplicated(COLUMNS).any() or rebuilt.duplicated(COLUMNS).any():
        raise ReplayError("served logs contain duplicate rows")
    merged = expected[COLUMNS].merge(rebuilt[COLUMNS], on=COLUMNS, how="outer", indicator=True)
    merged = merged[merged["_merge"] != "both"]
    side = merged.pop("_merge").map({"left_only": "served", "right_only": "rebuilt"}).astype(str)
    return merged.assign(side=side)


def replay(
    pool_path: Path | str,
    record_path: Path | str,
    served_log_path: Path | str,
    webapp_dir: Path | str,
    key: bytes | None = None,
    *,
    embedding_bundle: str | None = None,
    embeddings: str | None = None,
    reference_embeddings: str | None = None,
    surprise_scores: str | None = None,
    sinr_scores: str | None = None,
) -> ReplayResult:
    pool_path = Path(pool_path)
    webapp_dir = Path(webapp_dir)
    record = _load_record(record_path)

    _check_preconditions(record, pool_path, webapp_dir, key)
    inputs = dict(
        embedding_bundle=embedding_bundle,
        embeddings=embeddings,
        reference_embeddings=reference_embeddings,
        surprise_scores=surprise_scores,
        sinr_scores=sinr_scores,
    )
    groups = sorted(
        pd.read_parquet(pool_path, columns=["iconic_taxon"])["iconic_taxon"]
        .dropna()
        .astype(str)
        .unique()
    )
    try:
        candidates, references = resolve_embeddings(
            groups, embeddings, reference_embeddings, embedding_bundle
        )
        actual = build_inputs(candidates, references, surprise_scores, sinr_scores)
    except (OSError, ValueError, SystemExit) as exc:
        raise ReplayError(f"embedding or score input unavailable: {exc}") from exc
    expected = record.get("input_files", {})
    if not expected and (
        record.get("embeddings_sha256")
        or record.get("reference_sha256")
        or set(record["arms"]) & {"similarity", "novelty", "surprise"}
    ):
        raise ReplayError("embedding or score build has no complete input_files record; rebuild it")
    if actual != expected:
        raise ReplayError("embedding or score inputs do not match the recorded files")

    if "similarity" in record["arms"]:
        dependency = record.get("labelfirst")
        if dependency is None or dependency != labelfirst_provenance():
            raise ReplayError("installed LabelFirst does not match the recorded provenance")

    current = code_commit()
    if current != record.get("code_commit"):
        log.warning(
            "current code_commit %s does not match the record's code_commit %s",
            current,
            record.get("code_commit"),
        )

    # The rerun logs into a fresh served log of its own, through the same code path as the build.
    with tempfile.TemporaryDirectory() as tmp:
        rerun_log = Path(tmp) / "served.parquet"
        _rerun(
            record,
            pool_path,
            webapp_dir,
            key,
            Path(tmp) / "out",
            rerun_log,
            inputs,
            Path(served_log_path),
        )
        rebuilt = pd.read_parquet(rerun_log)

    build_date = str(record["freeze"])
    log_df = pd.read_parquet(served_log_path)
    expected = log_df[log_df["build_date"] == build_date]

    mismatches = _compare(expected, rebuilt)
    ok = mismatches.empty and len(expected) == len(rebuilt)
    log.info(
        "build_date %s: %d rows expected, %d rows rebuilt, %d mismatched",
        build_date,
        len(expected),
        len(rebuilt),
        len(mismatches),
    )
    return ReplayResult(
        ok=ok, expected_rows=len(expected), rebuilt_rows=len(rebuilt), mismatches=mismatches
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m what_to_id.replay",
        description="rerun a past daily build from a pool snapshot and check it against the "
        "served log",
    )
    p.add_argument("--pool", required=True, help="the frozen pool snapshot used for that build")
    p.add_argument("--record", required=True, help="that build's build_record.json")
    p.add_argument("--served-log", required=True, help="the served log to check the rerun against")
    p.add_argument("--webapp-dir", required=True, help="where-to-blitz cluster_results/ca")
    p.add_argument(
        "--key-env",
        default=None,
        metavar="NAME",
        help="for a keyed record, the env var NAME holding the hex list key",
    )
    for name in (
        "embedding-bundle",
        "embeddings",
        "reference-embeddings",
        "surprise-scores",
        "sinr-scores",
    ):
        p.add_argument("--" + name, help="original build input, verified against its recorded hash")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    key = key_from_env(args.key_env) if args.key_env else None
    try:
        result = replay(
            args.pool,
            args.record,
            args.served_log,
            args.webapp_dir,
            key=key,
            embedding_bundle=args.embedding_bundle,
            embeddings=args.embeddings,
            reference_embeddings=args.reference_embeddings,
            surprise_scores=args.surprise_scores,
            sinr_scores=args.sinr_scores,
        )
    except ReplayError as exc:
        log.error("replay refused: %s", exc)
        return 1
    if not result.ok:
        log.error(
            "replay mismatch: %d rows expected, %d rows rebuilt, %d mismatched rows (first few "
            "below)",
            result.expected_rows,
            result.rebuilt_rows,
            len(result.mismatches),
        )
        log.error("%s", result.mismatches.head().to_string())
        return 1
    log.info("replay matches: %d rows", result.expected_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
