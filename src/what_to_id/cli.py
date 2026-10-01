"""what-to-id command line: `what-to-id build` composes batches from a frozen pool."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from datetime import UTC, date, datetime, timedelta
from importlib import metadata
from pathlib import Path

import pandas as pd

from what_to_id.arms import Arm, build_arm, load_embeddings
from what_to_id.artifacts import build_inputs, resolve_embeddings, validate_coverage
from what_to_id.assign import assign, assign_keyed
from what_to_id.batches import build_batches, identify_url
from what_to_id.cells import WEBAPP_DIR, score_records
from what_to_id.labelfirst_adapter import labelfirst_provenance
from what_to_id.manifest import (
    Manifest,
    blind_labels,
    blind_labels_keyed,
    build_record,
    code_commit,
    grid_hash,
    key_fingerprint,
    key_from_env,
    sha256_file,
    utc_now_iso,
    write_manifest,
)
from what_to_id.page import write_site
from what_to_id.page_map import MAP_NAME, write_map
from what_to_id.page_rotation import write_rotation_site
from what_to_id.rest import REST_AFTER, REST_DAYS, REST_FRONT, resting_from_log
from what_to_id.served_log import append_served, served_rows
from what_to_id.signals import batch_signals, signals_by_batch

log = logging.getLogger("what_to_id")


def _iso_date(s: str) -> str:
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {s!r}") from e


def _iso_utc(s: str) -> str:
    """An ISO-8601 UTC timestamp, normalised to whole seconds like `utc_now_iso`."""
    try:
        t = datetime.fromisoformat(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected an ISO-8601 UTC timestamp, got {s!r}") from e
    if t.utcoffset() != timedelta(0):
        raise argparse.ArgumentTypeError(f"expected a UTC timestamp (Z or +00:00), got {s!r}")
    return t.astimezone(UTC).replace(microsecond=0).isoformat()


def _make_arms(
    names: list[str], batch_size: int, emb: dict[str, Path], ref: dict[str, Path]
) -> dict[str, Arm]:
    arms: dict[str, Arm] = {}
    for n in names:
        if n == "similarity":
            if not emb:
                raise SystemExit("--embeddings is required for the similarity arm")
            arms[n] = build_arm(n, embeddings=emb, batch_size=batch_size)
        elif n == "novelty":
            if not emb or not ref:
                raise SystemExit(
                    "--embeddings and --reference-embeddings are required for the novelty arm"
                )
            arms[n] = build_arm(n, embeddings=emb, reference=ref)
        else:
            arms[n] = build_arm(n)
    return arms


def _sha_if_single(paths: dict[str, Path]) -> str | None:
    """sha256 of the embedding file when there is exactly one; per-group files are not hashed."""
    if len(paths) != 1:
        return None
    return sha256_file(next(iter(paths.values())))


def build(args: argparse.Namespace) -> Path:
    pool_path = Path(args.pool)
    out = Path(args.out)
    pool = pd.read_parquet(pool_path)
    if pool.empty:
        raise SystemExit(f"{pool_path}: pool is empty")
    if pool["id"].duplicated().any():
        raise SystemExit(f"{pool_path}: duplicate ids")
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    log.info("pool %s: %d rows", pool_path, len(pool))

    pool["cell_score"] = score_records(pool, webapp_dir=Path(args.webapp_dir)).to_numpy()
    log.info("cell_score present for %d rows", int(pool["cell_score"].notna().sum()))
    if "surprise" in arm_names:
        if not args.surprise_scores:
            raise SystemExit("--surprise-scores is required for the surprise arm")
        cols = pd.read_parquet(args.surprise_scores).columns
        has_n_ref = "n_ref" in cols
        want = ["id", "surprise", "n_ref"] if has_n_ref else ["id", "surprise"]
        s = (
            pd.read_parquet(args.surprise_scores, columns=want)
            .drop_duplicates("id")
            .set_index("id")
        )
        pool["surprise"] = pool["id"].map(s["surprise"]).astype("float64").to_numpy()
        log.info("surprise present for %d rows", int(pool["surprise"].notna().sum()))
        if has_n_ref:
            pool["surprise_n_ref"] = pool["id"].map(s["n_ref"]).astype("float64").to_numpy()
    if args.sinr_scores:
        cols = pd.read_parquet(args.sinr_scores).columns
        missing = {"id", "sinr_rel"} - set(cols)
        if missing:
            raise SystemExit(f"{args.sinr_scores}: missing column(s) {sorted(missing)}")
        r = (
            pd.read_parquet(args.sinr_scores, columns=["id", "sinr_rel"])
            .drop_duplicates("id")
            .set_index("id")
        )
        sinr_rel = pool["id"].map(r["sinr_rel"]).astype("float64")
        pool["surprise_sinr"] = (1 - sinr_rel.clip(upper=1)).to_numpy()
        log.info("surprise_sinr present for %d rows", int(pool["surprise_sinr"].notna().sum()))

    key = key_from_env(args.key_env) if args.key_env else None
    if key is None:
        assign_df = assign(pool, arm_names, seed=args.seed)
        labels = blind_labels(arm_names, args.seed)
    else:
        assign_df = assign_keyed(pool, arm_names, key=key)
        labels = blind_labels_keyed(arm_names, key)
    # A keyed build runs in public CI next to a public served log: per-arm counts in the log
    # would give the letter-to-list map away, so it names lists by letter only.
    shown = labels if key is not None else {a: a for a in arm_names}
    groups = sorted(pool["iconic_taxon"].dropna().astype(str).unique())
    emb, ref = resolve_embeddings(
        groups, args.embeddings, args.reference_embeddings, args.embedding_bundle
    )
    if args.embedding_bundle:
        validate_coverage(pool, emb, ref)
    input_files = build_inputs(emb, ref, args.surprise_scores, args.sinr_scores)
    dependency = labelfirst_provenance() if "similarity" in arm_names else None
    cache_context = ""
    if args.ordering_cache:
        cache_context = json.dumps(
            {
                "inputs": input_files,
                "dependency": dependency,
                "sources": {
                    path.name: sha256_file(path)
                    for path in sorted(Path(__file__).parent.glob("*.py"))
                },
                "runtime": [sys.version, platform.platform(), platform.machine()],
                "versions": {name: metadata.version(name) for name in ("numpy", "scipy", "pandas")},
                "threads": {
                    name: value
                    for name, value in os.environ.items()
                    if name.startswith(("OMP_", "OPENBLAS_", "MKL_", "VECLIB_"))
                },
            },
            sort_keys=True,
        )
    arms = _make_arms(arm_names, args.batch_size, emb, ref)
    resting: set[int] = set()
    if args.rest_log:
        resting = resting_from_log(
            args.rest_log,
            args.freeze,
            front=args.rest_front,
            after=args.rest_after,
            days=args.rest_days,
        )
        resting &= set(pool["id"].astype("int64"))
        log.info("%d records at rest", len(resting))
    batches_df = build_batches(
        pool,
        assign_df,
        arms,
        size=args.batch_size,
        seed=args.seed,
        max_batches=args.max_batches,
        cache_dir=Path(args.ordering_cache) if args.ordering_cache else None,
        cache_context=cache_context,
        resting=resting,
    )
    del arms  # Release selection caches before loading embeddings for batch signals.

    for (arm, group), n in batches_df.groupby(["arm", "group"])["id"].size().items():
        nb = batches_df[(batches_df["arm"] == arm) & (batches_df["group"] == group)][
            "batch_id"
        ].nunique()
        log.info("%-11s %-16s %5d records in %3d batches", shown[arm], group, n, nb)

    emb_sha = _sha_if_single(emb)
    ref_sha = _sha_if_single(ref)
    backbone = None
    if emb:
        first = next(iter(sorted(emb.values(), key=str)))
        backbone = load_embeddings(first)[2] or None
    signals: dict[str, dict] = {}
    if emb:
        sig_df = batch_signals(batches_df, emb, ref or None)
        signals = signals_by_batch(sig_df)
        per_arm = sig_df.merge(
            batches_df[["batch_id", "arm"]].drop_duplicates(), on="batch_id"
        ).groupby("arm")[["cohesion", "novelty"]]
        for arm, row in per_arm.mean().iterrows():
            log.info(
                "%-11s mean cohesion %.3f  mean novelty %.3f",
                shown[arm],
                row["cohesion"],
                row["novelty"],
            )
    batches = {}
    for bid, sub in batches_df.groupby("batch_id", sort=True):
        ids = [int(i) for i in sub.sort_values("position")["id"]]
        batches[bid] = {
            "arm": str(sub["arm"].iloc[0]),
            "group": str(sub["group"].iloc[0]),
            "url": identify_url(ids),
            "ids": ids,
            **signals.get(bid, {}),
        }
    m = Manifest(
        freeze=args.freeze,
        d1=args.d1,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        arms=arm_names,
        arm_labels=labels,
        pool_sha256=sha256_file(pool_path),
        pool_rows=int(len(pool)),
        embeddings_sha256=emb_sha,
        reference_sha256=ref_sha,
        backbone=backbone,
        max_batches=args.max_batches,
        rest=(
            {
                "front": args.rest_front,
                "after": args.rest_after,
                "days": args.rest_days,
                "rows": len(resting),
            }
            if args.rest_log
            else None
        ),
        served_rows=int(len(batches_df)),
        created_at=args.created_at or utc_now_iso(),
        batches=batches,
        design=args.design,
        assignment="keyed" if key is not None else "stratified",
        key_fingerprint=key_fingerprint(key) if key is not None else None,
        where_to_blitz_grid=grid_hash(args.webapp_dir),
        code_commit=code_commit(),
        input_files=input_files,
        labelfirst=dependency,
        labelfirst_commit=str(dependency.get("commit") or "") if dependency else "",
    )
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(m, out / "manifest.json")
    with open(out / "build_record.json", "w") as fh:
        json.dump(build_record(m), fh, indent=1, sort_keys=True)
        fh.write("\n")
    assign_df.to_parquet(out / "assign.parquet", index=False)
    batches_df.to_parquet(out / "batches.parquet", index=False)
    write_map(out / "site", pool, freeze=args.freeze)
    if args.design == "rotation":
        write_rotation_site(out / "site", m, map_href=MAP_NAME)
    else:
        write_site(out / "site", m, batches_df, pool=pool)
    if args.served_log:
        rows = served_rows(batches_df, labels, pool, args.freeze)
        total = append_served(args.served_log, rows)
        log.info("served log %s: %d rows today, %d in all", args.served_log, len(rows), len(total))
    log.info("wrote %s (%d batches)", out, len(batches))
    return out


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="what-to-id")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="compose batches, manifest and site from a frozen pool")
    b.add_argument("--pool", default="data/pool.parquet")
    b.add_argument("--freeze", type=_iso_date, required=True, help="pool freeze date YYYY-MM-DD")
    b.add_argument("--d1", type=_iso_date, required=True, help="first blitz day YYYY-MM-DD")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--batch-size", type=int, default=120)
    b.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="serve only the first N batches per arm and taxon group",
    )
    b.add_argument("--arms", default="recency,gap_first", help="comma-separated arm names")
    b.add_argument(
        "--embedding-bundle", help="verified directory made by what_to_id.artifacts pack"
    )
    b.add_argument("--embeddings", default=None, help="npz path pattern with {group}, or a dir")
    b.add_argument(
        "--reference-embeddings",
        default=None,
        help="Research Grade reference npz pattern with {group}, or a dir (novelty arm)",
    )
    b.add_argument(
        "--surprise-scores",
        default=None,
        help="parquet with id and surprise columns (surprise arm)",
    )
    b.add_argument(
        "--sinr-scores",
        default=None,
        help="parquet with id and sinr_rel columns; breaks surprise ties by range-model score",
    )
    b.add_argument(
        "--webapp-dir", default=str(WEBAPP_DIR), help="where-to-blitz cluster_results/ca"
    )
    b.add_argument(
        "--design",
        choices=["sets", "rotation"],
        default="sets",
        help="'sets' (default) pages one list per browser; 'rotation' cycles every identifier "
        "through all lists",
    )
    b.add_argument(
        "--key-env",
        default=None,
        metavar="NAME",
        help="assign lists by a keyed hash of the record id, with the hex key read from the "
        "environment variable NAME, so a record keeps its list across daily builds",
    )
    b.add_argument(
        "--served-log",
        default=None,
        help="append this build's served records, by list letter, to this parquet",
    )
    b.add_argument(
        "--rest-log",
        default=None,
        help="served log to read earlier builds from: a record in the first --rest-front "
        "batches of its list on --rest-after builds in a row goes to the back of its list for "
        "the next --rest-days days",
    )
    b.add_argument("--rest-front", type=int, default=REST_FRONT)
    b.add_argument("--rest-after", type=int, default=REST_AFTER)
    b.add_argument("--rest-days", type=int, default=REST_DAYS)
    b.add_argument("--ordering-cache", help="private run-local ordering cache; omit for replay")
    b.add_argument(
        "--created-at",
        type=_iso_utc,
        default=None,
        help="stamp the manifest with this ISO-8601 UTC time (default: now); replay passes the "
        "original build's time so the artifacts are byte-identical",
    )
    b.add_argument("--out", default="out")
    b.set_defaults(func=build)
    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.max_batches is not None and args.max_batches < 1:
        raise SystemExit("--max-batches must be >= 1")
    if (
        getattr(args, "rest_log", None)
        and min(args.rest_front, args.rest_after, args.rest_days) < 1
    ):
        raise SystemExit("--rest-front, --rest-after and --rest-days must be >= 1")
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
