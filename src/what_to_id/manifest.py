"""Exposure manifest: everything needed to reproduce and read back a build."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# The where-to-blitz commit whose cluster_results/ca the cell scores are read from. Not the
# grid-outputs-v1 release: that tag predates the 25 km equal-area lattice the scores use.
WHERE_TO_BLITZ_REF = "where-to-blitz@3bdcc68f8d9d17aa5c6060dcbcd965a597dfadc7"
BLIND_LABELS = "ABCDEFGH"

REQUIRED_KEYS = {
    "freeze": str,
    "d1": str,
    "seed": int,
    "batch_size": int,
    "arms": list,
    "arm_labels": dict,
    "pool_sha256": str,
    "pool_rows": int,
    "where_to_blitz_ref": str,
    "labelfirst_commit": str,
    "created_at": str,
    "served_rows": int,
    "batches": dict,
}
BATCH_KEYS = ("arm", "group", "url", "ids")


def blind_labels(arms: Sequence[str], seed: int) -> dict[str, str]:
    """arm -> "A"/"B"/... in an order shuffled from seed, so pages never leak the arm name."""
    arms = list(arms)
    if len(arms) > len(BLIND_LABELS):
        raise ValueError(f"at most {len(BLIND_LABELS)} arms supported")
    perm = np.random.default_rng([int(seed), 0xB11D]).permutation(len(arms))
    return {arm: BLIND_LABELS[int(p)] for arm, p in zip(arms, perm, strict=True)}


def blind_labels_keyed(arms: Sequence[str], key: bytes) -> dict[str, str]:
    """Like `blind_labels`, but the permutation comes from the key, not a public seed.

    Deterministic for a given (key, arms): depends on neither the pool nor a public
    seed, so the blind label mapping cannot be reconstructed without the key.
    """
    arms = list(arms)
    if len(arms) > len(BLIND_LABELS):
        raise ValueError(f"at most {len(BLIND_LABELS)} arms supported")
    digest = hmac.new(key, b"what-to-id labels", hashlib.sha256).digest()
    seed = int.from_bytes(digest[:8], "big")
    perm = np.random.default_rng(seed).permutation(len(arms))
    return {arm: BLIND_LABELS[int(p)] for arm, p in zip(arms, perm, strict=True)}


def key_from_env(name: str = "WHAT_TO_ID_KEY") -> bytes:
    """Read a hex-encoded key from an environment variable.

    Requires at least 32 hex chars (16 bytes). Error messages never include the
    variable's value, so a raised error is always safe to log.
    """
    raw = os.environ.get(name)
    if raw is None:
        raise ValueError(f"environment variable {name!r} is not set")
    raw = raw.strip()
    if not raw:
        raise ValueError(f"environment variable {name!r} is empty")
    if len(raw) < 32:
        raise ValueError(f"environment variable {name!r} must be at least 32 hex chars")
    try:
        return bytes.fromhex(raw)
    except ValueError as exc:
        raise ValueError(f"environment variable {name!r} must be valid hex") from exc


def key_fingerprint(key: bytes) -> str:
    """First 12 hex chars of sha256(b"what-to-id fingerprint" + key). Safe to publish."""
    return hashlib.sha256(b"what-to-id fingerprint" + key).hexdigest()[:12]


def grid_hash(webapp_dir: Path | str) -> str | None:
    """The where-to-blitz grid's manifest_hash from its provenance.json, None if absent.

    Names the grid the cell scores were actually read from, whatever commit was checked out.
    """
    path = Path(webapp_dir) / "provenance.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("manifest_hash")


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass
class Manifest:
    freeze: str
    d1: str
    seed: int
    batch_size: int
    arms: list[str]
    arm_labels: dict[str, str]
    pool_sha256: str
    pool_rows: int
    where_to_blitz_ref: str = WHERE_TO_BLITZ_REF
    labelfirst_commit: str = ""
    labelfirst: dict | None = None
    embeddings_sha256: str | None = None
    reference_sha256: str | None = None
    backbone: str | None = None
    max_batches: int | None = None
    rest: dict | None = None
    served_rows: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    batches: dict[str, dict] = field(default_factory=dict)
    design: str = "sets"
    assignment: str = "stratified"
    key_fingerprint: str | None = None
    where_to_blitz_grid: str | None = None
    code_commit: str | None = None
    input_files: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(d: dict) -> None:
    """Raise ValueError naming the offending key when the manifest dict is malformed."""
    if not isinstance(d, dict):
        raise ValueError("manifest must be a dict")
    for key, typ in REQUIRED_KEYS.items():
        if key not in d:
            raise ValueError(f"manifest missing key {key!r}")
        if not isinstance(d[key], typ) or (typ is int and isinstance(d[key], bool)):
            raise ValueError(f"manifest key {key!r} must be {typ.__name__}")
    if not d["arms"]:
        raise ValueError("manifest key 'arms' must be non-empty")
    if "design" in d and d["design"] not in ("sets", "rotation"):
        raise ValueError("manifest key 'design' must be 'sets' or 'rotation'")
    if "assignment" in d and d["assignment"] not in ("stratified", "keyed"):
        raise ValueError("manifest key 'assignment' must be 'stratified' or 'keyed'")
    if set(d["arm_labels"]) != set(d["arms"]):
        raise ValueError("manifest key 'arm_labels' must cover exactly the arms")
    if len(set(d["arm_labels"].values())) != len(d["arm_labels"]):
        raise ValueError("manifest key 'arm_labels' must have unique labels")
    seen: set[int] = set()
    for bid, b in d["batches"].items():
        if not isinstance(b, dict):
            raise ValueError(f"batches[{bid!r}] must be a dict")
        for k in BATCH_KEYS:
            if k not in b:
                raise ValueError(f"batches[{bid!r}] missing key {k!r}")
        if b["arm"] not in d["arm_labels"]:
            raise ValueError(f"batches[{bid!r}] arm {b['arm']!r} not in arm_labels")
        if not isinstance(b["ids"], list) or not b["ids"]:
            raise ValueError(f"batches[{bid!r}] ids must be a non-empty list")
        for i in b["ids"]:
            if not isinstance(i, int) or isinstance(i, bool):
                raise ValueError(f"batches[{bid!r}] ids must be ints, got {i!r}")
            if i in seen:
                raise ValueError(f"batches[{bid!r}] id {i} appears in more than one batch")
            seen.add(i)


def write_manifest(m: Manifest, path: Path | str) -> None:
    d = m.to_dict()
    validate_manifest(d)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
        fh.write("\n")


def read_manifest(path: Path | str) -> Manifest:
    with open(path) as fh:
        d = json.load(fh)
    validate_manifest(d)
    return Manifest(**d)


# The public subset of the manifest: everything needed to rerun a build, and nothing that maps
# list letters to arms (arm_labels, batches) or the key. Single source of truth for build_record.
PUBLIC_BUILD_RECORD_FIELDS = (
    "freeze",
    "d1",
    "seed",
    "batch_size",
    "max_batches",
    "rest",
    "design",
    "assignment",
    "arms",
    "key_fingerprint",
    "pool_sha256",
    "pool_rows",
    "served_rows",
    "where_to_blitz_ref",
    "where_to_blitz_grid",
    "embeddings_sha256",
    "reference_sha256",
    "created_at",
    "input_files",
    "labelfirst",
    "backbone",
)


def code_commit() -> str | None:
    """The running code's commit: `git rev-parse HEAD` of this package's repo, else $GITHUB_SHA.

    The checkout comes first because the daily job may run a pinned CODE_REF, and then
    $GITHUB_SHA names the commit that triggered the run, not the code that ran. Never raises: a
    build record with a missing code_commit is still useful.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        if out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("GITHUB_SHA") or None


def build_record(m: Manifest) -> dict:
    """The public per-build record: everything needed to rerun this build, safe to publish."""
    d = m.to_dict()
    record = {k: d[k] for k in PUBLIC_BUILD_RECORD_FIELDS}
    record["code_commit"] = m.code_commit or code_commit()
    return record
