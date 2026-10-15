"""Build a geographic atlas of a frozen image-embedding sample.

No training or fabricated coverage: only observations matched by ID to validated caches.
Exact cosine neighbors use the full vectors; PCA is a separate, lossy display.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from what_to_id.arms import load_embeddings


def group_payload(
    pool: pd.DataFrame,
    path: Path,
    group: str,
    *,
    reference_path: Path | None = None,
    reference_pool: pd.DataFrame | None = None,
) -> dict:
    ids, vectors, backbone = load_embeddings(path)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("embedding IDs must be unique")
    rows = pool.loc[pool.iconic_taxon.eq(group)].set_index("id")
    if not rows.index.is_unique:
        raise ValueError("pool IDs must be unique")
    cache_count = len(ids)
    pool_count = len(rows)
    keep = np.isin(ids, rows.index)
    ids, vectors = ids[keep], vectors[keep]
    if len(ids) < 3:
        raise ValueError(f"{group}: at least three matched embeddings are required")
    rows = rows.loc[ids]
    coords = rows[["lon", "lat"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all() or (np.abs(coords) > [180, 90]).any():
        raise ValueError(f"{group}: invalid coordinates")
    centered = vectors.astype(np.float64) - vectors.mean(axis=0, dtype=np.float64)
    # Fix signs so rebuilding the same data does not arbitrarily mirror the plot.
    if min(centered.shape) > 2:
        u, singular, vt = svds(centered, k=2, random_state=0)
    else:
        u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        u, singular, vt = u[:, :2], singular[:2], vt[:2]
    order = np.argsort(singular)[::-1]
    u, singular, vt = u[:, order], singular[order], vt[order]
    for axis in range(len(singular)):
        if vt[axis, np.argmax(np.abs(vt[axis]))] < 0:
            u[:, axis] *= -1
    xy = u * singular
    if xy.shape[1] == 1:
        xy = np.column_stack([xy[:, 0], np.zeros(len(xy))])
    variance = float(np.square(singular).sum() / max(np.square(centered).sum(), 1e-30))
    cosine = np.clip(vectors @ vectors.T, -1, 1)
    np.fill_diagonal(cosine, -np.inf)
    n = min(12, len(ids) - 1)
    neighbors = np.argsort(-cosine, axis=1, kind="stable")[:, :n]
    reference = None
    reference_matches = None
    reference_scores = None
    reference_ids = None
    reference_names = {}
    if reference_path is not None:
        reference_ids, ref_vectors, ref_backbone = load_embeddings(reference_path)
        if not backbone or not ref_backbone or backbone != ref_backbone:
            raise ValueError(f"{group}: candidate and reference encoder labels must match")
        if vectors.shape[1] != ref_vectors.shape[1]:
            raise ValueError(f"{group}: candidate and reference dimensions differ")
        if not len(reference_ids):
            raise ValueError(f"{group}: reference cache is empty")
        reference_group_verified_count = 0
        if reference_pool is not None:
            known = reference_pool.loc[reference_pool.id.isin(reference_ids)]
            if not known.iconic_taxon.eq(group).all():
                raise ValueError(f"{group}: reference cache contains known cross-group IDs")
            reference_group_verified_count = len(known)
            ref_rows = reference_pool.loc[reference_pool.iconic_taxon.eq(group)]
            reference_names = {
                int(row.id): str(row.taxon_name)
                for row in ref_rows.itertuples()
                if pd.notna(row.taxon_name)
            }
        # Block candidates to bound memory for larger reference samples.
        reference_matches = np.empty(len(vectors), dtype=np.int64)
        reference_scores = np.empty(len(vectors), dtype=np.float32)
        for start in range(0, len(vectors), 256):
            scores = np.clip(vectors[start : start + 256] @ ref_vectors.T, -1, 1)
            best = np.argmax(scores, axis=1)
            stop = start + len(scores)
            reference_matches[start:stop] = best
            reference_scores[start:stop] = scores[np.arange(len(scores)), best]
        reference = {
            "count": len(reference_ids),
            "group_verified_count": reference_group_verified_count,
            "group_unverified_count": len(reference_ids) - reference_group_verified_count,
            "candidate_overlap_count": int(np.isin(ids, reference_ids).sum()),
            "backbone": ref_backbone,
            "dimensions": int(ref_vectors.shape[1]),
            "cache_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "named_count": sum(int(i) in reference_names for i in reference_ids),
            "checkpoint_revision": None,
            "score": "1 - maximum full-space cosine similarity to the reference sample",
        }
    records = []
    for i, (obs_id, row) in enumerate(rows.iterrows()):
        records.append(
            {
                "id": int(obs_id),
                "lon": float(row.lon),
                "lat": float(row.lat),
                "name": str(row.taxon_name) if pd.notna(row.taxon_name) else "Unidentified",
                "photo": str(row.photo_url) if pd.notna(row.photo_url) else "",
                "date": str(row.observed_on) if pd.notna(row.observed_on) else "Unknown date",
                "xy": [round(float(v), 6) for v in xy[i]],
                "neighbors": [[int(j), round(float(cosine[i, j]), 5)] for j in neighbors[i]],
            }
        )
    if reference is not None:
        for i, record in enumerate(records):
            ref_id = int(reference_ids[reference_matches[i]])
            record["reference_distance"] = round(1 - float(reference_scores[i]), 6)
            record["reference_cosine"] = round(float(reference_scores[i]), 6)
            record["closest_reference_id"] = ref_id
            record["included_in_reference"] = bool(np.isin(record["id"], reference_ids))
            record["closest_reference_name"] = reference_names.get(ref_id)
    return {
        "reference": reference,
        "group": group,
        "pool_group_count": pool_count,
        "cache_count": cache_count,
        "matched_count": len(ids),
        "pool_without_vector_count": pool_count - len(ids),
        "cache_without_pool_count": cache_count - len(ids),
        "checkpoint_revision": None,
        "backbone": backbone,
        "dimensions": int(vectors.shape[1]),
        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "projection": "PCA",
        "variance_retained": variance,
        "records": records,
    }


def build(
    pool_path: Path,
    embedding_pattern: str,
    out: Path,
    *,
    snapshot_date: str,
    reference_embedding_pattern: str | None = None,
    reference_pool_path: Path | None = None,
) -> None:
    snapshot_date = date.fromisoformat(snapshot_date).isoformat()
    pool = pd.read_parquet(pool_path)
    required = {"id", "iconic_taxon", "lon", "lat", "taxon_name", "photo_url", "observed_on"}
    if not required <= set(pool):
        raise ValueError(f"pool is missing columns: {sorted(required - set(pool))}")
    if pool.id.duplicated().any():
        raise ValueError("pool IDs must be unique")
    reference_pool = None
    if reference_pool_path is not None:
        if reference_embedding_pattern is None:
            raise ValueError("reference pool requires reference embeddings")
        reference_pool = pd.read_parquet(reference_pool_path)
        if not {"id", "iconic_taxon", "taxon_name"} <= set(reference_pool):
            raise ValueError("reference pool requires id, iconic_taxon and taxon_name")
        if reference_pool.id.duplicated().any():
            raise ValueError("reference pool IDs must be unique")
    groups = sorted(pool.iconic_taxon.dropna().unique())
    if (
        reference_embedding_pattern is not None
        and len(groups) > 1
        and "{group}" not in reference_embedding_pattern
    ):
        raise ValueError("multi-group reference embeddings require a {group} path pattern")
    payloads = []
    for group in groups:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", group):
            raise ValueError(f"invalid life group: {group!r}")
        path = Path(embedding_pattern.format(group=group))
        if not path.is_file():
            raise ValueError(f"{group}: embedding cache does not exist: {path}")
        reference_path = None
        if reference_embedding_pattern is not None:
            reference_path = Path(reference_embedding_pattern.format(group=group))
            if not reference_path.is_file():
                raise ValueError(f"{group}: reference cache does not exist: {reference_path}")
        payloads.append(
            group_payload(
                pool,
                path,
                group,
                reference_path=reference_path,
                reference_pool=reference_pool,
            )
        )
    if not payloads:
        raise ValueError("pool must contain at least one life group")
    data_dir = out / "embeddings-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "reference_pool_sha256": (
            hashlib.sha256(reference_pool_path.read_bytes()).hexdigest()
            if reference_pool_path is not None
            else None
        ),
        "snapshot_date": snapshot_date,
        "pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "groups": [{"id": p["group"], "count": len(p["records"])} for p in payloads],
        "total": sum(len(p["records"]) for p in payloads),
    }
    for payload in payloads:
        target = data_dir / f"{payload['group']}.json"
        target.write_text(json.dumps(payload, separators=(",", ":")))
    template = Path(__file__).with_name("embedding_atlas.html").read_text()
    html = template.replace("__ATLAS_META__", json.dumps(metadata).replace("<", "\\u003c"))
    logic = Path(__file__).with_name("embedding_atlas_logic.js").read_text()
    html = html.replace("__ATLAS_LOGIC__", logic)
    (out / "embeddings.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--embeddings", required=True, help="Cache pattern containing {group}")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--snapshot-date", required=True, help="Frozen source snapshot date, YYYY-MM-DD"
    )
    parser.add_argument(
        "--reference-embeddings", help="Optional reference cache pattern with {group}"
    )
    parser.add_argument(
        "--reference-pool", type=Path, help="Optional names for matched reference IDs"
    )
    args = parser.parse_args()
    build(
        args.pool,
        args.embeddings,
        args.out,
        snapshot_date=args.snapshot_date,
        reference_embedding_pattern=args.reference_embeddings,
        reference_pool_path=args.reference_pool,
    )


if __name__ == "__main__":
    main()
