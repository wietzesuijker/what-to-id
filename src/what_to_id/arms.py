"""Batch composition arms: each turns a pool subset into a priority order of positional indices."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class Arm(Protocol):
    name: str

    def order(self, pool: pd.DataFrame, *, seed: int) -> np.ndarray:
        """Positional indices into pool, highest priority first."""
        ...


def _created_ns(pool: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(pool["created_at"], utc=True, errors="coerce")
    return ts.astype("int64").to_numpy().astype(np.float64)


def _recency_order(pool: pd.DataFrame) -> np.ndarray:
    """created_at desc, ties id desc."""
    created = _created_ns(pool)
    ids = pool["id"].to_numpy(dtype=np.int64)
    return np.lexsort((-ids, -created))


@dataclass(frozen=True)
class Recency:
    name: str = "recency"

    def order(self, pool: pd.DataFrame, *, seed: int) -> np.ndarray:
        return _recency_order(pool)


def _score_order(pool: pd.DataFrame, col: str) -> np.ndarray:
    score = pool[col].to_numpy(dtype=np.float64)
    created = _created_ns(pool)
    ids = pool["id"].to_numpy(dtype=np.int64)
    missing = ~np.isfinite(score)
    score = np.where(missing, -np.inf, score)
    # lexsort sorts by the last key first: NaN last, then score desc, created desc, id desc.
    return np.lexsort((-ids, -created, -score, missing.astype(np.int8)))


@dataclass(frozen=True)
class GapFirst:
    name: str = "gap_first"

    def order(self, pool: pd.DataFrame, *, seed: int) -> np.ndarray:
        if "cell_score" not in pool.columns:
            raise ValueError(
                "gap_first requires pool['cell_score'] (run cells.score_records first)"
            )
        return _score_order(pool, "cell_score")


@dataclass(frozen=True)
class Surprise:
    """Records whose proposed taxon is least expected where, or in what climate, it was seen.

    Orders by pool['surprise'] (what_to_id.surprise.score, a per-taxon tail probability) desc,
    NaN last. Most records that score 1 (the maximum) tie there because their taxon has too few
    regional references to say anything: among those ties, and any other tie at the same
    surprise, records are broken first by pool['surprise_sinr'] desc when present (a precomputed
    range-model score; a higher value means the location is less expected for the taxon, and
    NaN goes after any scored record), then by fewest regional references (pool['surprise_n_ref'],
    NaN treated as more unknown than any count), so the least-attested taxa surface before merely
    unusual sightings of well-attested ones. Remaining ties fall back to recency, then id. Pools
    missing surprise_sinr or surprise_n_ref order by whichever of the two is present, falling
    back to surprise, recency and id only when neither is.
    """

    name: str = "surprise"

    def order(self, pool: pd.DataFrame, *, seed: int) -> np.ndarray:
        if "surprise" not in pool.columns:
            raise ValueError("surprise requires pool['surprise'] (pass --surprise-scores)")
        has_sinr = "surprise_sinr" in pool.columns
        has_n_ref = "surprise_n_ref" in pool.columns
        if not has_sinr and not has_n_ref:
            return _score_order(pool, "surprise")
        score = pool["surprise"].to_numpy(dtype=np.float64)
        created = _created_ns(pool)
        ids = pool["id"].to_numpy(dtype=np.int64)
        missing_score = ~np.isfinite(score)
        score = np.where(missing_score, -np.inf, score)
        keys = [-ids, -created]
        if has_n_ref:
            n_ref = pool["surprise_n_ref"].to_numpy(dtype=np.float64)
            missing_ref = ~np.isfinite(n_ref)
            keys.append(np.where(missing_ref, np.inf, n_ref))
        if has_sinr:
            sinr = pool["surprise_sinr"].to_numpy(dtype=np.float64)
            missing_sinr = ~np.isfinite(sinr)
            keys.append(-np.where(missing_sinr, -np.inf, sinr))
        keys.append(-score)
        keys.append(missing_score.astype(np.int8))
        # lexsort sorts by the last key first: NaN surprise last, then surprise desc, then (when
        # present) surprise_sinr desc with NaN after any scored record, then (when present)
        # n_ref asc with NaN after known n_ref, then created desc, then id desc.
        return np.lexsort(tuple(keys))


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (ids int64, E float32 unit rows, backbone) from an npz written by the embed step."""
    with np.load(path, allow_pickle=False) as z:
        raw_ids = np.asarray(z["ids"])
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu":
            raise ValueError(f"{path}: ids must be a one-dimensional integer array")
        if raw_ids.size and raw_ids.max() > np.iinfo(np.int64).max:
            raise ValueError(f"{path}: ids exceed the int64 range")
        ids = raw_ids.astype(np.int64)
        E = np.asarray(z["E"], dtype=np.float32)
        backbone = str(z["backbone"]) if "backbone" in z.files else ""
    if E.ndim != 2 or E.shape[0] != ids.shape[0]:
        raise ValueError(f"{path}: E shape {E.shape} does not match {ids.shape[0]} ids")
    if E.shape[1] == 0:
        raise ValueError(f"{path}: embeddings must have at least one dimension")
    if not np.isfinite(E).all():
        raise ValueError(f"{path}: embeddings must contain only finite values")
    if not np.allclose(np.linalg.norm(E.astype(np.float64), axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{path}: embedding rows must have unit norm (zero rows are invalid)")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids")
    return ids, E, backbone


class _EmbeddingStore:
    """Lazy per-group access to embedding npz files.

    ``paths`` is either one npz path (used for every group) or a mapping group -> npz path with
    an optional ``"*"`` fallback. Groups with no file resolve to empty arrays.
    """

    def __init__(self, paths: Mapping[str, Path] | Path | str):
        self._paths = paths
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._models: dict[str, str] = {}
        self.backbones: dict[str | None, str] = {}

    def lookup(self, group: str | None) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self._paths, Mapping):
            path = self._paths.get(group if group is not None else "")
            if path is None and group is not None and "*" in self._paths:
                path = self._paths["*"]
            if path is None:
                return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        else:
            path = Path(self._paths)
        key = str(path)
        if key not in self._cache:
            ids, E, backbone = load_embeddings(Path(path))
            self._cache[key] = (ids, E)
            self._models[key] = backbone
        self.backbones[group] = self._models.get(key, "")
        return self._cache[key]


def _pool_group(pool: pd.DataFrame) -> str | None:
    groups = pool["iconic_taxon"].dropna().unique() if "iconic_taxon" in pool else []
    return str(groups[0]) if len(groups) == 1 else None


def _align(pool: pd.DataFrame, emb_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (emb row per pool row or -1, embedded pool positions, unembedded positions)."""
    n = len(pool)
    pool_ids = pool["id"].to_numpy(dtype=np.int64)
    pos_in_emb = pd.Index(emb_ids).get_indexer(pool_ids) if emb_ids.size else np.full(n, -1)
    has = pos_in_emb >= 0
    return pos_in_emb, np.flatnonzero(has), np.flatnonzero(~has)


class Similarity:
    """Group records into visually similar batches via CoreSet centres.

    Within the pool handed to ``order`` (one group, one arm), n_batches = ceil(n / batch_size)
    seeds are picked with labelfirst CoreSet (farthest-first, so seeds are spread), then each
    seed grows into one batch of exactly batch_size by repeatedly taking the free record nearest
    the batch's running centroid. Records are ordered by cosine to their batch centroid within
    a batch, and batches are ordered by the mean cell_score of their members (desc) when that
    column is present, else by seed index. Records without an embedding row go last in recency
    order.

    Batches cut from this order are therefore one grown cluster each. On the 2026-09-11 BC
    sample with BioCLIP 2.5 this gave mean batch cohesion 0.69 against 0.63 for random batches
    of the same size, versus 0.67 for the earlier assign-to-nearest-centre construction.
    """

    name = "similarity"

    def __init__(self, embeddings: Mapping[str, Path] | Path | str, batch_size: int):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = int(batch_size)
        self._store = _EmbeddingStore(embeddings)

    def _lookup(self, group: str | None) -> tuple[np.ndarray, np.ndarray]:
        return self._store.lookup(group)

    def order(self, pool: pd.DataFrame, *, seed: int) -> np.ndarray:
        from labelfirst import CoreSet

        if len(pool) == 0:
            return np.empty(0, dtype=np.int64)
        emb_ids, E_all = self._lookup(_pool_group(pool))
        pos_in_emb, embedded, rest = _align(pool, emb_ids)
        if embedded.size == 0:
            return _recency_order(pool)

        X = E_all[pos_in_emb[embedded]]
        k = min(math.ceil(embedded.size / self.batch_size), embedded.size)
        picks = CoreSet(seed=seed).select(X, labeled=np.array([], dtype=int), k=k).picks
        member_of, member_sim = self._grow(X, picks)

        if "cell_score" in pool.columns:
            cs = pool["cell_score"].to_numpy(dtype=np.float64)[embedded]
            centre_key = np.array(
                [
                    -np.nanmean(cs[member_of == c])
                    if np.isfinite(cs[member_of == c]).any()
                    else np.inf
                    for c in range(k)
                ]
            )
            centre_rank = np.argsort(centre_key, kind="stable")
        else:
            centre_rank = np.arange(k)
        rank_of_centre = np.empty(k, dtype=np.int64)
        rank_of_centre[centre_rank] = np.arange(k)

        order_emb = np.lexsort((-member_sim, rank_of_centre[member_of]))
        head = embedded[order_emb]
        tail = rest[_recency_order(pool.iloc[rest])] if rest.size else rest
        return np.concatenate([head, tail]).astype(np.int64)

    def _grow(self, X: np.ndarray, picks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grow one batch per seed: repeatedly add the free row nearest the running centroid.

        Returns (batch index per row, cosine of each row to its batch's final centroid). Seeds
        that were already absorbed by an earlier batch are replaced by the nearest free row.
        """
        n = X.shape[0]
        free = np.ones(n, dtype=bool)
        member_of = np.full(n, -1, dtype=np.int64)
        for c, p in enumerate(picks):
            if not free.any():
                break
            if not free[p]:
                cand = np.flatnonzero(free)
                p = cand[(X[cand] @ X[p]).argmax()]
            members = [int(p)]
            free[p] = False
            centroid = X[p].astype(np.float64).copy()
            while len(members) < self.batch_size and free.any():
                sims = X @ (centroid / max(np.linalg.norm(centroid), 1e-12))
                sims[~free] = -np.inf
                q = int(sims.argmax())
                members.append(q)
                free[q] = False
                centroid += X[q]
            member_of[members] = c
        # Any rows left when seeds ran out join the last batch.
        member_of[member_of < 0] = len(picks) - 1
        member_sim = np.empty(n, dtype=np.float64)
        for c in np.unique(member_of):
            rows = np.flatnonzero(member_of == c)
            centroid = X[rows].mean(axis=0)
            member_sim[rows] = X[rows] @ (centroid / max(np.linalg.norm(centroid), 1e-12))
        return member_of, member_sim


class Novelty:
    """Put records that look least like anything already identified first.

    Each embedded record is scored by its cosine distance to the nearest Research Grade
    reference embedding of the same group (1 - max cosine similarity). Records are ordered by that
    distance descending, ties created_at desc. Records without an embedding row, or in a group
    with no reference embeddings, go last in recency order.

    This is the discovery arm: a large distance means the photo sits in a part of image space
    with no verified record from the region nearby, so an ID there is more likely to add a
    species or a look that the reference does not cover yet.
    """

    name = "novelty"

    def __init__(
        self,
        embeddings: Mapping[str, Path] | Path | str,
        reference: Mapping[str, Path] | Path | str,
        chunk: int = 4096,
    ):
        if chunk < 1:
            raise ValueError("chunk must be >= 1")
        self._pool = _EmbeddingStore(embeddings)
        self._ref = _EmbeddingStore(reference)
        self.chunk = int(chunk)

    def distances(self, X: np.ndarray, R: np.ndarray) -> np.ndarray:
        """1 - max cosine similarity of each row of X to any row of R; rows are unit norm."""
        if R.shape[1] != X.shape[1]:
            raise ValueError(f"embedding dims differ: pool {X.shape[1]} vs reference {R.shape[1]}")
        out = np.empty(X.shape[0], dtype=np.float64)
        for start in range(0, X.shape[0], self.chunk):
            sims = X[start : start + self.chunk] @ R.T
            # float32 self-similarity can exceed 1 by rounding; a distance is never negative.
            out[start : start + self.chunk] = np.maximum(0.0, 1.0 - sims.max(axis=1))
        return out

    def order(self, pool: pd.DataFrame, *, seed: int) -> np.ndarray:
        if len(pool) == 0:
            return np.empty(0, dtype=np.int64)
        group = _pool_group(pool)
        emb_ids, E_all = self._pool.lookup(group)
        ref_ids, R = self._ref.lookup(group)
        candidate_model = self._pool.backbones.get(group)
        reference_model = self._ref.backbones.get(group)
        if candidate_model and reference_model and candidate_model != reference_model:
            raise ValueError(
                f"embedding backbones differ: pool {candidate_model!r} "
                f"vs reference {reference_model!r}"
            )
        pos_in_emb, embedded, rest = _align(pool, emb_ids)
        if embedded.size == 0 or R.size == 0:
            return _recency_order(pool)

        dist = self.distances(E_all[pos_in_emb[embedded]], R)
        created = _created_ns(pool)[embedded]
        ids = pool["id"].to_numpy(dtype=np.int64)[embedded]
        head = embedded[np.lexsort((-ids, -created, -dist))]
        tail = rest[_recency_order(pool.iloc[rest])] if rest.size else rest
        return np.concatenate([head, tail]).astype(np.int64)


ARMS: dict[str, type] = {
    "recency": Recency,
    "gap_first": GapFirst,
    "similarity": Similarity,
    "novelty": Novelty,
    "surprise": Surprise,
}


def build_arm(name: str, **kw) -> Arm:
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; choose from {sorted(ARMS)}")
    return ARMS[name](**kw)
