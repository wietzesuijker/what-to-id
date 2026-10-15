"""Measure compressed-index capacity; synthetic capacity is not real retrieval quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def normalized(rows):
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("vectors must be a matrix")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(rows).all() or (norms == 0).any():
        raise ValueError("vectors must be finite and nonzero")
    return np.ascontiguousarray(rows / norms)


def synthetic_rows(rng, seeds, count):
    """Stream resampled/jittered seeds, never allocate a full-size raw vector matrix."""
    rows = seeds[rng.integers(len(seeds), size=count)].copy()
    noise = rng.standard_normal(rows.shape, dtype=np.float32)
    rows += noise * np.float32(0.02 / np.sqrt(rows.shape[1]))
    return normalized(rows)


def index_for(faiss, dimensions, lists, pq, iterations):
    # L2 order on normalized vectors equals cosine order before quantization.
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(dimensions), dimensions, lists, pq, 8)
    index.cp.niter = iterations
    index.cp.seed = 17
    index.pq.cp.niter = iterations
    index.pq.cp.seed = 17
    index.use_precomputed_table = -1
    return index


def populate_codes(faiss, index, count, chunk, rng, check):
    """Balanced random PQ codes test capacity, not full vector insertion cost."""
    offset = 0
    for list_id in range(index.nlist):
        size = count // index.nlist + (list_id < count % index.nlist)
        for start in range(0, size, chunk):
            check("synthetic_codes")
            n = min(chunk, size - start)
            ids = np.arange(offset, offset + n, dtype=np.int64)
            codes = rng.integers(0, 256, (n, index.code_size), dtype=np.uint8)
            index.invlists.add_entries(list_id, n, faiss.swig_ptr(ids), faiss.swig_ptr(codes))
            offset += n
    index.ntotal = count


def real_quality(faiss, path, queries, pq, threads, nprobe=16):
    """Held-out photo queries against the remaining real sample, not all BC records."""
    with np.load(path, allow_pickle=False) as archive:
        key = "E" if "E" in archive else "embeddings"
        rows = normalized(archive[key])
    if rows.ndim != 2 or len(rows) < 1024 or rows.shape[1] % pq:
        raise ValueError("real sample needs >=1024 rows and dimensions divisible by PQ size")
    rng = np.random.default_rng(23)
    rows = rows[rng.permutation(len(rows))]
    query_count = min(queries, len(rows) // 5)
    query, database = rows[:query_count], rows[query_count:]
    lists = min(64, max(1, len(database) // 100))
    index = index_for(faiss, rows.shape[1], lists, pq, 10)
    start = time.perf_counter()
    index.train(database)
    index.add(database)
    index.nprobe = min(nprobe, lists)
    _, approximate = index.search(query, 12)
    exact = faiss.IndexFlatL2(rows.shape[1])
    exact.add(database)
    _, truth = exact.search(query, 12)
    recall = np.mean([len(set(a) & set(b)) / 12 for a, b in zip(approximate, truth, strict=True)])
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "scope": "held-out queries against remaining real supplied sample only",
        "full_population_quality_verified": False,
        "input_sha256": digest,
        "sample_rows": len(rows),
        "database_rows": len(database),
        "query_rows": len(query),
        "dimensions": rows.shape[1],
        "recall_at_12": float(recall),
        "nlist": lists,
        "nprobe": index.nprobe,
        "threads": threads,
        "seconds": time.perf_counter() - start,
    }


def benchmark(args, faiss, result):
    start = time.perf_counter()
    faiss.omp_set_num_threads(args.threads)
    result.update(
        status="running",
        faiss_version=faiss.__version__,
        numpy_version=np.__version__,
        threads=args.threads,
        synthetic=True,
        quality_warning="Balanced random PQ codes test capacity, not population recall or skew.",
        synthetic_method="Direct random PQ codes with balanced lists; not encoded photos",
        query_append_source="resampled seed vectors with normalized synthetic jitter",
        initial_full_population_encoding_benchmarked=False,
        historical_neighbor_refresh_benchmarked=False,
        count=args.count,
        query_count=args.queries,
        append_count=args.append,
        dimensions=args.dimensions,
        nlist=args.lists,
        pq_subquantizers=args.pq,
        pq_bits=8,
        nprobe=args.nprobe,
        chunk_rows=args.chunk,
        timings_seconds={},
    )
    times = result["timings_seconds"]

    def check(stage):
        result["last_stage"] = stage
        if time.perf_counter() - start > args.max_seconds:
            raise TimeoutError("benchmark wall-clock budget exceeded")

    rng = np.random.default_rng(args.seed)
    if args.vectors:
        with np.load(args.vectors, allow_pickle=False) as archive:
            key = "E" if "E" in archive else "embeddings"
            seeds = normalized(archive[key])
        if seeds.shape[1] != args.dimensions:
            raise ValueError("real sample dimensionality differs from --dimensions")
        seeds = seeds[: args.train_rows]
        result["training_source"] = "supplied real sample"
    else:
        seeds = normalized(
            rng.standard_normal((args.train_rows, args.dimensions), dtype=np.float32)
        )
        result["training_source"] = "synthetic normal vectors"
    result["training_rows"] = len(seeds)
    index = index_for(faiss, args.dimensions, args.lists, args.pq, args.iterations)
    t = time.perf_counter()
    check("train")
    index.train(seeds)
    times["train"] = time.perf_counter() - t
    t = time.perf_counter()
    populate_codes(faiss, index, args.count, args.chunk, rng, check)
    result["indexed_rows"] = index.ntotal
    times["populate_synthetic_codes"] = time.perf_counter() - t
    index.nprobe = args.nprobe
    queries = synthetic_rows(rng, seeds, args.queries)
    with tempfile.TemporaryDirectory(prefix="vector-index-", dir=args.work_dir) as temporary:
        target = Path(temporary) / "capacity.faiss"
        check("write")
        t = time.perf_counter()
        faiss.write_index(index, str(target))
        times["write"] = time.perf_counter() - t
        result["initial_index_bytes"] = target.stat().st_size
        del index
        check("read")
        t = time.perf_counter()
        index = faiss.read_index(str(target))
        times["read"] = time.perf_counter() - t
        check("search")
        t = time.perf_counter()
        _, found = index.search(queries, 12)
        times["search"] = time.perf_counter() - t
        if (found < 0).any():
            raise ValueError("synthetic search did not return twelve candidates per query")
        additions = synthetic_rows(rng, seeds, args.append)
        check("append")
        t = time.perf_counter()
        index.add(additions)
        times["append"] = time.perf_counter() - t
        # Explicit IDs remain stable when IVF entries are deleted.
        deleted_ids = np.arange(min(64, args.count), dtype=np.int64)
        check("remove_and_search")
        t = time.perf_counter()
        removed = index.remove_ids(deleted_ids)
        _, after = index.search(queries, 12)
        times["remove_and_search"] = time.perf_counter() - t
        if removed != len(deleted_ids) or np.isin(after, deleted_ids).any():
            raise ValueError("deleted observations remain in the index")
        result["removed_rows"] = int(removed)
        result["final_rows"] = int(index.ntotal)
        check("rewrite")
        t = time.perf_counter()
        faiss.write_index(index, str(target))
        times["rewrite"] = time.perf_counter() - t
        result["final_index_bytes"] = target.stat().st_size
        if target.stat().st_size >= 1_000_000_000:
            raise ValueError("compressed index exceeds the benchmark's 1GB bound")
        if args.index_out is not None:
            check("persist_final_index")
            t = time.perf_counter()
            args.index_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, args.index_out)
            times["persist_final_index"] = time.perf_counter() - t
            result["persisted_index"] = str(args.index_out)
    del index
    if args.vectors is not None:
        check("real_sample_quality")
        result["real_sample_quality"] = real_quality(
            faiss, args.vectors, args.queries, args.pq, args.threads, args.quality_nprobe
        )
    else:
        result["real_sample_quality"] = None
    check("complete")
    result["status"] = "complete"
    result["elapsed_seconds"] = time.perf_counter() - start


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vectors", type=Path)
    p.add_argument(
        "--index-out", type=Path, help="Optional retained final index for transfer tests"
    )
    p.add_argument(
        "--work-dir", type=Path, help="Existing scratch directory; index files are removed"
    )
    for name, default in {
        "count": 3_068_186,
        "dimensions": 1024,
        "queries": 512,
        "threads": 4,
        "lists": 1024,
        "pq": 64,
        "nprobe": 8,
        "quality-nprobe": 16,
        "chunk": 4096,
        "train-rows": 10_000,
        "iterations": 10,
        "append": 512,
        "seed": 17,
    }.items():
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--max-seconds", type=float, default=600)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (
        min(
            args.count,
            args.dimensions,
            args.queries,
            args.threads,
            args.lists,
            args.pq,
            args.nprobe,
            args.quality_nprobe,
            args.chunk,
            args.iterations,
            args.append,
        )
        <= 0
        or args.train_rows < max(256, args.lists)
        or args.dimensions % args.pq
        or args.count < 12
        or not 0 < args.max_seconds < float("inf")
        or args.nprobe > args.lists
    ):
        p.error("positive sizes, count>=12, sufficient training rows, valid PQ and nprobe required")
    result = {}
    code = 0
    started = time.perf_counter()
    try:
        import faiss

        benchmark(args, faiss, result)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        code = 1
    finally:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
        result["elapsed_seconds"] = time.perf_counter() - started
        result["io_scope"] = "local index file I/O only; no remote transfer or Pages deployment"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
