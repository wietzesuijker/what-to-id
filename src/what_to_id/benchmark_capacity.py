"""Bounded CPU capacity experiment with verified chunk reuse, not production serving."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from what_to_id import benchmark_embeddings as small
from what_to_id import embed
from what_to_id.manifest import sha256_file
from what_to_id.prepare_embeddings import MODEL_HASHES, MODEL_REVISION


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def _versions() -> dict:
    versions = {}
    for name in ("torch", "torchvision", "open_clip_torch", "pillow", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _validate_vectors(ids, vectors, expected) -> None:
    if not np.array_equal(ids, expected):
        raise ValueError("chunk observation IDs do not match their assigned rows")
    if vectors.shape != (len(expected), 1024) or not np.isfinite(vectors).all():
        raise ValueError("chunk must contain finite 1024-dimensional BioCLIP vectors")
    if not np.allclose(np.linalg.norm(vectors.astype(np.float64), axis=1), 1, atol=1e-5):
        raise ValueError("chunk embedding rows must have unit norm")


def _photo_hashes(rows: pd.DataFrame, work: Path) -> dict:
    return {
        str(row.id): sha256_file(embed.photo_path(work, int(row.id), row.photo_url))
        for row in rows.itertuples()
    }


def benchmark(
    pool: Path,
    out: Path,
    work: Path,
    *,
    limit: int = 512,
    chunk_size: int = 32,
    batch: int = 1,
    threads: int = 4,
    stop_after_chunks: int | None = None,
) -> dict:
    if min(limit, chunk_size, batch, threads) <= 0:
        raise ValueError("limit, chunk size, batch size and thread count must be positive")
    if stop_after_chunks is not None and stop_after_chunks <= 0:
        raise ValueError("stop-after-chunks must be positive")
    source = pd.read_csv(pool) if pool.suffix.lower() == ".csv" else pd.read_parquet(pool)
    rows = small.sample_pool(source, limit)
    identity = {
        "schema": 1,
        "source_sha256": sha256_file(pool),
        "sample": rows[["id", "iconic_taxon", "photo_url"]].to_dict("records"),
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_HASHES,
        "preprocessing": "embed.load_backbone frozen open_clip_config.json preprocess_cfg",
        "normalization": "L2 float32",
        "runtime_versions": _versions(),
        "chunk_size": chunk_size,
        "batch": batch,
        "threads": threads,
    }
    work.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "checkpoints.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity:
            raise ValueError("capacity work directory belongs to different image/model inputs")
    else:
        manifest = {"identity": identity, "chunks": {}}
        _atomic_json(manifest_path, manifest)
    report = {
        "status": "running",
        "phase": "checkpoint_validation",
        "model": {"backbone": "bioclip25", "revision": MODEL_REVISION, "device": "cpu"},
        "runtime": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "requested_threads": threads,
            "batch_size": batch,
            "physical_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        },
        "sample_count": len(rows),
        "source_sha256": identity["source_sha256"],
        "seconds": {},
        "batches": [],
        "chunks": [],
        "failures": [],
        "encoded_count": 0,
        "reused_count": 0,
        "completed_count": 0,
        "model_loaded": False,
    }
    started = time.perf_counter()

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_rss_bytes"] = small.peak_rss_bytes()
        report["disk_free_bytes"] = shutil.disk_usage(work).free
        _atomic_json(out, report)

    save()
    try:
        parts = [rows.iloc[start : start + chunk_size] for start in range(0, len(rows), chunk_size)]
        if set(manifest["chunks"]) - {str(i) for i in range(len(parts))}:
            raise ValueError("checkpoint manifest has unexpected chunk indices")
        pending = []
        for index, part in enumerate(parts):
            previous = manifest["chunks"].get(str(index))
            artifact = work / f"chunk-{index:05d}.npz"
            if previous is None:
                pending.append(index)
                continue
            if sha256_file(artifact) != previous["sha256"]:
                raise ValueError(f"checkpoint hash mismatch: chunk {index}")
            with np.load(artifact, allow_pickle=False) as saved:
                _validate_vectors(saved["ids"], saved["E"], part.id.to_numpy())
            if previous["photos_sha256"] != _photo_hashes(part, work):
                raise ValueError(f"checkpoint source photo bytes changed: chunk {index}")
            report["reused_count"] += len(part)
        report["completed_count"] = report["reused_count"]
        report["seconds"]["checkpoint_validation"] = time.perf_counter() - started
        save()
        if not pending:
            report["status"] = "cached"
            report["phase"] = "finished"
            return report
        batch_fn, preprocess = small._load_cpu_encoder(work, threads, report, save)
        report["model_loaded"] = True
        report["phase"] = "photo_download"
        save()
        phase_started = time.perf_counter()
        staged = embed.stage_images(
            pd.concat([parts[index] for index in pending]), work, workers=4, timeout=20, retries=2
        )
        report["seconds"]["photo_download"] = time.perf_counter() - phase_started
        report["photo_files_bytes"] = sum(
            Path(path).stat().st_size for path in staged.local_path if Path(path).is_file()
        )
        report["failures"] = [
            {"id": int(obs_id), "error": error} for obs_id, error in staged.attrs.get("failed", [])
        ]
        if report["failures"]:
            raise ValueError("capacity benchmark requires complete photo coverage")
        staged = staged.set_index("id", drop=False)
        for index in pending:
            part = staged.loc[parts[index].id]
            chunk_started = time.perf_counter()
            timings = {"decode": 0.0, "preprocess": 0.0, "inference": 0.0}

            def decode(path, timings=timings):
                phase_started = time.perf_counter()
                try:
                    return small._decode_image(path)
                finally:
                    timings["decode"] += time.perf_counter() - phase_started

            def prepare(image, timings=timings):
                phase_started = time.perf_counter()
                try:
                    return preprocess(image)
                finally:
                    image.close()
                    timings["preprocess"] += time.perf_counter() - phase_started

            def encode(items, timings=timings, index=index):
                report["phase"] = f"inference_chunk_{index}"
                save()
                phase_started = time.perf_counter()
                vectors = batch_fn(items)
                elapsed = time.perf_counter() - phase_started
                timings["inference"] += elapsed
                report["batches"].append(
                    {"chunk": index, "count": len(items), "inference_seconds": elapsed}
                )
                save()
                return vectors

            vectors, kept, failed = embed._embed_paths(
                list(part.local_path), encode, prepare, batch=batch, open_image=decode
            )
            if failed or len(kept) != len(part):
                report["failures"].extend(
                    {"id": int(part.iloc[i].id), "error": error} for i, error in failed
                )
                raise ValueError(f"incomplete encoding for chunk {index}")
            _validate_vectors(part.id.to_numpy(), vectors, parts[index].id.to_numpy())
            photos_sha256 = _photo_hashes(part, work)
            artifact = work / f"chunk-{index:05d}.npz"
            temporary = artifact.with_suffix(".partial")
            report["phase"] = f"persist_chunk_{index}"
            save()
            phase_started = time.perf_counter()
            with temporary.open("wb") as target:
                np.savez_compressed(target, ids=part.id.to_numpy(), E=vectors)
            temporary.replace(artifact)
            timings["persist"] = time.perf_counter() - phase_started
            manifest["chunks"][str(index)] = {
                "sha256": sha256_file(artifact),
                "photos_sha256": photos_sha256,
                "count": len(part),
            }
            _atomic_json(manifest_path, manifest)
            report["encoded_count"] += len(part)
            report["completed_count"] += len(part)
            report["chunks"].append(
                {
                    "index": index,
                    "count": len(part),
                    "seconds": timings,
                    "elapsed_seconds": time.perf_counter() - chunk_started,
                    "artifact_bytes": artifact.stat().st_size,
                    "peak_rss_bytes": small.peak_rss_bytes(),
                }
            )
            save()
            if stop_after_chunks is not None and len(report["chunks"]) >= stop_after_chunks:
                report["status"] = "intentionally_stopped"
                report["phase"] = "checkpointed"
                return report
        report["status"] = "complete"
        report["phase"] = "finished"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stop-after-chunks", type=int)
    args = parser.parse_args()
    benchmark(
        args.pool,
        args.out,
        args.work_dir,
        limit=args.limit,
        chunk_size=args.chunk_size,
        batch=args.batch,
        threads=args.threads,
        stop_after_chunks=args.stop_after_chunks,
    )


if __name__ == "__main__":
    main()
