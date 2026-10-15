"""Small CPU-only BioCLIP benchmark, isolated from production caches and bundles."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from what_to_id import embed
from what_to_id.manifest import sha256_file
from what_to_id.prepare_embeddings import MODEL_HASHES, MODEL_REVISION, _verify_snapshot


def sample_pool(pool: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Deterministic round-robin over life groups, ordered by observation ID."""
    required = {"id", "iconic_taxon", "photo_url"}
    if not required <= set(pool):
        raise ValueError(f"pool lacks columns: {sorted(required - set(pool))}")
    if limit <= 0 or pool.empty:
        raise ValueError("sample limit and pool must be nonempty")
    if pool.id.isna().any() or pool.id.duplicated().any():
        raise ValueError("pool observation IDs must be present and unique")
    if pool.id.dtype.kind not in "iu" or (pool.id <= 0).any():
        raise ValueError("pool observation IDs must be positive integers")
    if (
        pool.iconic_taxon.isna().any()
        or not pool.iconic_taxon.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all()
    ):
        raise ValueError("pool life groups must be nonempty strings")
    if not pool.photo_url.map(
        lambda value: isinstance(value, str) and value.startswith("https://")
    ).all():
        raise ValueError("pool photo URLs must be nonempty HTTPS URLs")
    groups = [part.sort_values("id") for _, part in pool.groupby("iconic_taxon", sort=True)]
    chosen = []
    for position in range(max(map(len, groups))):
        for group in groups:
            if position < len(group):
                chosen.append(group.iloc[[position]])
                if len(chosen) == limit:
                    return pd.concat(chosen, ignore_index=True)
    return pd.concat(chosen, ignore_index=True)


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _decode_image(path: str):
    image = embed._open_image(path)
    image.load()
    return image


def _load_cpu_encoder(work: Path, threads: int, report: dict, save: Callable):
    report["phase"] = "framework_import"
    save()
    started = time.perf_counter()
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
    import torch
    from huggingface_hub import snapshot_download

    report["seconds"]["framework_import"] = time.perf_counter() - started

    torch.set_num_threads(threads)
    report["runtime"]["torch_threads"] = torch.get_num_threads()
    report["runtime"]["torch_interop_threads"] = torch.get_num_interop_threads()
    report["runtime"]["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "open_clip_torch", "pillow", "numpy")
    }
    report["phase"] = "model_download"
    save()
    started = time.perf_counter()
    snapshot = Path(
        snapshot_download(
            repo_id="imageomics/bioclip-2.5-vith14",
            revision=MODEL_REVISION,
            allow_patterns=list(MODEL_HASHES),
            cache_dir=work / "model-cache",
            etag_timeout=20,
        )
    )
    report["seconds"]["model_download_or_cache_lookup"] = time.perf_counter() - started
    report["model"]["snapshot"] = snapshot.name
    report["model"]["snapshot_bytes"] = sum(
        (snapshot / name).stat().st_size for name in MODEL_HASHES
    )
    report["phase"] = "model_verification"
    save()
    started = time.perf_counter()
    report["model"]["sha256"] = _verify_snapshot(snapshot)
    report["seconds"]["model_verification"] = time.perf_counter() - started
    report["phase"] = "model_load"
    save()
    started = time.perf_counter()
    model, preprocess = embed.load_backbone("bioclip25", "cpu", model_snapshot=snapshot)
    report["seconds"]["model_load"] = time.perf_counter() - started
    save()
    return embed._torch_batch_fn(model, "cpu"), preprocess


def benchmark(pool: Path, out: Path, *, limit: int = 32, batch: int = 1, threads: int = 4):
    if batch <= 0 or threads <= 0:
        raise ValueError("batch size and CPU thread count must be positive")
    source = pd.read_csv(pool) if pool.suffix.lower() == ".csv" else pd.read_parquet(pool)
    rows = sample_pool(source, limit)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / (out.stem + "-assets")
    work.mkdir(exist_ok=True)
    report = {
        "status": "running",
        "phase": "initializing",
        "model": {"backbone": "bioclip25", "revision": MODEL_REVISION, "device": "cpu"},
        "runtime": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "requested_threads": threads,
            "batch_size": batch,
        },
        "source_sha256": sha256_file(pool),
        "sample": rows[["id", "iconic_taxon", "photo_url"]].to_dict("records"),
        "seconds": {},
        "batches": [],
        "failures": [],
        "cached_photo_count_before": sum(
            embed.photo_path(work, int(row.id), row.photo_url).exists() for row in rows.itertuples()
        ),
        "model_cache_existed_before": (work / "model-cache").exists(),
    }
    started = time.perf_counter()

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_rss_bytes"] = peak_rss_bytes()
        temporary = out.with_suffix(out.suffix + ".partial")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(out)

    save()
    try:
        batch_fn, preprocess = _load_cpu_encoder(work, threads, report, save)
        report["phase"] = "photo_download"
        save()
        phase_started = time.perf_counter()
        staged = embed.stage_images(rows, work, workers=4, timeout=20, retries=2)
        report["seconds"]["photo_download"] = time.perf_counter() - phase_started
        report["failures"] = [
            {"id": int(obs_id), "error": error} for obs_id, error in staged.attrs.get("failed", [])
        ]
        report["phase"] = "decode_preprocess_inference"
        report["seconds"]["decode"] = 0.0
        report["seconds"]["preprocess"] = 0.0
        report["seconds"]["inference"] = 0.0
        save()

        def decode(path):
            phase_started = time.perf_counter()
            try:
                return _decode_image(path)
            finally:
                report["seconds"]["decode"] += time.perf_counter() - phase_started

        def prepare(image):
            phase_started = time.perf_counter()
            try:
                return preprocess(image)
            finally:
                image.close()
                report["seconds"]["preprocess"] += time.perf_counter() - phase_started

        def encode(items):
            report["phase"] = "inference"
            save()
            phase_started = time.perf_counter()
            vectors = batch_fn(items)
            elapsed = time.perf_counter() - phase_started
            report["seconds"]["inference"] += elapsed
            report["batches"].append({"count": len(items), "inference_seconds": elapsed})
            save()
            return vectors

        vectors, kept, failed = embed._embed_paths(
            list(staged.local_path), encode, prepare, batch=batch, open_image=decode
        )
        existing_failures = {item["id"] for item in report["failures"]}
        report["failures"].extend(
            {"id": int(staged.iloc[i].id), "error": error}
            for i, error in failed
            if int(staged.iloc[i].id) not in existing_failures
        )
        report["encoded_count"] = len(kept)
        if not kept:
            raise ValueError("no sample images could be encoded")
        report["phase"] = "persist"
        save()
        phase_started = time.perf_counter()
        artifact = work / "benchmark-vectors.npz"
        np.savez_compressed(artifact, ids=staged.iloc[kept].id.to_numpy(), E=vectors)
        report["seconds"]["persist"] = time.perf_counter() - phase_started
        report["vector_artifact_bytes"] = artifact.stat().st_size
        report["phase"] = "reload"
        save()
        phase_started = time.perf_counter()
        with np.load(artifact, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["E"], vectors)
            np.testing.assert_array_equal(saved["ids"], staged.iloc[kept].id.to_numpy())
        report["seconds"]["reload"] = time.perf_counter() - phase_started
        report["status"] = "complete" if not report["failures"] else "complete_with_failures"
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
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backbone", choices=["bioclip25"], default="bioclip25")
    args = parser.parse_args()
    benchmark(args.pool, args.out, limit=args.limit, batch=args.batch, threads=args.threads)


if __name__ == "__main__":
    main()
