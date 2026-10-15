"""Gate the bulk GPU queue on exact pilot coverage and CPU vector compatibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from what_to_id.manifest import sha256_file
from what_to_id.prepare_embeddings import MODEL_HASHES, MODEL_REVISION


def verify(manifest: Path, results: Path, reference: Path, photo_hashes: Path) -> dict:
    source = json.loads(manifest.read_text())
    expected_photos = json.loads(photo_hashes.read_text())
    with np.load(reference, allow_pickle=False) as saved:
        ref_ids, ref_vectors = saved["ids"], saved["E"]
    if len(ref_ids) != 512 or len(set(ref_ids.tolist())) != 512:
        raise ValueError("gate requires the complete 512-photo CPU reference")
    if ref_vectors.shape != (512, 1024) or not np.isfinite(ref_vectors).all():
        raise ValueError("invalid CPU reference vectors")
    if not np.allclose(np.linalg.norm(ref_vectors, axis=1), 1, atol=1e-5):
        raise ValueError("CPU vectors are not normalized")
    if set(expected_photos) != {str(value) for value in ref_ids} or not all(
        isinstance(value, str) and len(value) == 64 for value in expected_photos.values()
    ):
        raise ValueError("photo attestations must cover the entire CPU reference")
    ids, vectors, batch_seconds = [], [], 0.0
    for spec in source["shards"]:
        directory = results / f"shard-{spec['index']:05d}"
        state = json.loads((directory / "state.json").read_text())
        identity = state["identity"]
        if (
            state["status"] != "complete"
            or state["failures"]
            or identity["manifest_sha256"] != sha256_file(manifest)
            or identity["shard_input_sha256"] != spec["sha256"]
            or identity["model_revision"] != MODEL_REVISION
            or identity["model_sha256"] != MODEL_HASHES
        ):
            raise ValueError("pilot source/model identity or completion differs")
        path = directory / state["vectors_file"]
        if (
            path.resolve().parent != directory.resolve()
            or sha256_file(path) != state["vectors_sha256"]
        ):
            raise ValueError("pilot vector artifact is unsafe or corrupt")
        with np.load(path, allow_pickle=False) as saved:
            group_ids, group_vectors = saved["ids"], saved["E"]
        if set(group_ids.tolist()) != set(spec["ids"]) or len(group_ids) != spec["count"]:
            raise ValueError("pilot shard coverage differs")
        if group_vectors.shape != (len(group_ids), 1024) or not np.isfinite(group_vectors).all():
            raise ValueError("invalid GPU vectors")
        if not np.allclose(np.linalg.norm(group_vectors, axis=1), 1, atol=1e-5):
            raise ValueError("GPU vectors are not normalized")
        if any(
            state["photos_sha256"].get(str(value)) != expected_photos.get(str(value))
            for value in group_ids
        ):
            raise ValueError("GPU pilot did not use the CPU reference photo bytes")
        ids.extend(group_ids.tolist())
        vectors.extend(group_vectors)
        batch_seconds += sum(batch["elapsed_seconds"] for batch in state["batches"])
    if len(ids) != 512 or set(ids) != set(ref_ids.tolist()):
        raise ValueError("GPU pilot does not cover exactly the 512 CPU reference IDs")
    lookup = dict(zip(ids, vectors, strict=True))
    gpu = np.asarray([lookup[int(value)] for value in ref_ids], dtype=np.float64)
    cpu = ref_vectors.astype(np.float64)
    cosines = np.sum(cpu * gpu, axis=1) / (
        np.linalg.norm(cpu, axis=1) * np.linalg.norm(gpu, axis=1)
    )
    minimum = float(cosines.min())
    if minimum < 0.9999:
        raise ValueError(f"CPU/GPU cosine compatibility below 0.9999: {minimum}")
    throughput = len(ids) / batch_seconds if batch_seconds > 0 else 0
    if throughput < 5:
        raise ValueError(
            f"GPU pilot below five photos/second; benchmark before scaling: {throughput}"
        )
    return {
        "status": "passes",
        "scope": "512 identical-photo CPU/GPU numerical compatibility, not retrieval accuracy",
        "count": len(ids),
        "minimum_cosine": minimum,
        "median_cosine": float(np.median(cosines)),
        "threshold": 0.9999,
        "batch_decode_preprocess_inference_seconds": batch_seconds,
        "photos_per_batch_second": throughput,
        "minimum_photos_per_batch_second": 5,
        "manifest_sha256": sha256_file(manifest),
        "reference_sha256": sha256_file(reference),
        "photo_hashes_sha256": sha256_file(photo_hashes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "results", "reference", "photo-hashes", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(args.manifest, args.results, args.reference, args.photo_hashes)
    except Exception as error:
        args.out.write_text(json.dumps({"status": "fails", "error": str(error)}, indent=2) + "\n")
        raise
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
