"""Frozen primary-photo manifests and bounded, checkpointed embedding workers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import socket
import time
import uuid
from pathlib import Path
from threading import Event
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from what_to_id import embed
from what_to_id.manifest import sha256_file
from what_to_id.prepare_embeddings import MODEL_HASHES, MODEL_REVISION, _verify_snapshot


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    if not {"id", "photo_url"} <= set(rows):
        raise ValueError("primary-photo input requires id and photo_url")
    if rows.empty or rows.id.dtype.kind not in "iu" or (rows.id <= 0).any():
        raise ValueError("primary-photo IDs must be nonempty positive integers")
    if rows.id.max() > np.iinfo(np.int64).max or rows.id.duplicated().any():
        raise ValueError("primary-photo IDs must be unique int64 values")
    if not rows.photo_url.map(
        lambda url: (
            isinstance(url, str)
            and urlsplit(url).scheme in {"http", "https"}
            and bool(urlsplit(url).netloc)
        )
    ).all():
        raise ValueError("each primary-photo row requires a nonempty HTTP(S) photo URL")
    return rows


def prepare(pool: Path, out: Path, shard_size: int = 1024) -> Path:
    if shard_size <= 0:
        raise ValueError("shard size must be positive")
    rows = _rows(pool).sort_values("id", kind="stable")
    if out.exists() and any(out.iterdir()):
        raise ValueError("prepare destination must be empty")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "source_sha256": sha256_file(pool),
        "row_count": len(rows),
        "shard_size": shard_size,
        "backbone": "bioclip25",
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_HASHES,
        "primary_photo_policy": "Exactly the supplied photo_url per observation; no sampling",
        "shards": [],
    }
    for index, start in enumerate(range(0, len(rows), shard_size)):
        part = rows.iloc[start : start + shard_size]
        path = out / f"input-{index:05d}.csv"
        temporary = path.with_suffix(".partial")
        part.to_csv(temporary, index=False)
        temporary.replace(path)
        manifest["shards"].append(
            {
                "index": index,
                "file": path.name,
                "sha256": sha256_file(path),
                "count": len(part),
                "ids": [int(value) for value in part.id],
            }
        )
    path = out / "manifest.json"
    _atomic_json(path, manifest)
    return path


def _runtime(device: str) -> dict:
    packages = {}
    for name in ("torch", "torchvision", "open_clip_torch", "pillow", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"device": device, "packages": packages}


def _validate(ids, vectors, permitted):
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or len(np.unique(ids)) != len(ids)
        or not np.isin(ids, permitted).all()
    ):
        raise ValueError("checkpoint contains invalid or unexpected observation IDs")
    if vectors.shape != (len(ids), 1024) or not np.isfinite(vectors).all():
        raise ValueError("checkpoint requires finite 1024-dimensional vectors")
    if not np.allclose(np.linalg.norm(vectors.astype(np.float64), axis=1), 1, atol=1e-5):
        raise ValueError("checkpoint vectors must have unit norm")


def _load_progress(directory, identity, permitted):
    pointer = directory / "state.json"
    if not pointer.exists():
        return (
            {
                "identity": identity,
                "status": "partial",
                "failures": {},
                "batches": [],
                "photos_sha256": {},
            },
            np.empty(0, dtype=np.int64),
            np.empty((0, 1024), dtype=np.float32),
        )
    state = json.loads(pointer.read_text())
    if state.get("identity") != identity:
        raise ValueError("existing shard belongs to different source/model/runtime inputs")
    ids, vectors = np.empty(0, dtype=np.int64), np.empty((0, 1024), dtype=np.float32)
    if state.get("vectors_file"):
        path = directory / state["vectors_file"]
        if (
            path.parent.resolve() != directory.resolve()
            or sha256_file(path) != state["vectors_sha256"]
        ):
            raise ValueError("checkpoint vector file is unsafe or corrupt")
        with np.load(path, allow_pickle=False) as archive:
            ids, vectors = archive["ids"], archive["E"]
    _validate(ids, vectors, permitted)
    if state.get("completed_count") != len(ids):
        raise ValueError("checkpoint completed count differs from vector rows")
    if state.get("status") == "complete" and (len(ids) != len(permitted) or state.get("failures")):
        raise ValueError("incomplete shard cannot be reused as complete")
    return state, ids, vectors


def _checkpoint(directory, state, ids, vectors, *, complete=False):
    previous = state.get("vectors_file")
    name = "vectors.npz" if complete else f"progress-{uuid.uuid4().hex}.npz"
    path = directory / name
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, ids=ids, E=vectors, backbone="bioclip25")
    temporary.replace(path)
    state.update(
        vectors_file=name,
        vectors_sha256=sha256_file(path),
        completed_count=len(ids),
        status="complete" if complete else "partial",
    )
    _atomic_json(directory / "state.json", state)
    if previous and previous != name:
        (directory / previous).unlink(missing_ok=True)


def run(
    manifest_path: Path,
    shards: list[int],
    model_snapshot: Path,
    work: Path,
    out: Path,
    *,
    device: str = "cuda",
    batch: int = 64,
    offline_photos: bool = False,
    max_seconds: float | None = None,
    _loader=None,
    _clock=time.monotonic,
    _stop_requested=lambda: False,
) -> dict:
    if batch <= 0 or not shards or len(set(shards)) != len(shards):
        raise ValueError("positive batch size and distinct shard indices are required")
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max seconds must be finite and positive")
    if device not in {"cpu", "cuda"}:
        raise ValueError("bulk device must be cpu or cuda")
    started = _clock()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != 1
        or manifest.get("backbone") != "bioclip25"
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("model_sha256") != MODEL_HASHES
    ):
        raise ValueError("manifest does not identify the frozen BioCLIP snapshot")
    verified = _verify_snapshot(model_snapshot)
    source_hashes = {
        "embed.py": sha256_file(Path(embed.__file__)),
        "bulk_embeddings.py": sha256_file(Path(__file__)),
    }
    base_identity = {
        "manifest_sha256": sha256_file(manifest_path),
        "model_revision": MODEL_REVISION,
        "model_sha256": verified,
        "runtime": _runtime(device),
        "encoder_source_sha256": source_hashes,
        "preprocessing": "Frozen open_clip_config.json preprocess_cfg; float32 L2 normalization",
        "batch": batch,
    }
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "completed_shards": [],
        "partial_shards": [],
        "encoded_count": 0,
        "reused_count": 0,
        "model_loaded": False,
        "soft_limit_reached": False,
    }
    batch_fn = preprocess = None
    for index in shards:
        if index < 0 or index >= len(manifest["shards"]):
            raise ValueError(f"shard index outside manifest: {index}")
        spec = manifest["shards"][index]
        source = manifest_path.parent / spec["file"]
        if not source.resolve().is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("shard input path escapes manifest directory")
        if spec["index"] != index or sha256_file(source) != spec["sha256"]:
            raise ValueError("shard input differs from manifest")
        rows = _rows(source)
        if len(rows) != spec["count"] or list(rows.id) != spec["ids"]:
            raise ValueError("shard rows do not match manifest IDs/count")
        directory = out / f"shard-{index:05d}"
        directory.mkdir(exist_ok=True)
        lock = directory / ".writer-lock"
        try:
            lock.mkdir()
        except FileExistsError:
            raise ValueError(
                f"shard {index} already has a writer lock; inspect its owner"
            ) from None
        try:
            _atomic_json(
                lock / "owner.json",
                {
                    "started_UTC_epoch": time.time(),
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                },
            )
            identity = {**base_identity, "shard_index": index, "shard_input_sha256": spec["sha256"]}
            state, ids, vectors = _load_progress(directory, identity, rows.id.to_numpy())
            for row in rows.loc[rows.id.isin(ids)].itertuples():
                photo = embed.photo_path(work, int(row.id), row.photo_url)
                recorded = state.get("photos_sha256", {}).get(str(row.id))
                if not recorded or (photo.exists() and sha256_file(photo) != recorded):
                    raise ValueError(
                        "completed photo has missing attestation or changed cached bytes"
                    )
            result["reused_count"] += len(ids)
            if state["status"] == "complete":
                result["completed_shards"].append(index)
                continue
            todo = rows.loc[~rows.id.isin(ids)]
            for start in range(0, len(todo), batch):
                if _stop_requested() or (
                    max_seconds is not None and _clock() - started >= max_seconds
                ):
                    result["soft_limit_reached"] = True
                    break
                part = todo.iloc[start : start + batch]
                if offline_photos:
                    staged = part.assign(
                        local_path=[
                            str(embed.photo_path(work, int(row.id), row.photo_url))
                            for row in part.itertuples()
                        ]
                    )
                    failures = [
                        (int(row.id), "photo missing from offline cache")
                        for row in staged.itertuples()
                        if not Path(row.local_path).is_file()
                    ]
                else:
                    staged = embed.stage_images(part, work, workers=4, timeout=20, retries=2)
                    failures = staged.attrs.get("failed", [])
                failed_ids = {int(obs_id) for obs_id, _ in failures}
                state["failures"].update({str(obs_id): error for obs_id, error in failures})
                usable = staged.loc[~staged.id.isin(failed_ids)]
                if len(usable):
                    if batch_fn is None:
                        if _loader is not None:
                            model, preprocess = _loader(model_snapshot, device)
                            batch_fn = embed._numpy_batch_fn(model)
                        else:
                            model, preprocess = embed.load_backbone(
                                "bioclip25", device, model_snapshot=model_snapshot
                            )
                            batch_fn = embed._torch_batch_fn(model, device)
                        result["model_loaded"] = True
                    if _stop_requested():
                        result["soft_limit_reached"] = True
                        break
                    batch_started = _clock()
                    try:
                        new_vectors, kept, failed = embed._embed_paths(
                            list(usable.local_path),
                            batch_fn,
                            preprocess,
                            batch=batch,
                            open_image=embed._passthrough_path
                            if _loader is not None
                            else embed._open_image,
                        )
                        if new_vectors.shape[1] != 1024 and len(kept):
                            raise ValueError("frozen BioCLIP output must have 1024 dimensions")
                    except Exception as error:
                        state["failures"].update(
                            {str(value): f"encoder error: {error}" for value in usable.id}
                        )
                        _checkpoint(directory, state, ids, vectors)
                        raise
                    for position, error in failed:
                        if not offline_photos:
                            Path(usable.iloc[position].local_path).unlink(missing_ok=True)
                        state["failures"][str(int(usable.iloc[position].id))] = error
                    succeeded = usable.iloc[kept]
                    if len(succeeded):
                        _validate(succeeded.id.to_numpy(), new_vectors, rows.id.to_numpy())
                        ids = np.concatenate([ids, succeeded.id.to_numpy(dtype=np.int64)])
                        vectors = np.concatenate([vectors, new_vectors])
                        for row in succeeded.itertuples():
                            state["failures"].pop(str(row.id), None)
                            state["photos_sha256"][str(row.id)] = sha256_file(Path(row.local_path))
                        result["encoded_count"] += len(succeeded)
                    state["batches"].append(
                        {
                            "attempted_ids": [int(value) for value in usable.id],
                            "completed_ids": [int(value) for value in succeeded.id],
                            "elapsed_seconds": _clock() - batch_started,
                        }
                    )
                _checkpoint(directory, state, ids, vectors)
            complete = len(ids) == len(rows) and not state["failures"]
            if complete:
                _checkpoint(directory, state, ids, vectors, complete=True)
                result["completed_shards"].append(index)
            else:
                _checkpoint(directory, state, ids, vectors)
                result["partial_shards"].append(index)
            if result["soft_limit_reached"]:
                break
        finally:
            (lock / "owner.json").unlink(missing_ok=True)
            lock.rmdir()
    result["elapsed_seconds"] = _clock() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--pool", type=Path, required=True)
    preparation.add_argument("--out", type=Path, required=True)
    preparation.add_argument("--shard-size", type=int, default=1024)
    worker = commands.add_parser("run")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--shard", type=int, action="append", required=True)
    worker.add_argument("--model-snapshot", type=Path, required=True)
    worker.add_argument("--work-dir", type=Path, required=True)
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    worker.add_argument("--batch", type=int, default=64)
    worker.add_argument("--offline-photos", action="store_true")
    worker.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.pool, args.out, args.shard_size))
    else:
        stop = Event()
        previous_handlers = {}
        for signum in (signal.SIGTERM, signal.SIGUSR1):
            previous_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
        try:
            result = run(
                args.manifest,
                args.shard,
                args.model_snapshot,
                args.work_dir,
                args.out,
                device=args.device,
                batch=args.batch,
                offline_photos=args.offline_photos,
                max_seconds=args.max_seconds,
                _stop_requested=stop.is_set,
            )
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        print(json.dumps(result))
        return 0 if not result["partial_shards"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
