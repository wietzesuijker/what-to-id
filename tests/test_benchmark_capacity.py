"""Capacity checkpoints reuse only completed, identity-verified chunks."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from what_to_id import benchmark_capacity as capacity


def setup_benchmark(tmp_path, monkeypatch):
    pool = tmp_path / "pool.csv"
    pd.DataFrame(
        {
            "id": range(1, 9),
            "iconic_taxon": ["Aves", "Plantae"] * 4,
            "photo_url": [f"https://example.test/{i}.jpg" for i in range(1, 9)],
        }
    ).to_csv(pool, index=False)
    calls = {"load": 0, "stage": 0, "inferred": []}

    def loader(work, threads, report, save):
        calls["load"] += 1
        report["seconds"]["model_load"] = 0.0
        save()

        def encode(items):
            calls["inferred"].extend(int(item[0]) for item in items)
            return np.stack(items)

        return encode, lambda image: np.full(1024, image.value, dtype=np.float32)

    def stage(rows, work, **kwargs):
        calls["stage"] += 1
        paths = []
        for row in rows.itertuples():
            path = capacity.embed.photo_path(work, int(row.id), row.photo_url)
            path.parent.mkdir(exist_ok=True)
            path.write_text(str(row.id))
            paths.append(str(path))
        return rows.assign(local_path=paths)

    monkeypatch.setattr(capacity.small, "_load_cpu_encoder", loader)
    monkeypatch.setattr(capacity.embed, "stage_images", stage)
    monkeypatch.setattr(
        capacity.small,
        "_decode_image",
        lambda path: SimpleNamespace(
            value=int(Path(path).read_text()),
            close=lambda: None,
        ),
    )
    return pool, tmp_path / "work", calls


def test_stop_resume_and_cached_rerun_infer_every_id_exactly_once(tmp_path, monkeypatch):
    pool, work, calls = setup_benchmark(tmp_path, monkeypatch)
    first = capacity.benchmark(
        pool, tmp_path / "first.json", work, chunk_size=2, stop_after_chunks=2
    )
    assert first["status"] == "intentionally_stopped"
    assert first["encoded_count"] == 4
    hashes = {p.name: capacity.sha256_file(p) for p in work.glob("chunk-*.npz")}
    resumed = capacity.benchmark(pool, tmp_path / "resume.json", work, chunk_size=2)
    assert resumed["status"] == "complete"
    assert resumed["reused_count"] == 4 and resumed["encoded_count"] == 4
    assert resumed["completed_count"] == 8
    assert sorted(calls["inferred"]) == list(range(1, 9))
    assert len(calls["inferred"]) == 8
    assert all(capacity.sha256_file(work / name) == digest for name, digest in hashes.items())
    cached = capacity.benchmark(pool, tmp_path / "cached.json", work, chunk_size=2)
    assert cached["status"] == "cached"
    assert cached["encoded_count"] == 0 and cached["reused_count"] == 8
    assert cached["batches"] == [] and not cached["model_loaded"]
    assert calls["load"] == 2 and calls["stage"] == 2
    assert json.loads((tmp_path / "cached.json").read_text())["status"] == "cached"


@pytest.mark.parametrize("mutation", ["photo", "shard", "pool", "model"])
def test_identity_or_corruption_rejected_before_model_loading(tmp_path, monkeypatch, mutation):
    pool, work, calls = setup_benchmark(tmp_path, monkeypatch)
    capacity.benchmark(pool, tmp_path / "first.json", work, chunk_size=2, stop_after_chunks=1)
    if mutation == "photo":
        capacity.embed.photo_path(work, 1, "https://example.test/1.jpg").write_text("changed bytes")
    elif mutation == "shard":
        next(work.glob("chunk-*.npz")).write_bytes(b"corrupt")
    elif mutation == "pool":
        rows = pd.read_csv(pool)
        rows.loc[0, "photo_url"] = "https://example.test/replaced.jpg"
        rows.to_csv(pool, index=False)
    else:
        monkeypatch.setattr(capacity, "MODEL_REVISION", "another-model")
    with pytest.raises(ValueError):
        capacity.benchmark(pool, tmp_path / "resume.json", work, chunk_size=2)
    assert calls["load"] == 1


def test_orphan_shard_is_recomputed_without_trusting_partial_commit(tmp_path, monkeypatch):
    pool, work, calls = setup_benchmark(tmp_path, monkeypatch)
    capacity.benchmark(pool, tmp_path / "first.json", work, chunk_size=2, stop_after_chunks=1)
    (work / "chunk-00001.npz").write_bytes(b"orphan")
    report = capacity.benchmark(pool, tmp_path / "resume.json", work, chunk_size=2)
    assert report["completed_count"] == 8
    assert len(calls["inferred"]) == 8
