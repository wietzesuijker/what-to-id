"""The CPU benchmark's sampling, phase evidence, and failure handling without models."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from what_to_id import benchmark_embeddings as bench


def pool():
    return pd.DataFrame(
        {
            "id": [4, 2, 3, 1],
            "iconic_taxon": ["Aves", "Plantae", "Aves", "Plantae"],
            "photo_url": [f"https://example.test/{i}.jpg" for i in [4, 2, 3, 1]],
        }
    )


def test_sample_is_deterministic_balanced_and_bounded():
    expected = [3, 1, 4]
    assert list(bench.sample_pool(pool(), 3).id) == expected
    assert list(bench.sample_pool(pool().iloc[::-1], 3).id) == expected
    assert len(bench.sample_pool(pool(), 32)) == 4


@pytest.mark.parametrize("column,value", [("id", 4), ("photo_url", None), ("iconic_taxon", "")])
def test_invalid_pool_is_rejected(column, value):
    rows = pool()
    rows.loc[1, column] = value
    with pytest.raises(ValueError):
        bench.sample_pool(rows, 2)


def fake_runtime(monkeypatch, model):
    def load(work, threads, report, save):
        report["phase"] = "model_load"
        report["seconds"]["model_load"] = 0.0
        save()
        return model, lambda image: np.ones(8)

    def stage(rows, work, **kwargs):
        assert kwargs == {"workers": 4, "timeout": 20, "retries": 2}
        return rows.assign(local_path="fake.jpg")

    monkeypatch.setattr(bench, "_load_cpu_encoder", load)
    monkeypatch.setattr(bench.embed, "stage_images", stage)
    monkeypatch.setattr(bench, "_decode_image", lambda path: SimpleNamespace(close=lambda: None))


def test_benchmark_records_batches_and_persists_vectors(tmp_path, monkeypatch):
    fake_runtime(monkeypatch, lambda items: np.ones((len(items), 8)))
    source = tmp_path / "sample.csv"
    pool().to_csv(source, index=False)
    out = tmp_path / "results" / "result.json"
    report = bench.benchmark(source, out, limit=3, batch=2, threads=1)
    assert report["status"] == "complete"
    assert report["encoded_count"] == 3
    assert [batch["count"] for batch in report["batches"]] == [2, 1]
    assert all(batch["inference_seconds"] >= 0 for batch in report["batches"])
    assert set(report["seconds"]) >= {
        "model_load",
        "photo_download",
        "decode",
        "preprocess",
        "inference",
        "persist",
        "reload",
    }
    assert report["peak_rss_bytes"] > 0
    assert report["model"]["device"] == "cpu"
    assert out.is_file()
    assert (out.parent / "result-assets" / "benchmark-vectors.npz").is_file()


def test_failed_batch_keeps_completed_progress_and_no_vectors(tmp_path, monkeypatch):
    calls = []

    def model(items):
        calls.append(len(items))
        if len(calls) == 2:
            raise RuntimeError("simulated model failure")
        return np.ones((len(items), 8))

    fake_runtime(monkeypatch, model)
    source = tmp_path / "sample.parquet"
    pool().to_parquet(source)
    out = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="simulated"):
        bench.benchmark(source, out, batch=1)
    import json

    report = json.loads(out.read_text())
    assert report["status"] == "failed"
    assert report["phase"] == "inference"
    assert len(report["batches"]) == 1
    assert not (tmp_path / "result-assets" / "benchmark-vectors.npz").exists()


def test_invalid_batch_fails_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        bench.benchmark(tmp_path / "missing.csv", tmp_path / "out.json", batch=0)
