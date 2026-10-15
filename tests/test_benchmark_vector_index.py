"""Small Faiss operations verify the harness, not full-scale capacity claims."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "benchmarks/benchmark_vector_index.py"
spec = importlib.util.spec_from_file_location("vector_capacity", SCRIPT)
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


def test_capacity_round_trip_append_delete_and_size(tmp_path):
    faiss = pytest.importorskip("faiss")
    target = tmp_path / "result.json"
    index_path = tmp_path / "retained.faiss"
    code = capacity.main(
        [
            "--out",
            str(target),
            "--index-out",
            str(index_path),
            "--count",
            "512",
            "--dimensions",
            "32",
            "--queries",
            "8",
            "--threads",
            "1",
            "--lists",
            "4",
            "--pq",
            "4",
            "--nprobe",
            "4",
            "--train-rows",
            "512",
            "--iterations",
            "2",
            "--append",
            "8",
            "--chunk",
            "31",
        ]
    )
    result = json.loads(target.read_text())
    assert code == 0, result
    assert result["status"] == "complete"
    assert result["indexed_rows"] == 512
    assert result["final_rows"] == 512 + 8 - 64
    assert result["removed_rows"] == 64
    assert faiss.read_index(str(index_path)).ntotal == result["final_rows"]
    assert index_path.stat().st_size == result["final_index_bytes"]
    assert result["initial_full_population_encoding_benchmarked"] is False
    assert result["synthetic"] is True
    assert result["real_sample_quality"] is None
    assert 0 < result["final_index_bytes"] < 1_000_000
    assert all(t >= 0 for t in result["timings_seconds"].values())


def test_real_quality_queries_are_held_out(tmp_path):
    faiss = pytest.importorskip("faiss")
    faiss.omp_set_num_threads(1)
    path = tmp_path / "vectors.npz"
    np.savez(path, E=np.random.default_rng(2).normal(size=(1100, 32)).astype("f4"))
    result = capacity.real_quality(faiss, path, queries=16, pq=4, threads=1)
    assert result["sample_rows"] == result["database_rows"] + result["query_rows"]
    assert result["query_rows"] == 16
    assert 0 <= result["recall_at_12"] <= 1
    assert result["full_population_quality_verified"] is False
    assert len(result["input_sha256"]) == 64


@pytest.mark.parametrize("rows", [np.zeros((2, 3)), [[float("nan"), 1]], [1, 2]])
def test_normalization_rejects_invalid_vectors(rows):
    with pytest.raises(ValueError):
        capacity.normalized(rows)


def test_budget_failure_retains_partial_measurements(tmp_path):
    pytest.importorskip("faiss")
    target = tmp_path / "failed.json"
    code = capacity.main(["--out", str(target), "--max-seconds", "0.000000001"])
    result = json.loads(target.read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert result["error"].startswith("TimeoutError:")
    assert result["peak_rss_bytes"] > 0
