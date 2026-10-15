"""Data correctness of the public embedding atlas, independent of its projection."""

import json

import numpy as np
import pandas as pd
import pytest

from what_to_id.embedding_atlas import build, group_payload


def inputs(tmp_path):
    vectors = np.array([[1, 0, 0], [0.8, 0.6, 0], [0, 1, 0], [0, 0, 1]], dtype="f4")
    cache = tmp_path / "Aves.npz"
    np.savez(cache, ids=np.array([11, 12, 13, 14]), E=vectors, backbone="test-encoder")
    pool = pd.DataFrame(
        {
            "id": [13, 11, 14, 12],
            "iconic_taxon": ["Aves"] * 4,
            "lon": [-123.0, -122.0, -124.0, -121.0],
            "lat": [49.0, 50.0, 51.0, 52.0],
            "taxon_name": ["C", "A", "D", "B"],
            "photo_url": [""] * 4,
            "observed_on": ["2026-09-11"] * 4,
        }
    )
    return pool, cache, vectors


def test_matches_ids_and_exact_full_space_neighbors(tmp_path):
    pool, cache, vectors = inputs(tmp_path)
    payload = group_payload(pool, cache, "Aves")
    assert [r["name"] for r in payload["records"]] == ["A", "B", "C", "D"]
    assert payload["records"][0]["lon"] == -122
    for i, row in enumerate(payload["records"]):
        actual = row["neighbors"]
        assert i not in [n[0] for n in actual]
        for j, cosine in actual:
            assert cosine == pytest.approx(float(vectors[i] @ vectors[j]), abs=1e-5)
        assert [n[1] for n in actual] == sorted([n[1] for n in actual], reverse=True)
    assert 0 < payload["variance_retained"] <= 1
    assert payload["backbone"] == "test-encoder"
    assert len(payload["cache_sha256"]) == 64


def test_build_writes_real_data_and_no_private_fields(tmp_path):
    pool, cache, _ = inputs(tmp_path)
    pool["private_latitude"] = 88.1234567
    path = tmp_path / "pool.parquet"
    pool.to_parquet(path)
    out = tmp_path / "site"
    build(path, str(tmp_path / "{group}.npz"), out, snapshot_date="2026-09-11")
    html = (out / "embeddings.html").read_text()
    assert "__ATLAS_META__" not in html
    assert "new URLSearchParams(location.search)" in html
    assert "No sampled observations" in html
    text = (out / "embeddings-data/Aves.json").read_text()
    assert "private_latitude" not in text
    assert "88.1234567" not in text
    assert len(json.loads(text)["records"]) == 4


@pytest.mark.parametrize("issue", ["empty", "invalid_coordinate", "duplicate", "missing_cache"])
def test_invalid_inputs_fail_clearly(tmp_path, issue):
    pool, cache, _ = inputs(tmp_path)
    if issue == "empty":
        pool = pool.iloc[:0]
    elif issue == "invalid_coordinate":
        pool.loc[0, "lat"] = np.nan
    elif issue == "duplicate":
        pool.loc[0, "id"] = pool.loc[1, "id"]
    elif issue == "missing_cache":
        cache.unlink()
    path = tmp_path / "pool.parquet"
    pool.to_parquet(path)
    with pytest.raises(ValueError):
        build(path, str(tmp_path / "{group}.npz"), tmp_path / "out", snapshot_date="2026-09-11")


def test_coverage_counts_do_not_imply_complete_spatial_coverage(tmp_path):
    pool, cache, _ = inputs(tmp_path)
    pool.loc[3, "id"] = 99
    payload = group_payload(pool, cache, "Aves")
    assert payload["pool_group_count"] == 4
    assert payload["cache_count"] == 4
    assert payload["matched_count"] == 3
    assert payload["pool_without_vector_count"] == 1
    assert payload["cache_without_pool_count"] == 1
    assert payload["checkpoint_revision"] is None


def test_one_dimensional_cache_and_invalid_snapshot_date(tmp_path):
    pool, cache, _ = inputs(tmp_path)
    np.savez(cache, ids=np.array([11, 12, 13, 14]), E=np.array([[1], [-1], [1], [-1]], dtype="f4"))
    payload = group_payload(pool, cache, "Aves")
    assert all(len(row["xy"]) == 2 for row in payload["records"])
    assert all(row["xy"][1] == 0 for row in payload["records"])
    with pytest.raises(ValueError):
        build(tmp_path / "absent.parquet", "", tmp_path, snapshot_date="unknown")
