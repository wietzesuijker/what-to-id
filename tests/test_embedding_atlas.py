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


def test_reference_distance_uses_full_vectors_and_joins_names_by_id(tmp_path):
    pool, cache, vectors = inputs(tmp_path)
    ref_path = tmp_path / "ref.npz"
    references = np.array([[0, 0, 1], [-1, 0, 0]], dtype="f4")
    np.savez(ref_path, ids=np.array([502, 501]), E=references, backbone="test-encoder")
    ref_pool = pd.DataFrame(
        {"id": [501, 502], "iconic_taxon": ["Aves", "Aves"], "taxon_name": ["Opposite", "Vertical"]}
    )
    result = group_payload(pool, cache, "Aves", reference_path=ref_path, reference_pool=ref_pool)
    assert result["reference"]["count"] == 2
    assert result["reference"]["named_count"] == 2
    assert len(result["reference"]["cache_sha256"]) == 64
    for i, row in enumerate(result["records"]):
        expected = vectors[i] @ references.T
        j = np.argmax(expected)
        assert row["reference_distance"] == pytest.approx(1 - expected[j], abs=1e-6)
        assert row["reference_cosine"] == pytest.approx(expected[j], abs=1e-6)
        assert row["closest_reference_id"] == [502, 501][j]
        assert row["closest_reference_name"] == ["Vertical", "Opposite"][j]


@pytest.mark.parametrize("issue", ["label", "dimensions", "empty", "unnamed_encoder"])
def test_incompatible_reference_fails_instead_of_silently_comparing(tmp_path, issue):
    pool, cache, _ = inputs(tmp_path)
    path = tmp_path / "ref.npz"
    vectors = np.array([[1, 0, 0]], dtype="f4")
    label = "test-encoder"
    if issue == "label":
        label = "another-encoder"
    elif issue == "dimensions":
        vectors = np.array([[1, 0]], dtype="f4")
    elif issue == "empty":
        vectors = np.empty((0, 3), dtype="f4")
    elif issue == "unnamed_encoder":
        label = ""
    np.savez(path, ids=np.arange(len(vectors)), E=vectors, backbone=label)
    with pytest.raises(ValueError):
        group_payload(pool, cache, "Aves", reference_path=path)


def test_without_reference_exploration_still_works(tmp_path):
    pool, cache, _ = inputs(tmp_path)
    result = group_payload(pool, cache, "Aves")
    assert result["reference"] is None
    assert all("reference_distance" not in r for r in result["records"])


def test_reference_pool_requires_vectors(tmp_path):
    pool, _, _ = inputs(tmp_path)
    path = tmp_path / "pool.parquet"
    pool.to_parquet(path)
    with pytest.raises(ValueError, match="requires reference embeddings"):
        build(
            path,
            str(tmp_path / "{group}.npz"),
            tmp_path / "site",
            snapshot_date="2026-09-11",
            reference_pool_path=path,
        )


def test_reference_overlap_is_explicit_not_a_false_novelty_result(tmp_path):
    pool, cache, vectors = inputs(tmp_path)
    reference = tmp_path / "reference.npz"
    np.savez(reference, ids=np.array([11]), E=vectors[:1], backbone="test-encoder")
    result = group_payload(pool, cache, "Aves", reference_path=reference)
    assert result["reference"]["candidate_overlap_count"] == 1
    assert result["records"][0]["included_in_reference"] is True
    assert result["records"][0]["reference_distance"] == 0
    assert result["records"][1]["included_in_reference"] is False
