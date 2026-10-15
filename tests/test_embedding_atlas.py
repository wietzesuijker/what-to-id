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
    assert "__ATLAS_LOGIC__" not in html
    assert "const AtlasScope" in html
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


def test_rejects_known_cross_group_reference_vector_before_comparing(tmp_path):
    pool, cache, _ = inputs(tmp_path)
    path = tmp_path / "references.npz"
    np.savez(path, ids=np.array([99]), E=np.array([[1, 0, 0]], dtype="f4"), backbone="test-encoder")
    reference_pool = pd.DataFrame(
        {"id": [99], "iconic_taxon": ["Plantae"], "taxon_name": ["Not a bird"]}
    )
    with pytest.raises(ValueError, match="cross-group"):
        group_payload(pool, cache, "Aves", reference_path=path, reference_pool=reference_pool)


def test_cache_only_reference_ids_explicitly_count_as_group_unverified(tmp_path):
    pool, cache, _ = inputs(tmp_path)
    reference_pool = pd.DataFrame({"id": [11], "iconic_taxon": ["Aves"], "taxon_name": ["A"]})
    result = group_payload(pool, cache, "Aves", reference_path=cache, reference_pool=reference_pool)
    assert result["reference"]["group_verified_count"] == 1
    assert result["reference"]["group_unverified_count"] == 3


def test_multigroup_build_requires_distinct_reference_pattern(tmp_path):
    pool, _, _ = inputs(tmp_path)
    pool.loc[0, "iconic_taxon"] = "Plantae"
    path = tmp_path / "pool.parquet"
    pool.to_parquet(path)
    with pytest.raises(ValueError, match=r"require a \{group\}"):
        build(
            path,
            str(tmp_path / "{group}.npz"),
            tmp_path / "site",
            snapshot_date="2026-09-11",
            reference_embedding_pattern=str(tmp_path / "shared.npz"),
        )


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "//evil.example/a",
        "/absolute",
        "https://evil.example/a",
        r"..\evil.html",
        "a\nb.html",
    ],
)
def test_model_navigation_rejects_executable_or_nonrelative_urls(url):
    from what_to_id.embedding_atlas import model_views

    with pytest.raises(ValueError, match="relative"):
        model_views([f"Model={url}"])


def test_model_navigation_metadata_is_optional_and_escaped(tmp_path):
    pool, _, _ = inputs(tmp_path)
    path = tmp_path / "pool.parquet"
    pool.to_parquet(path)
    out = tmp_path / "site"
    label = "Model </script><script>alert(1)</script>"
    build(
        path,
        str(tmp_path / "{group}.npz"),
        out,
        snapshot_date="2026-09-11",
        region_name="British Columbia <test>",
        model_view_specs=[f"{label}=embeddings.html", "Other=dinov3/embeddings.html"],
        current_label=label,
    )
    html = (out / "embeddings.html").read_text()
    assert label not in html
    assert "British Columbia <test>" not in html
    assert "British Columbia \\u003ctest>" in html
    assert '"current_label": "Model \\u003c/script>' in html
    assert "dinov3/embeddings.html" in html
    assert json.loads((out / "embeddings-data/Aves.json").read_text())["reference"] is None


def test_current_model_label_must_match_configured_navigation(tmp_path):
    with pytest.raises(ValueError, match="current model label"):
        build(
            tmp_path / "missing.parquet",
            "",
            tmp_path,
            snapshot_date="2026-09-11",
            model_view_specs=["A=embeddings.html"],
            current_label="B",
        )


def test_three_component_rgb_is_deterministic_clipped_and_display_only(tmp_path):
    from what_to_id.embedding_atlas import representation_projection

    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(80, 5))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    xy, variance2, rgb, metadata = representation_projection(vectors)
    other = representation_projection(vectors)
    np.testing.assert_allclose(xy, other[0], atol=1e-10)
    np.testing.assert_array_equal(rgb, other[2])
    assert xy.shape == (80, 2)
    assert rgb.shape == (80, 3) and rgb.dtype == np.uint8
    assert np.isfinite(rgb).all() and rgb.min() == 0 and rgb.max() == 255
    assert 0 <= variance2 <= metadata["variance_retained"] <= 1 + 1e-12
    assert metadata["percentiles"] == [2, 98] and metadata["clipped"] is True
    pool, cache, _ = inputs(tmp_path)
    payload = group_payload(pool, cache, "Aves")
    assert all(len(row["xy"]) == 2 and len(row["rgb"]) == 3 for row in payload["records"])
    assert all("E" not in row for row in payload["records"])


@pytest.mark.parametrize(
    "vectors,neutral_axes",
    [
        (np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), [2]),
        (np.array([[1.0], [-1.0], [1.0]]), [1, 2]),
        (np.ones((3, 4)), [0, 1, 2]),
        (np.ones((5, 5)), [0, 1, 2]),
        (np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]] * 3), [1, 2]),
        (np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), [1, 2]),
    ],
)
def test_small_and_rank_deficient_representation_has_finite_neutral_channels(vectors, neutral_axes):
    from what_to_id.embedding_atlas import representation_projection

    xy, variance2, rgb, metadata = representation_projection(vectors)
    assert xy.shape == (len(vectors), 2)
    assert np.isfinite(xy).all() and np.isfinite(rgb).all()
    assert np.isfinite(variance2) and np.isfinite(metadata["variance_retained"])
    assert (rgb[:, neutral_axes] == 128).all()


def test_map_first_default_keeps_advanced_controls_and_projection_optional():
    from html.parser import HTMLParser

    class Layout(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.elements = {}

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            ident = attrs.get("id")
            if ident:
                assert ident not in self.elements, f"duplicate control ID: {ident}"
                self.elements[ident] = (attrs, [row[1] for row in self.stack])
            if tag not in {"input", "meta", "link", "img", "br", "hr"}:
                self.stack.append((tag, ident))

        def handle_endtag(self, tag):
            for index in range(len(self.stack) - 1, -1, -1):
                if self.stack[index][0] == tag:
                    del self.stack[index:]
                    break

    from pathlib import Path

    html = (Path(__file__).parents[1] / "src/what_to_id/embedding_atlas.html").read_text()
    layout = Layout()
    layout.feed(html)
    assert "hidden" in layout.elements["projection-pane"][0]
    assert layout.elements["projection-toggle"][0]["aria-expanded"] == "false"
    assert "open" not in layout.elements["advanced"][0]
    for control in ["model-nav", "color-mode", "order", "unfamiliar", "only-view"]:
        assert "advanced" in layout.elements[control][1]
    for control in ["group", "search", "fit", "record"]:
        assert "advanced" not in layout.elements[control][1]
    assert html.index('id="details"') < html.index('id="projection-pane"')
    assert "Canada · Wildlife &amp; plants" in html
    assert "https://wietzesuijker.github.io/bci-dashboard/crownfirst-atlas.html" not in html
    assert 'aria-label="Countries"' not in html
