"""Full-map publication preserves the canonical dataset, without loading its binaries."""

import copy
import json

import pytest

from what_to_id.publish_full_map import asset_meta, extract_meta, restore_html


@pytest.fixture
def canonical():
    meta = {
        "n": 3068186,
        "groups": ["Aves", "Unknown"],
        "shards": [
            {"file": "pool-0.bin?v=abc", "n": 272952},
            {"file": "pool-1.bin?v=abc", "n": 2795234},
        ],
        "taxa": "pool-taxa.bin?v=abc",
        "parks": "bc-parks.bin?v=def",
        "totals": {"obs": [[7295991]], "on": "2026-10-14"},
        "projects": {"file": "pool-projects.bin?v=ghi", "list": [{"id": 86886}]},
    }
    html = (
        "<html><header><h1>Records that need an ID in BC</h1></header>"
        '<b id="count">Loading</b><span id="more"></span><script>var META='
        + json.dumps(meta)
        + ';var BASEMAPS={"light":"https://example.org/style.json"};'
        "window.originalMapViewer=true;</script></html>"
    )
    return meta, html


def test_full_canonical_metadata_preserved_and_matching_live_viewer_is_opened(canonical):
    meta, html = canonical
    before = copy.deepcopy(meta)
    result = restore_html(
        html, "https://example.org/what-to-id/map.html", sample_url="embeddings.html"
    )
    published, _, _ = extract_meta(result)
    expected = asset_meta(meta, "https://example.org/what-to-id/map.html")
    assert published == expected
    assert meta == before
    assert published["n"] == 3068186
    assert sum(published["totals"]["obs"][0]) == 7295991
    assert published["groups"] == ["Aves", "Unknown"]
    assert published["projects"]["list"] == meta["projects"]["list"]
    assert published["shards"][0]["file"] == "https://example.org/what-to-id/pool-0.bin?v=abc"
    assert published["projects"]["file"].endswith("/pool-projects.bin?v=ghi")
    assert "window.originalMapViewer=true;" not in result
    assert "location.replace(target.href)" in result
    assert "Embedding experiment (10,000-photo sample)" in result
    assert "Open the full map" in result
    assert "10000" not in str(published)


def test_relative_asset_queries_survive_and_unsafe_urls_are_rejected(canonical):
    meta, html = canonical
    meta["taxa"] = "../data/taxa.bin?v=abc&x=2"
    result = asset_meta(meta, "https://example.org/maps/map.html")
    assert result["taxa"] == "https://example.org/data/taxa.bin?v=abc&x=2"
    for source in ["http://example.org/map.html", "javascript:alert(1)"]:
        with pytest.raises(ValueError):
            restore_html(html, source)
    with pytest.raises(ValueError):
        restore_html(html, "https://example.org/map.html", sample_url="javascript:alert(1)")
    meta["taxa"] = "javascript:alert(1)"
    with pytest.raises(ValueError):
        asset_meta(meta, "https://example.org/map.html")


def test_malformed_or_partial_metadata_never_creates_fake_full_map(canonical):
    meta, html = canonical
    meta["n"] = 10000
    with pytest.raises(ValueError, match="complete shard"):
        asset_meta(meta, "https://example.org/map.html")
    with pytest.raises(ValueError, match="unsupported format"):
        extract_meta("<script>var META={};alert(1);var BASEMAPS={};</script>")
    with pytest.raises(ValueError, match="valid JSON"):
        extract_meta("<script>var META=(()=>evil())();var BASEMAPS={};</script>")
    result = restore_html(html, "https://example.org/map.html", sample_url='photo.html?x="<test>')
    assert 'href="photo.html?x=&quot;&lt;test&gt;"' in result


def test_redirect_preserves_query_and_hash_including_exact_record(canonical):
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    _, html = canonical
    published = restore_html(html, "https://example.org/map.html")
    script = re.search(r"<script>(.*?)</script>", published, re.S).group(1)
    stub = (
        "const document={getElementById:()=>({})};"
        'const location={search:"?mode=all&x=2",hash:"#record=123&groups=Aves",'
        "replace:url=>console.log(url)};"
    )
    result = subprocess.run([node, "-e", stub + script], check=True, capture_output=True, text=True)
    assert (
        result.stdout.strip() == "https://example.org/map.html?mode=all&x=2#record=123&groups=Aves"
    )
