import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from what_to_id.manifest import Manifest
from what_to_id.page import ARM_WORDS
from what_to_id.page_method import ORDER_DETAIL, ORDER_TEXT, method_section
from what_to_id.page_rotation import (
    ROTATION_JS,
    _rotation_data,
    render_rotation_index,
    write_rotation_site,
)

ARM_NAMES = ["recency", "gap_first", "similarity", "novelty", "surprise"]


def _manifest():
    return Manifest(
        freeze="2026-09-01",
        d1="2026-09-15",
        seed=0,
        batch_size=2,
        arms=["recency", "gap_first"],
        arm_labels={"recency": "B", "gap_first": "A"},
        pool_sha256="0" * 64,
        pool_rows=6,
        design="rotation",
        batches={
            "recency-Aves-000": {
                "arm": "recency",
                "group": "Aves",
                "url": "https://x?id=1,2",
                "ids": [1, 2],
            },
            "recency-Aves-001": {
                "arm": "recency",
                "group": "Aves",
                "url": "https://x?id=5,6",
                "ids": [5, 6],
            },
            "recency-Insecta-000": {
                "arm": "recency",
                "group": "Insecta",
                "url": "https://x?id=3",
                "ids": [3],
            },
            "gap_first-Aves-000": {
                "arm": "gap_first",
                "group": "Aves",
                "url": "https://y",
                "ids": [4],
            },
        },
    )


def _assert_blind(html: str) -> None:
    low = html.lower()
    for name in ARM_NAMES:
        assert name not in low


def test_render_rotation_index_blind_and_minimal():
    html = render_rotation_index(_manifest(), title="What to ID next in BC")
    _assert_blind(html)
    assert "batch_id" not in html
    assert "recency-Aves-000" not in html
    assert "<img" not in html
    assert "Next batch" in html
    assert "Batch 0" in html


def test_render_rotation_index_picked_date_is_a_time_element():
    m = replace(_manifest(), freeze="2026-09-11")
    html = render_rotation_index(m, title="t", map_href="map.html")
    assert '<time id="picked" datetime="2026-09-11" title="2026-09-11">2026-09-11</time>' in html
    assert "in British Columbia, picked <time" in html
    assert "British Columbia on 2026-09-11" not in html
    assert "See today's records on a map</a>" in html
    assert "var STALE_DAYS = 2;" in html
    assert "days > STALE_DAYS" in html


def test_render_rotation_index_no_freeze_has_no_date_sentence():
    html = render_rotation_index(replace(_manifest(), freeze=""), title="t")
    assert "needed an ID" not in html
    assert '<time id="picked"' not in html


def test_render_rotation_index_explains_the_cycle():
    html = render_rotation_index(_manifest(), title="t")
    assert html.count('<span class="n">') == 2
    assert "up to 2 records" in html
    assert "next of 2 lists" in html


def test_render_rotation_index_sets_expectations_and_structure():
    html = render_rotation_index(_manifest(), title="t")
    assert "<noscript>" in html
    assert '<h2 class="rungroup" id="runGroup">' in html
    assert '<h2 class="pickhead">Pick a group</h2>' in html
    assert "Sign in to iNaturalist first" in html
    assert "in this browser only" in html
    assert re.search(r'<p class="undo" id="buildMsg"[^>]*\bhidden\b', html)


def test_every_visit_opens_on_the_picker_with_a_resume_button():
    html = render_rotation_index(_manifest(), title="t")
    assert re.search(r'<p class="resumerow" id="resumeRow"[^>]*\bhidden\b', html)
    assert "'Continue with '" in html
    assert re.search(r"renderPicker\(\);\s*\}\)\(\);", html)
    assert "renderRunner(false); } else { renderPicker(); }" not in html


def test_render_rotation_index_explains_the_method_for_this_builds_orders():
    html = render_rotation_index(_manifest(), title="t")
    assert "How the test works" in html
    assert "Newest first." in html and "Data-poor places first." in html
    assert "Look-alike" not in html and "Unfamiliar" not in html
    _assert_blind(html)


def test_every_order_has_page_words_that_do_not_name_the_arm():
    assert set(ORDER_TEXT) == set(ARM_NAMES)
    for text in ORDER_TEXT.values():
        assert not any(w in text.lower() for w in ARM_WORDS)


def test_render_rotation_index_refuses_an_order_without_page_words():
    m = _manifest()
    m.arms = ["recency", "mystery"]
    with pytest.raises(ValueError, match="mystery"):
        render_rotation_index(m, title="t")


def test_every_order_has_detail_words_that_do_not_name_the_arm():
    assert set(ORDER_DETAIL) == set(ARM_NAMES)
    for text in ORDER_DETAIL.values():
        assert not any(w in text.lower() for w in ARM_WORDS)


def test_pipeline_steps_and_group_labels_come_from_the_build():
    m = _manifest()
    m.batch_size = 7
    how = method_section(m, 2)
    pipe = re.search(r'<div class="pipe".*?</ol></div></div>', how, re.S).group(0)
    assert re.findall(r"<b>([^<]+)</b>", pipe) == [
        "We collect",
        "We split them into 2 lists",
        "Each list gets its own order",
        "You press Next batch",
        "You identify",
        "We count",
    ]
    assert re.findall(r'<i class="when">([^<]+)</i>', pipe) == [
        "Each build",
        "During the blitz",
        "After",
    ]
    assert "up to 7 records" in pipe
    assert "species-level IDs on each list, against newest first, and whether the gap" in pipe
    assert "data-poor places first is judged on IDs weighted by map score" in pipe
    assert "Every day" not in pipe and "Only the order differs" not in pipe
    from what_to_id.page_method import _pipeline

    m.arms = ["recency", "similarity"]
    assert "weighted by map score" not in _pipeline(m, 2)
    # Each stage holds its own steps: 3, 2 and 1.
    stages = re.findall(r'<div class="stage">.*?</ol></div>', pipe, re.S)
    assert [s.count("<li>") for s in stages] == [3, 2, 1]
    _assert_blind(pipe)


def test_method_has_one_summary_numbered_methods_and_folds():
    how = method_section(_manifest(), 2)
    # The workflow and orders are readable without opening the technical sections.
    assert "In short" not in how
    assert '<details class="more"><summary>More detail: ' in how
    assert '<div class="pipe"' in how
    assert re.findall(r"<h3>(\d+)\. ", how) == [str(i) for i in range(1, 12)]
    assert "Why each identifier's batches rotate over the lists" not in how
    assert '<section id="methods-orders"><details class="method" open>' in how
    assert '<section id="methods-analysis"><details class="method">' in how
    assert '<nav class="toc"' not in how
    assert "batches of up to 2 records" in how
    _assert_blind(how)


def test_method_describes_this_builds_assignment():
    m = _manifest()
    assert m.assignment == "stratified"
    seeded = method_section(m, 2)
    assert "a seed shuffles" in seeded and "HMAC" not in seeded
    m.assignment = "keyed"
    keyed = method_section(m, 2)
    assert "HMAC-SHA256" in keyed and "seed shuffles" not in keyed


def test_method_keeps_claims_the_code_does_not_make_off_the_page():
    # Each phrase was on the page once and is false or overstated against the code.
    how = method_section(_manifest(), 2)
    for claim in (
        "nobody had",  # the placebo is before participants got the page; the page is public
        "private build file",  # keyed builds take the letters from the key
        "sit in the same batch",  # arms.py re-sorts groups by map score; cuts cross groups
        "shows only the records",  # Identify also hides records the viewer reviewed
        "Blitz the Gap map",  # the score is where-to-blitz's (cells.py)
        "up to 12 participants",  # 12 counts non-zero differences (analysis.sign_flip_p)
        "is set before",  # the protocol is a draft
    ):
        assert claim not in how, claim
    assert "counts 0" in how and "People with no difference drop out" in how


def test_method_links_point_at_files_and_headings_in_the_repo():
    root = Path(__file__).resolve().parents[1]
    how = method_section(_manifest(), 2)
    urls = re.findall(r'href="https://github\.com/PollockLab/what-to-id/blob/main/([^"]+)"', how)
    assert len(urls) >= 10
    for url in urls:
        path, _, anchor = url.partition("#")
        assert (root / path).is_file(), url
        if anchor:
            heads = re.findall(r"^#+ (.+)$", (root / path).read_text(), re.M)
            slugs = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in heads}
            assert anchor in slugs, url


def test_render_rotation_index_data_has_every_url_once_in_order():
    m = _manifest()
    html = render_rotation_index(m, title="t")
    match = re.search(r"var DATA=(\{.*?\});var GROUP_NAMES", html)
    assert match
    data = json.loads(match.group(1))
    assert data == {
        "B": {"Aves": ["https://x?id=1,2", "https://x?id=5,6"], "Insecta": ["https://x?id=3"]},
        "A": {"Aves": ["https://y"]},
    }
    all_urls = [u for by_group in data.values() for urls in by_group.values() for u in urls]
    assert sorted(all_urls) == sorted(b["url"] for b in m.batches.values())
    assert len(all_urls) == len(set(all_urls))


def test_write_rotation_site_writes_only_index(tmp_path):
    written = write_rotation_site(tmp_path, _manifest())
    names = sorted(p.name for p in written)
    assert names == ["index.html"]
    html = (tmp_path / "index.html").read_text()
    _assert_blind(html)


def test_write_rotation_site_refuses_leaked_arm_name(tmp_path):
    m = _manifest()
    m.arm_labels = {"recency": "recency", "gap_first": "A"}
    with pytest.raises(ValueError, match="leaked"):
        write_rotation_site(tmp_path, m)


@pytest.fixture
def node_available():
    import shutil

    if not shutil.which("node"):
        pytest.skip("node not available")


DRIVER = """
var labels = ["A", "B", "C", "D"];
var data = {
  A: {G: ["A-0", "A-1"]},
  B: {G: ["B-0", "B-1", "B-2"]},
  C: {G: []},
  D: {G: ["D-0"]}
};
var state = {perm: labels, progress: {}, offsets: OFFSETS};
var served = [];
var results = [];
for (var i = 0; i < 12; i++) {
  var res = nextBatch(state, data, "G");
  state = res.state;
  results.push({done: res.done, label: res.label, url: res.url, batchNumber: res.batchNumber});
  if (!res.done) served.push(res.label);
}
console.log(JSON.stringify({results: results, served: served}));
"""


def _drive(tmp_path, offsets):
    """Extract the pure nextBatch() step function and drive it with a small node script."""
    import subprocess

    fn_match = re.search(r"function nextBatch\(state, data, group\) \{.*?\n\}", ROTATION_JS, re.S)
    assert fn_match, "nextBatch function not found in ROTATION_JS"
    script = tmp_path / "drive.js"
    script.write_text(fn_match.group(0) + DRIVER.replace("OFFSETS", json.dumps(offsets)))
    out = subprocess.run(["node", str(script)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_next_batch_js_offsets_deal_disjoint_starts(tmp_path, node_available):
    payload = _drive(tmp_path, {"A": {"G": 1}, "B": {"G": 2}})
    urls = [r["url"] for r in payload["results"] if not r["done"]]
    assert urls == ["A-1", "B-2", "D-0", "A-0", "B-0", "B-1"]
    assert sorted(urls) == ["A-0", "A-1", "B-0", "B-1", "B-2", "D-0"]


def test_next_batch_js_logic(tmp_path, node_available):
    payload = _drive(tmp_path, {})
    results = payload["results"]
    served = payload["served"]

    # exactly sum(counts) = 2+3+0+1 = 6 batches served, then done for the rest
    assert served == ["A", "B", "D", "A", "B", "B"]
    assert [r["done"] for r in results] == [False] * 6 + [True] * 6
    assert all(r["url"] is None for r in results if r["done"])
    assert [r["url"] for r in results if not r["done"]] == [
        "A-0",
        "B-0",
        "D-0",
        "A-1",
        "B-1",
        "B-2",
    ]
    # C, which has no batches in this group, is never visited
    assert "C" not in served
    # each label's pointer counts 0,1,2,... as it is repeatedly chosen
    assert [r["batchNumber"] for r in results if not r["done"]] == [1, 2, 3, 4, 5, 6]
    # once exhausted, done stays true and batchNumber stays at the last served count
    assert all(r["batchNumber"] == 6 for r in results if r["done"])


def test_runner_offers_start_over_with_undo():
    html = render_rotation_index(_manifest(), title="t")
    button = re.search(r'<button[^>]*id="startOver"[^>]*>', html)
    assert button and "Start over" in html
    assert " hidden" not in button.group(0), "Start over must stay visible"
    assert 'id="undoBtn"' in html
    _assert_blind(html)


def test_start_over_js_resets_only_that_group(tmp_path, node_available):
    import subprocess

    fns = [
        re.search(rf"function {name}\(state, [a-z]+(, group)?\) \{{.*?\n\}}", ROTATION_JS, re.S)
        for name in ("nextBatch", "startOver")
    ]
    assert all(fns), "nextBatch or startOver not found in ROTATION_JS"
    script = tmp_path / "start_over.js"
    script.write_text(
        "\n".join(f.group(0) for f in fns)
        + """
var data = {A: {G: ["A-0", "A-1"], H: ["A-h"]}, B: {G: ["B-0"]}};
var state = {perm: ["A", "B"], progress: {}, offsets: {}, build: "d1"};
state = nextBatch(state, data, "G").state;
state = nextBatch(state, data, "G").state;
state = nextBatch(state, data, "H").state;
var before = JSON.stringify(state);
var reset = startOver(state, "G");
var again = nextBatch(reset, data, "G");
console.log(JSON.stringify({same: JSON.stringify(state) === before, reset: reset,
  url: again.url, n: again.batchNumber}));
"""
    )
    out = json.loads(
        subprocess.run(["node", str(script)], capture_output=True, text=True, check=True).stdout
    )
    assert out["same"], "startOver mutated its input"
    assert "G" not in out["reset"]["progress"]
    assert out["reset"]["progress"]["H"]["served"] == 1
    assert out["reset"]["perm"] == ["A", "B"] and out["reset"]["build"] == "d1"
    assert (out["url"], out["n"]) == ("A-0", 1)


def _pure_fns(*names):
    """Top-level pure functions from ROTATION_JS, by name, as one script."""
    found = [re.search(rf"function {n}\([^)]*\) \{{.*?\n\}}", ROTATION_JS, re.S) for n in names]
    assert all(found), f"one of {names} not found in ROTATION_JS"
    return "\n".join(f.group(0) for f in found)


def _node(tmp_path, source):
    import subprocess

    script = tmp_path / "pure.js"
    script.write_text(source)
    out = subprocess.run(["node", str(script)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_group_progress_counts_all_lists_together(tmp_path, node_available):
    out = _node(
        tmp_path,
        _pure_fns("nextBatch", "leftInGroup", "groupProgress")
        + """
var data = {A: {G: ["A-0", "A-1"], H: ["A-h"]}, B: {G: ["B-0"]}, C: {}};
var state = {perm: ["A", "B", "C"], progress: {}, offsets: {}};
var out = [groupProgress(state, data, "G"), groupProgress(state, data, "none")];
state = nextBatch(state, data, "G").state;
out.push(groupProgress(state, data, "G"));
state = nextBatch(nextBatch(nextBatch(state, data, "G").state, data, "G").state, data, "G").state;
out.push(groupProgress(state, data, "G"));
console.log(JSON.stringify(out));
""",
    )
    assert out == [
        {"opened": 0, "left": 3, "total": 3},
        {"opened": 0, "left": 0, "total": 0},
        {"opened": 1, "left": 2, "total": 3},
        {"opened": 3, "left": 0, "total": 3},
    ]


def test_with_last_and_start_over_track_the_open_batch_per_group(tmp_path, node_available):
    out = _node(
        tmp_path,
        _pure_fns("withLast", "startOver")
        + """
var s0 = {perm: ["A"], progress: {G: {served: 1}, H: {served: 1}}};
var s1 = withLast(withLast(s0, "G", "u-g"), "H", "u-h");
var s2 = startOver(s1, "G");
console.log(JSON.stringify({s0: s0, s1: s1.last, s2: s2.last, p2: Object.keys(s2.progress)}));
""",
    )
    assert "last" not in out["s0"], "withLast mutated its input"
    assert out["s1"] == {"G": "u-g", "H": "u-h"}
    assert out["s2"] == {"H": "u-h"} and out["p2"] == ["H"]


def test_runner_is_the_same_for_every_list():
    html = render_rotation_index(_manifest(), title="t")
    runner = re.search(r'<section id="runner".*?</section>', html, re.S).group(0)
    assert 'aria-keyshortcuts="n"' in runner and "<kbd>N</kbd>" in runner
    assert 'id="reopen"' in runner and 'id="pickOther"' in runner
    for label in ("A", "B"):
        assert f'"{label}"' not in runner and f"list {label}" not in runner
    assert "<details" in html and "<summary>How the test works</summary>" in html
    _assert_blind(html)


def test_page_embeds_build_id():
    m = _manifest()
    m.created_at = "2026-11-03T06:00:00Z"
    html = render_rotation_index(m, title="t")
    assert 'var BUILD="2026-11-03T06:00:00Z";' in html


def test_new_build_keeps_cycle_and_resets_batches(tmp_path, node_available):
    import subprocess

    block = re.search(r"  if \(state\.build !== BUILD\) \{.*?\n  \}", ROTATION_JS, re.S)
    assert block, "build reset block not found in ROTATION_JS"
    old = {"perm": ["B", "A"], "progress": {"Aves": {"step": 3}}, "offsets": {"A": {"Aves": 1}}}
    script = tmp_path / "reset.js"
    script.write_text(
        "var out = [];\n"
        f"[['day1', 'day2'], ['day2', 'day2']].forEach(function(p){{\n"
        f"  var state = Object.assign({json.dumps(old)}, {{build: p[0]}}), BUILD = p[1];\n"
        f"{block.group(0)}\n"
        "  out.push(state);\n"
        "});\n"
        "console.log(JSON.stringify(out));\n"
    )
    new_build, same_build = json.loads(
        subprocess.run(["node", str(script)], capture_output=True, text=True, check=True).stdout
    )
    assert new_build == {"perm": ["B", "A"], "progress": {}, "build": "day2"}
    assert same_build["progress"] == old["progress"] and same_build["offsets"] == old["offsets"]


def test_map_link_carries_the_picked_group():
    html = render_rotation_index(_manifest(), title="t", map_href="map.html")
    assert 'id="mapLink" href="map.html" data-base="map.html"' in html
    assert "mapBase + '#groups=' + encodeURIComponent(g)" in html
    assert "setMap(group)" in html and "setMap(null)" in html
    plain = render_rotation_index(_manifest(), title="t")
    assert 'id="mapLink"' not in plain


def test_batches_past_999_keep_their_order():
    batches = {
        f"recency-Aves-{i:03d}": {"arm": "recency", "group": "Aves", "url": f"u{i}", "ids": [i]}
        for i in (1000, 101, 999, 0, 1001, 100)
    }
    m = replace(_manifest(), batches=batches)
    assert _rotation_data(m) == {"B": {"Aves": ["u0", "u100", "u101", "u999", "u1000", "u1001"]}}
