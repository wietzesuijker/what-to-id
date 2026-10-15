"""Execute the atlas's actual public-coordinate filtering logic with Node."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")
LOGIC = Path(__file__).parents[1] / "src/what_to_id/embedding_atlas_logic.js"


def evaluate(expression):
    script = (
        f"const scope=require({json.dumps(str(LOGIC))}); console.log(JSON.stringify({expression}));"
    )
    result = subprocess.run([NODE, "-e", script], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    ("lon", "lat", "bounds", "expected"),
    [
        (-123, 49, [-123, 49, -122, 50], True),
        (-122, 50, [-123, 49, -122, 50], True),
        (-124, 49, [-123, 49, -122, 50], False),
        (-122, 51, [-123, 49, -122, 50], False),
        (179, 0, [170, -10, -170, 10], True),
        (-179, 0, [170, -10, -170, 10], True),
        (0, 0, [170, -10, -170, 10], False),
        (-179, 0, [170, -10, 190, 10], True),
        (0, 0, [-400, -10, 400, 10], True),
        (-123, 49, None, False),
    ],
)
def test_public_coordinate_bounds_including_dateline(lon, lat, bounds, expected):
    assert (
        evaluate(f"scope.inBounds({json.dumps(dict(lon=lon, lat=lat))}, {json.dumps(bounds)})")
        is expected
    )


def test_filter_then_search_then_unfamiliar_order_preserves_record_indices():
    records = [
        dict(id=21, name="Bird beta", lon=-123, lat=49, reference_distance=0.3),
        dict(id=22, name="Bird alpha", lon=-122, lat=49, reference_distance=0.8),
        dict(id=23, name="Bird outside", lon=-100, lat=49, reference_distance=1.2),
        dict(id=24, name="Other taxon", lon=-122, lat=49, reference_distance=0.9),
    ]
    options = dict(
        bounds=[-124, 48, -121, 50],
        onlyInView=True,
        term="bird",
        order="reference",
        hasReference=True,
    )
    result = evaluate(f"scope.matches({json.dumps(records)}, {json.dumps(options)})")
    assert [r["r"]["id"] for r in result] == [22, 21]
    assert [r["i"] for r in result] == [1, 0]
    assert [r["id"] for r in records] == [21, 22, 23, 24]


def test_zero_viewport_matches_and_inactive_filter_recovery():
    records = [dict(id=1, name="Fish", lon=-123, lat=49)]
    options = dict(bounds=[10, 10, 11, 11], onlyInView=True)
    assert evaluate(f"scope.matches({json.dumps(records)}, {json.dumps(options)})") == []
    options["onlyInView"] = False
    assert len(evaluate(f"scope.matches({json.dumps(records)}, {json.dumps(options)})")) == 1


def test_group_changes_do_not_reuse_previous_groups_observations():
    birds = [dict(id=1, name="Bird", lon=-123, lat=49)]
    fish = [dict(id=2, name="Fish", lon=-123, lat=49)]
    options = dict(bounds=[-124, 48, -122, 50], onlyInView=True)
    assert evaluate(f"scope.matches({json.dumps(birds)}, {json.dumps(options)})")[0]["r"]["id"] == 1
    assert evaluate(f"scope.matches({json.dumps(fish)}, {json.dumps(options)})")[0]["r"]["id"] == 2


def test_no_map_bounds_cannot_offer_viewport_matches():
    assert evaluate('scope.matches([{id:1,name:"Bird",lon:0,lat:0}], {onlyInView:true})') == []


@pytest.mark.parametrize("rectangle", [[0, 0, 10, 10], [10, 10, 0, 0]])
def test_brush_geometric_boundaries_and_reverse_drag(rectangle):
    points = [[0, 0], [10, 10], [5, 5], [-0.01, 5], [5, 10.01]]
    assert evaluate(f"scope.brushIndices({json.dumps(points)}, {json.dumps(rectangle)})") == [
        0,
        1,
        2,
    ]


def test_empty_or_cleared_brush_has_no_geometric_members():
    assert evaluate("scope.brushIndices([[5,5]], null)") == []
    assert evaluate("scope.brushIndices([[5,5]], [0,0,1,1])") == []
    assert evaluate("scope.brushIndices([[5,5]], [0,0,NaN,10])") == []


def test_brush_intersects_viewport_search_and_reference_order_then_clear_recovers():
    records = [
        dict(id=1, name="Bird A", lon=-123, lat=49, reference_distance=0.2),
        dict(id=2, name="Bird B", lon=-123, lat=49, reference_distance=0.9),
        dict(id=3, name="Bird outside view", lon=-100, lat=49, reference_distance=1.1),
        dict(id=4, name="Different taxon", lon=-123, lat=49, reference_distance=0.8),
        dict(id=5, name="Bird outside brush", lon=-123, lat=49, reference_distance=1.5),
    ]
    options = dict(
        onlyBrushed=True,
        brushedIndices=[0, 1, 2, 3],
        onlyInView=True,
        bounds=[-124, 48, -122, 50],
        term="Bird",
        order="reference",
        hasReference=True,
    )
    result = evaluate(f"scope.matches({json.dumps(records)}, {json.dumps(options)})")
    assert [x["r"]["id"] for x in result] == [2, 1]
    options["brushedIndices"] = []
    assert evaluate(f"scope.matches({json.dumps(records)}, {json.dumps(options)})") == []
    options["onlyBrushed"] = False
    result = evaluate(f"scope.matches({json.dumps(records)}, {json.dumps(options)})")
    assert [x["r"]["id"] for x in result] == [5, 2, 1]


def test_keyboard_set_membership_matches_brush_scope_without_mutating_input():
    assert evaluate("scope.toggleIndex([0,2], 1)") == [0, 1, 2]
    assert evaluate("scope.toggleIndex([0,2], 2)") == [0]
    assert evaluate("scope.toggleIndex([0], 0)") == []
    assert evaluate(
        "(()=>{const initial=Object.freeze([0]);return scope.toggleIndex(initial,1);})()"
    ) == [0, 1]
    assert evaluate(
        'scope.matches([{id:1,name:"A"},{id:2,name:"B"}], '
        "{onlyBrushed:true, brushedIndices:scope.toggleIndex([],1)}).map(x=>x.r.id)"
    ) == [2]


def test_separate_model_navigation_preserves_group_and_selected_observation():
    assert (
        evaluate(
            'scope.modelHref("dinov3/embeddings.html", '
            '"https://example.org/atlas/embeddings.html?bbox=1,2,3,4#group=Aves&id=1", '
            '"Plantae", 123)'
        )
        == "https://example.org/atlas/dinov3/embeddings.html#group=Plantae&id=123"
    )
    assert (
        evaluate(
            'scope.modelHref("../embeddings.html", '
            '"https://example.org/atlas/dinov3/embeddings.html", "Aves", 456)'
        )
        == "https://example.org/atlas/embeddings.html#group=Aves&id=456"
    )


def test_unknown_cross_model_observation_is_explicitly_reported_not_lost():
    assert evaluate('scope.resolveSelection([{id:11},{id:12}], "99")') == {
        "index": 0,
        "missingRequestedId": 99,
    }
    assert evaluate('scope.resolveSelection([{id:11},{id:12}], "12")') == {
        "index": 1,
        "missingRequestedId": None,
    }
    assert evaluate("scope.resolveSelection([{id:11}], null)") == {
        "index": 0,
        "missingRequestedId": None,
    }


def test_atlas_returns_same_record_and_exact_group_to_sibling_map():
    assert (
        evaluate(
            'scope.mapHref("../map.html", '
            '"https://example.org/dinov3/embeddings.html", "Aves", 123)'
        )
        == "https://example.org/map.html#groups=Aves&record=123"
    )


def neighborhood_payloads(expression):
    setup = """(()=>{const ids=Array.from({length:26},(_,i)=>i+1);
function payload(backbone,order,neighbors){return {group:'Aves',backbone,
 records:order.map(id=>({id,name:'Photo '+id,neighbors:id===1?
 neighbors.map(n=>[order.indexOf(n),backbone==='a'?.1:999]):[]}))};}
const a=payload('a',ids,ids.slice(1,13));
const b=payload('b',[...ids].reverse(),[2,3,4,5,14,15,16,17,18,19,20,21]);
"""
    return evaluate(setup + expression + "})()")


def test_cross_encoder_overlap_uses_id_sets_not_indices_or_raw_scores():
    result = neighborhood_payloads("return scope.compareNeighborhoods(a,b,1);")
    assert result == {
        "status": "comparable",
        "leftCount": 12,
        "rightCount": 12,
        "sharedIds": [2, 3, 4, 5],
        "cohortCount": 26,
        "exhaustive": False,
    }


def test_cross_encoder_comparison_rejects_different_pools_or_groups():
    assert neighborhood_payloads(
        "b.records[0].id=99;return scope.compareNeighborhoods(a,b,1);"
    ) == {"status": "pool_mismatch"}
    assert neighborhood_payloads("b.group='Plantae';return scope.compareNeighborhoods(a,b,1);") == {
        "status": "pool_mismatch"
    }


def test_cross_encoder_missing_query_and_invalid_neighbors_are_explicit():
    assert neighborhood_payloads("return scope.compareNeighborhoods(a,b,99);") == {
        "status": "missing_query",
    }
    assert neighborhood_payloads(
        "a.records[0].neighbors=[[999,.1]];return scope.compareNeighborhoods(a,b,1);"
    ) == {"status": "invalid_neighbors"}


def test_cross_encoder_empty_overlap_is_valid_agreement_result():
    assert (
        neighborhood_payloads("""b.records.find(r=>r.id===1).neighbors=
      Array.from({length:12},(_,i)=>[b.records.findIndex(r=>r.id===i+14),999]);
      return scope.compareNeighborhoods(a,b,1);""")["sharedIds"]
        == []
    )


def test_comparison_gate_rejects_late_selection_and_group_requests():
    result = evaluate("""(()=>{const gate=scope.requestGate(),old=gate.begin(),next=gate.begin();
      const before=[gate.current(old),gate.current(next)];gate.cancel();
      return {before,after:gate.current(next),fresh:gate.current(gate.begin())};})()""")
    assert result == {"before": [False, True], "after": False, "fresh": True}


def test_cross_encoder_requires_full_intended_top_k_list():
    assert neighborhood_payloads(
        "a.records[0].neighbors=a.records[0].neighbors.slice(0,11);"
        "return scope.compareNeighborhoods(a,b,1);"
    ) == {"status": "invalid_neighbors"}
    assert neighborhood_payloads(
        "a.records[0].neighbors[1]=a.records[0].neighbors[0];"
        "return scope.compareNeighborhoods(a,b,1);"
    ) == {"status": "invalid_neighbors"}
    assert neighborhood_payloads(
        "a.records[0].neighbors[0]=[0,0];return scope.compareNeighborhoods(a,b,1);"
    ) == {"status": "invalid_neighbors"}


def test_cross_encoder_small_pool_overlap_is_marked_trivial():
    result = evaluate("""scope.compareNeighborhoods(
      {group:'Aves',backbone:'a',records:[{id:1,neighbors:[[1,.1]]},{id:2}]},
      {group:'Aves',backbone:'b',records:[{id:2},{id:1,neighbors:[[0,999]]}]},1)""")
    assert result["sharedIds"] == [2]
    assert result["exhaustive"] is True
    assert result["leftCount"] == result["rightCount"] == 1
    assert evaluate("""scope.compareNeighborhoods(
      {group:'Aves',backbone:'a',records:[{id:1,neighbors:[]}]},
      {group:'Aves',backbone:'b',records:[{id:1,neighbors:[]}]},1)""") == {
        "status": "insufficient_pool",
    }


def test_representation_rgb_is_same_render_value_and_old_payload_disables_mode():
    assert evaluate("scope.representationColor([0,128,255])") == "rgb(0,128,255)"
    assert evaluate("scope.representationColor([0,Infinity,255])") == "#808080"
    assert evaluate("scope.representationColor([0,256,255])") == "#808080"
    assert evaluate("scope.hasRepresentationColors({records:[{id:1}]})") is False
    assert (
        evaluate(
            "scope.hasRepresentationColors({representation_colors:{variance_retained:.2},records:[{rgb:[0,128,255]}]})"
        )
        is True
    )
    assert evaluate('scope.matches([{id:1,name:"Bird"}],{term:"Bird"}).length') == 1


def test_bounded_marker_hit_prefers_nearest_center_not_render_order():
    assert (
        evaluate("""scope.nearestMarker([
      {index:1,id:101,x:10,y:10},{index:2,id:102,x:4,y:3}],{x:0,y:0})""")
        == 2
    )
    assert (
        evaluate("""scope.nearestMarker([
      {index:1,id:101,x:13,y:0}],{x:0,y:0})""")
        is None
    )
    assert evaluate("scope.nearestMarker([],{x:0,y:0})") is None


def test_marker_hit_radius_boundary_and_stable_id_ties():
    assert (
        evaluate("""scope.nearestMarker([
      {index:1,id:101,x:12,y:0}],{x:0,y:0})""")
        == 1
    )
    assert (
        evaluate("""scope.nearestMarker([
      {index:1,id:102,x:3,y:4},{index:2,id:101,x:-3,y:-4}],{x:0,y:0})""")
        == 2
    )
    assert (
        evaluate("""scope.nearestMarker([
      {index:2,id:101,x:-3,y:-4},{index:1,id:102,x:3,y:4}],{x:0,y:0})""")
        == 2
    )


def test_marker_projection_handles_wrapped_world_copy():
    assert evaluate("scope.nearestLongitude(-123,237)") == 237
    assert evaluate("scope.nearestLongitude(179,-181)") == -181
    assert evaluate("scope.nearestLongitude(-123,-124)") == -123
