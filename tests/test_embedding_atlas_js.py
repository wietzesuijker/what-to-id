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
