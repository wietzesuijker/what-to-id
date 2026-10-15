"""Execute map navigation and shard-aware record lookup with Node."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")
LOGIC = Path(__file__).parents[1] / "src/what_to_id/map_assets/map_links.js"


def evaluate(expression):
    script = (
        f"const links=require({json.dumps(str(LOGIC))});console.log(JSON.stringify({expression}));"
    )
    result = subprocess.run([NODE, "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def test_map_handoff_uses_exact_group_and_observation_not_display_name():
    assert (
        evaluate('links.atlasHref("embeddings.html", "https://example.org/map.html", "Aves", 123)')
        == "https://example.org/embeddings.html#group=Aves&id=123"
    )


def test_missing_record_notice_waits_for_all_shards_and_does_not_substitute():
    assert evaluate("links.resolveRecord(new Uint32Array([1,2]),99,false)") == {
        "index": -1,
        "missing": False,
    }
    assert evaluate("links.resolveRecord(new Uint32Array([1,2,99]),99,true)") == {
        "index": 2,
        "missing": False,
    }
    assert evaluate("links.resolveRecord(new Uint32Array([1,2]),99,true)") == {
        "index": -1,
        "missing": True,
    }
