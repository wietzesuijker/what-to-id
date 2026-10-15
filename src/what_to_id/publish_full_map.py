"""Publish the canonical observation map using its public assets, without copying the pool."""

from __future__ import annotations

import argparse
import copy
import json
import re
from html import escape
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

from what_to_id.page_links import page_url

CANONICAL = "https://pollocklab.github.io/what-to-id/map.html"


def https_url(value: str) -> str:
    page_url(value)
    if urlsplit(value).scheme != "https":
        raise ValueError("canonical map and asset URLs must use HTTPS")
    return value


def extract_meta(html: str) -> tuple[dict, int, int]:
    matches = list(re.finditer(r"<script>\s*var META=", html))
    if len(matches) != 1:
        raise ValueError("expected exactly one canonical inline META script")
    start = matches[0].end()
    try:
        meta, length = json.JSONDecoder().raw_decode(html[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("canonical map META is not valid JSON") from exc
    end = start + length
    if not isinstance(meta, dict) or not html[end:].startswith(";var BASEMAPS="):
        raise ValueError("canonical map script has an unsupported format")
    return meta, start, end


def asset_meta(meta: dict, source: str) -> dict:
    source = https_url(source)
    result = copy.deepcopy(meta)
    if (
        not isinstance(result.get("n"), int)
        or result["n"] < 1
        or not isinstance(result.get("shards"), list)
        or sum(shard["n"] for shard in result["shards"]) != result["n"]
    ):
        raise ValueError("canonical map count must match its complete shard manifest")
    for shard in result["shards"]:
        shard["file"] = https_url(urljoin(source, shard["file"]))
    for key in ("taxa", "parks"):
        if result.get(key):
            result[key] = https_url(urljoin(source, result[key]))
    if result.get("projects"):
        result["projects"]["file"] = https_url(urljoin(source, result["projects"]["file"]))
    return result


def restore_html(html: str, source: str, *, sample_url: str | None = None) -> str:
    """Restore the full map by routing to its current canonical HTML and matching assets."""
    source = https_url(source)
    meta, _, _ = extract_meta(html)
    absolute = asset_meta(meta, source)
    literal = json.dumps(absolute, separators=(",", ":")).replace("<", "\\u003c")
    target = json.dumps(source).replace("<", "\\u003c")
    optional = ""
    if sample_url is not None:
        page_url(sample_url)
        optional = (
            f'<p><a id="embedding-experiment" href="{escape(sample_url, quote=True)}">'
            "Embedding experiment (10,000-photo sample)</a></p>"
        )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Full British Columbia observation map</title>"
        "<style>body{font:17px/1.5 system-ui,sans-serif;max-width:46rem;"
        "margin:10vh auto;padding:24px;color:#163448}a{color:#135d86}</style>"
        "<h1>Opening the full observation map</h1>"
        f"<p>The complete published map contains {absolute['n']:,} records needing identification. "
        "The embedding experiment remains a separate sample.</p>"
        f'<p><a id="full-map" href="{escape(source, quote=True)}">Open the full map</a></p>'
        + optional
        + "<script>var META="
        + literal
        + ";var BASEMAPS={};"
        "const target=new URL(" + target + ");"
        "if(location.search)target.search=location.search;target.hash=location.hash;"
        'document.getElementById("full-map").href=target.href;'
        "location.replace(target.href);</script></html>"
    )


def build(source: str, out: Path, *, sample_url: str | None = None) -> None:
    response = requests.get(https_url(source), timeout=45)
    response.raise_for_status()
    html = restore_html(response.text, source, sample_url=sample_url)
    out.mkdir(parents=True, exist_ok=True)
    (out / "map.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=CANONICAL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-url", help="Optional separate embedding experiment page")
    args = parser.parse_args()
    build(args.source, args.out, sample_url=args.sample_url)


if __name__ == "__main__":
    main()
