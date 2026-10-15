import gzip
import json
import re
import shutil
import subprocess
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from what_to_id.batches import MAX_URL_LEN
from what_to_id.inat import BC_PLACE_ID
from what_to_id.page import ARM_WORDS
from what_to_id.page_map import (
    _ASSETS,
    BYTES_PER_RECORD,
    IMPRECISE_M,
    MAP_NAME,
    NO_DAY,
    PARKS_NAME,
    RECENT_DAYS,
    TAXA_NAME,
    TOTAL_ONLY,
    _decode_shard,
    align_totals,
    decode_points,
    encode_points,
    fetch_totals,
    main,
    over_totals,
    render_map,
    shard_name,
    write_map,
)

from .conftest import make_pool


def _with_extras(pool):
    n = len(pool)
    pool = pool.copy()
    pool["common_name"] = [None if i % 3 else f"Common {i}" for i in range(n)]
    pool["introduced"] = pd.array([i % 2 == 0 for i in range(n)], dtype="boolean")
    pool["threatened"] = pd.array([None] + [i % 5 == 0 for i in range(1, n)], dtype="boolean")
    pool["obscured"] = pd.array([i % 7 == 0 for i in range(n)], dtype="boolean")
    pool["pos_acc"] = pd.array([None] + [i * 100 for i in range(1, n)], dtype="Int64")
    pool["ident_count"] = np.arange(n) % 20
    pool["agree"] = np.arange(n) % 3
    return pool


def test_encode_round_trips_within_quantization(monkeypatch):
    monkeypatch.setattr("what_to_id.page_map.SHARD_MAX", 40)
    pool = _with_extras(make_pool(300, seed=3)).sample(frac=1, random_state=0)
    pool["rank"] = np.where(pool["taxon_id"] % 2 == 1, "species", "genus")
    pool["id"] = pool["id"] * 1000 + np.where(pool["id"] % 2 == 1, 0xFFFFFFFF - 10**7, 0)
    pool["observed_on"] = (
        pd.Timestamp("2026-08-31") - pd.to_timedelta(np.arange(len(pool)) * 7, "D")
    ).strftime("%Y-%m-%d")
    blobs, meta = encode_points(pool)
    assert len(meta["shards"]) > 3
    assert sum(s["n"] for s in meta["shards"]) == meta["n"] == len(pool)
    for s in meta["shards"]:
        assert len(gzip.decompress(blobs[s["file"]])) == BYTES_PER_RECORD * s["n"]
    back = decode_points(blobs, meta)
    want = pool.sort_values("id").reset_index(drop=True)
    assert back["id"].tolist() == want["id"].tolist()
    x0, y0, x1, y1 = meta["bbox"]
    assert np.abs(back["lon"] - want["lon"]).max() <= (x1 - x0) / 65535
    assert np.abs(back["lat"] - want["lat"]).max() <= (y1 - y0) / 65535
    assert (back["observed_on"] == pd.to_datetime(want["observed_on"])).all()
    up = pd.to_datetime(want["created_at"], utc=True).dt.tz_localize(None).dt.normalize()
    assert (back["uploaded_on"] == up).all()
    assert back["iconic_taxon"].tolist() == want["iconic_taxon"].tolist()
    assert back["taxon_name"].tolist() == want["taxon_name"].tolist()
    assert back["rank"].tolist() == want["rank"].tolist()
    assert back["taxon_id"].tolist() == want["taxon_id"].tolist()
    assert back["ident_count"].tolist() == np.minimum(want["ident_count"], 15).tolist()
    assert back["agree"].tolist() == want["agree"].tolist()
    assert meta["flags"] == ["introduced", "threatened", "obscured", "imprecise"]
    assert back["introduced"].tolist() == want["introduced"].tolist()
    assert back["threatened"].tolist() == want["threatened"].fillna(False).tolist()
    assert back["obscured"].tolist() == want["obscured"].tolist()
    assert back["imprecise"].tolist() == (want["pos_acc"].fillna(0) > 1000).tolist()
    assert meta["sha256"] and meta["groups"] == sorted(pool["iconic_taxon"].unique())


def test_common_names_follow_the_taxon():
    pool = make_pool(4)
    pool["taxon_id"] = pd.array([7, 3, 7, None], dtype="Int64")
    pool["taxon_name"] = ["Alnus", "Acer", "Alnus", None]
    pool["common_name"] = ["alders", None, "alders", None]
    blobs, meta = encode_points(pool)
    assert json.loads(gzip.decompress(blobs[TAXA_NAME])) == [
        ["Alnus", "alders", 7, 0],
        ["Acer", "", 3, 0],
    ]
    back = decode_points(blobs, meta)
    assert back["common_name"].iloc[:3].tolist() == ["alders", "", "alders"]
    assert back["taxon_name"].iloc[:3].tolist() == ["Alnus", "Acer", "Alnus"]
    assert back[["common_name", "taxon_name"]].iloc[3].isna().all()


def test_old_pool_without_extras_has_no_flags():
    blobs, meta = encode_points(make_pool(10))
    assert meta["flags"] == []
    assert "introduced" not in decode_points(blobs, meta)


def test_recent_records_come_first_and_old_ones_after():
    pool = make_pool(6)
    last = pd.Timestamp("2026-09-01")
    edge = (last - pd.Timedelta(days=RECENT_DAYS - 1)).strftime("%Y-%m-%d")
    out = (last - pd.Timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d")
    pool["observed_on"] = ["1987-06-01", out, edge, None, "2026-09-01", "2020-01-01"]
    blobs, meta = encode_points(pool)
    assert [s["file"] for s in meta["shards"]] == [shard_name(0), shard_name(1)]
    assert [s["n"] for s in meta["shards"]] == [3, 3]
    first = decode_points({**blobs}, {**meta, "shards": meta["shards"][:1]})
    assert first["id"].tolist() == pool["id"].iloc[2:5].tolist()
    assert meta["day0"] == "1987-06-01"
    assert decode_points(blobs, meta)["id"].tolist() == pool["id"].tolist()


def test_older_records_follow_newest_first_in_capped_shards(monkeypatch):
    monkeypatch.setattr("what_to_id.page_map.SHARD_MAX", 2)
    pool = make_pool(6)
    pool["observed_on"] = ["2001-01-01", "2005-01-01", "2003-01-01", "2004-01-01", "2002-01-01"] + [
        "2026-09-01"
    ]
    blobs, meta = encode_points(pool)
    assert [s["n"] for s in meta["shards"]] == [1, 2, 2, 1]
    got = [decode_points(blobs, {**meta, "shards": [s]})["id"].tolist() for s in meta["shards"]]
    ids = pool["id"].tolist()
    assert got == [[ids[5]], [ids[1], ids[3]], [ids[2], ids[4]], [ids[0]]]


def test_pool_with_no_recent_records_starts_with_older_ones():
    pool = make_pool(5)
    pool["observed_on"] = "2019-05-01"
    blobs, meta = encode_points(pool)
    assert meta["shards"] == [{"file": shard_name(0), "n": 5}]
    assert sorted(blobs) == [shard_name(0), TAXA_NAME]
    assert decode_points(blobs, meta)["id"].tolist() == pool["id"].tolist()


def test_shards_store_columns_byte_by_byte_and_ids_as_steps():
    pool = make_pool(3)
    pool["id"] = [5, 300, 70000]
    blobs, meta = encode_points(pool)
    raw = gzip.decompress(blobs[shard_name(0)])
    assert raw[:12] == bytes([5, 39, 68, 0, 1, 16, 0, 0, 1, 0, 0, 0])
    assert _decode_shard(raw, 3)["id"].tolist() == [5, 300, 70000]
    empty = _decode_shard(b"", 0)
    assert all(len(v) == 0 for v in empty.values())


def test_histogram_starts_after_rare_old_records():
    pool = make_pool(1000)
    pool.loc[0, "observed_on"] = "1949-05-01"
    _, meta = encode_points(pool)
    assert meta["day0"] == "1949-05-01"
    start = pd.Timestamp(meta["day0"]) + pd.Timedelta(days=meta["hist0"])
    assert start == pd.Timestamp("2026-01-01")


def test_late_upload_and_missing_date():
    pool = make_pool(3)
    pool["observed_on"] = ["2025-01-01", None, "2026-09-01"]
    pool["created_at"] = ["2026-09-20T10:00:00Z", "2026-09-21T10:00:00Z", "2026-09-02T01:00:00Z"]
    blobs, meta = encode_points(pool)
    assert meta["day0"] == "2025-01-01"
    back = decode_points(blobs, meta)
    assert back["observed_on"].isna().tolist() == [False, True, False]
    assert back["uploaded_on"].iloc[0] == pd.Timestamp("2026-09-20")
    n = meta["shards"][0]["n"]
    obs = _decode_shard(gzip.decompress(blobs[shard_name(0)]), n)["obs"]
    assert NO_DAY in obs.tolist()
    assert meta["days"] == (pd.Timestamp("2026-09-21") - pd.Timestamp("2025-01-01")).days + 1


def test_files_are_deterministic():
    pool = _with_extras(make_pool(50))
    assert encode_points(pool) == encode_points(pool.sample(frac=1, random_state=1))


@pytest.mark.parametrize(
    "change, msg",
    [
        (lambda p: p.iloc[0:0], "empty"),
        (lambda p: p.drop(columns="created_at"), "lacks columns"),
        (lambda p: p.assign(lat=np.nan), "without coordinates"),
        (lambda p: p.assign(id=-1), "uint32"),
        (lambda p: p.assign(observed_on=None), "observed date"),
    ],
)
def test_encode_rejects_bad_pools(change, msg):
    with pytest.raises(ValueError, match=msg):
        encode_points(change(make_pool(5)))


def test_decode_rejects_wrong_size():
    blobs, meta = encode_points(make_pool(5))
    cut = gzip.compress(gzip.decompress(blobs[shard_name(0)])[:-1])
    with pytest.raises(ValueError, match="expected"):
        decode_points({**blobs, shard_name(0): cut}, meta)


def test_page_is_blind_and_carries_meta():
    _, meta = encode_points(make_pool(20))
    html = render_map(meta, freeze="2026-09-28")
    low = html.lower()
    assert not [w for w in ARM_WORDS if w in low]
    assert "{{" not in html
    m = re.search(r"var META=(\{.*?\});var BASEMAPS", html)
    page_meta = json.loads(m.group(1))
    v = meta["sha256"][:12]
    assert page_meta["n"] == 20
    assert page_meta["shards"] == [{"file": f"{shard_name(0)}?v={v}", "n": 20}]
    assert page_meta["taxa"] == f"{TAXA_NAME}?v={v}"
    assert f"{BYTES_PER_RECORD}*n" in html and ">2026-09-28</time>" in html


def test_meta_cannot_close_the_script():
    pool = make_pool(3).assign(iconic_taxon="</script><b>")
    _, meta = encode_points(pool)
    assert "</script><b>" not in render_map(meta, freeze=None)


def test_write_map_and_cli(tmp_path):
    pool = make_pool(40)
    paths = write_map(tmp_path / "a", pool)
    names = [MAP_NAME, shard_name(0), TAXA_NAME, PARKS_NAME]
    assert sorted(p.name for p in paths) == sorted(names)
    pool.to_parquet(tmp_path / "pool.parquet", index=False)
    assert main(["--pool", str(tmp_path / "pool.parquet"), "--out", str(tmp_path / "b")]) == 0
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_taxa_file_numbers_taxa_by_frequency_and_carries_their_rank():
    pool = make_pool(6)
    pool["taxon_id"] = pd.array([1700000, 3, 1700000, None, 3, 3], dtype="Int64")
    pool["taxon_name"] = ["Alnus", "Acer rubrum", "Alnus", None, "Acer rubrum", "Acer rubrum"]
    pool["rank"] = ["genus", "species", "genus", "kingdom", "species", "species"]
    blobs, meta = encode_points(pool)
    assert meta["ranks"] == ["genus", "species"]
    assert json.loads(gzip.decompress(blobs[TAXA_NAME])) == [
        ["Acer rubrum", "", 3, 1],
        ["Alnus", "", 1700000, 0],
    ]
    back = decode_points(blobs, meta)
    assert back["taxon_id"].tolist()[:3] == [1700000, 3, 1700000]
    assert back["taxon_id"].isna().tolist() == [False, False, False, True, False, False]
    assert back["rank"].fillna("-").tolist() == ["genus", "species", "genus", "-"] + ["species"] * 2


def test_pool_without_ranks_or_taxa_has_none():
    blobs, meta = encode_points(make_pool(5).drop(columns="rank"))
    assert meta["ranks"] == [] and json.loads(gzip.decompress(blobs[TAXA_NAME]))[0][3] == 255
    assert decode_points(blobs, meta)["rank"].isna().all()
    blobs, meta = encode_points(make_pool(5).drop(columns=["taxon_id", "rank"]))
    assert json.loads(gzip.decompress(blobs[TAXA_NAME])) == []
    assert decode_points(blobs, meta)["taxon_id"].isna().all()


def test_page_passes_the_place_and_url_limit_and_has_the_identify_link():
    _, meta = encode_points(make_pool(20))
    html = render_map(meta, freeze=None)
    page_meta = json.loads(re.search(r"var META=(\{.*?\});var BASEMAPS", html).group(1))
    assert page_meta["place_id"] == BC_PLACE_ID and page_meta["max_url"] == MAX_URL_LEN
    assert page_meta["imprecise_m"] == IMPRECISE_M
    assert f"place_id={BC_PLACE_ID}" in html
    assert 'id="identify"' in html and 'rel="noopener"' in html
    assert "function identifyUrl(" in html
    assert "https://www.inaturalist.org/observations/identify?" in html
    for key in ("quality_grade", "iconic_taxa", "created_d1", "taxon_id", "month"):
        assert key in html
    assert "place_id=7085" not in re.search(r"<script>var META.*", html, re.S).group(0)


def test_page_carries_the_update_date_for_the_age_check():
    _, meta = encode_points(make_pool(20))
    html = render_map(meta, freeze="2026-09-30")
    assert '<time id="updated" datetime="2026-09-30" title="2026-09-30">2026-09-30</time>' in html
    assert "STALE_DAYS" in html
    assert 'id="updated"' not in render_map(meta, freeze=None).split("<script>")[0]


def test_records_before_1900_sit_at_the_floor_and_are_marked():
    pool = make_pool(50)
    pool.loc[0, "observed_on"] = "1850-06-01"
    blobs, meta = encode_points(pool)
    assert meta["day0"] == "1900-01-01"
    back = decode_points(blobs, meta)
    assert len(back) == len(pool)
    old = back[back["id"] == pool.loc[0, "id"]].iloc[0]
    assert old["observed_on"] == pd.Timestamp("1900-01-01") and old["observed_before_1900"]
    assert back["observed_before_1900"].sum() == 1
    assert not back["uploaded_before_1900"].any()


def test_normal_pool_keeps_its_own_day0_and_no_marks():
    blobs, meta = encode_points(make_pool(50))
    assert meta["day0"] > "1900-01-01"
    back = decode_points(blobs, meta)
    assert not back["observed_before_1900"].any()


class _Hist:
    """A stand-in session answering iNaturalist histogram requests from a table."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def get(self, url, params, timeout):
        self.calls.append((url, params))
        flag = next((k for k, v in TOTAL_ONLY.items() if v.items() <= params.items()), None)
        months = self.table.get((params["date_field"], params["iconic_taxa"], flag), {})

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"results": {"month": months}}

        return R()


def test_fetch_totals_asks_once_per_group_date_and_status_filter():
    s = _Hist(
        {
            ("observed", "Aves", None): {"2026-08-01": 7},
            ("created", "unknown", None): {"2026-09-01": 2},
            ("observed", "Aves", "threatened"): {"2026-08-01": 1},
        }
    )
    t = fetch_totals(["Aves", "Unknown"], on="2026-09-30", session=s, sleep=0)
    assert len(s.calls) == 4 * (1 + len(TOTAL_ONLY))
    assert all(u.endswith("/observations/histogram") for u, _ in s.calls)
    p = s.calls[0][1]
    assert p["quality_grade"] == "needs_id,research" and p["photos"] == "true"
    assert p["place_id"] == BC_PLACE_ID and p["interval"] == "month"
    assert t == {
        "on": "2026-09-30",
        "obs": {"Aves": {"2026-08": 7}, "Unknown": {}},
        "up": {"Aves": {}, "Unknown": {"2026-09": 2}},
        "only": {
            "introduced": {"obs": {"Aves": {}, "Unknown": {}}, "up": {"Aves": {}, "Unknown": {}}},
            "threatened": {
                "obs": {"Aves": {"2026-08": 1}, "Unknown": {}},
                "up": {"Aves": {}, "Unknown": {}},
            },
            "exact": {"obs": {"Aves": {}, "Unknown": {}}, "up": {"Aves": {}, "Unknown": {}}},
        },
    }
    exact = [p for _, p in s.calls if "obscuration" in p]
    assert exact and exact[0]["obscuration"] == "none" and exact[0]["acc_below_or_unknown"] == 1001


def test_align_totals_folds_early_months_and_drops_late_ones():
    meta = {"day0": "2026-08-15", "days": 40, "groups": ["Aves", "Fungi"]}
    raw = {
        "on": "2026-09-30",
        "obs": {"Aves": {"2020-01": 3, "2026-08": 5, "2026-09": 4, "2026-11": 9}},
        "up": {"Fungi": {"2026-09": 1}},
    }
    t = align_totals(raw, meta)
    assert t["m0"] == "2026-08" and t["on"] == "2026-09-30"
    assert t["obs"] == [[8, 4], [0, 0]]
    assert t["up"] == [[0, 0], [0, 1]]
    assert t["only"] == {}
    raw["only"] = {"threatened": {"obs": {"Aves": {"2026-09": 1}}, "up": {}}}
    assert align_totals(raw, meta)["only"] == {
        "threatened": {"obs": [[0, 1], [0, 0]], "up": [[0, 0], [0, 0]]}
    }


def test_over_totals_counts_months_the_pool_exceeds():
    pool = make_pool(30)
    _, meta = encode_points(pool)
    big = {g: {"2026-08": 10**6, "2026-09": 10**6} for g in meta["groups"]}
    meta["totals"] = align_totals({"on": "x", "obs": big, "up": big}, meta)
    assert over_totals(pool, meta) == 0
    meta["totals"] = align_totals({"on": "x", "obs": {}, "up": big}, meta)
    assert over_totals(pool, meta) > 0


def test_page_carries_totals_only_when_given(tmp_path):
    pool = make_pool(20)
    write_map(tmp_path / "a", pool)
    html = (tmp_path / "a" / MAP_NAME).read_text()
    assert '"totals"' not in re.search(r"var META=(\{.*?\});var BASEMAPS", html).group(1)
    groups = sorted(pool["iconic_taxon"].unique())
    raw = {"on": "2026-09-30", "obs": {g: {"2026-08": 50} for g in groups}, "up": {}}
    write_map(tmp_path / "b", pool, totals=raw)
    html = (tmp_path / "b" / MAP_NAME).read_text()
    meta = json.loads(re.search(r"var META=(\{.*?\});var BASEMAPS", html).group(1))
    assert meta["totals"]["on"] == "2026-09-30" and len(meta["totals"]["obs"]) == len(groups)
    assert 'id="share"' in html and 'id="hint"' in html
    assert not [w for w in ARM_WORDS if w in html.lower()]


def test_cli_builds_without_totals_when_the_fetch_fails(tmp_path, monkeypatch):
    import requests

    import what_to_id.page_map as pm

    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(pm, "fetch_totals", boom)
    make_pool(20).to_parquet(tmp_path / "pool.parquet", index=False)
    assert main(["--pool", str(tmp_path / "pool.parquet"), "--out", str(tmp_path), "--totals"]) == 0
    assert '"totals"' not in (tmp_path / MAP_NAME).read_text()


def test_page_inlines_the_area_filter_before_the_map_script():
    _, meta = encode_points(make_pool(20))
    html = render_map(meta, freeze=None)
    area = (_ASSETS / "map_area.js").read_text()
    assert area in html and (_ASSETS / "map_area.css").read_text() in html
    assert html.index("var MapArea=") < html.index("var AREA=MapArea(")
    assert "AREA.mask(n)" in html and "h.push(AREA.hash())" in html
    # a park comes back from the hash by its place id or BC Parks number, and its place id goes
    # into the Identify link
    assert "park:S.park,bcpark:S.bcpark," in html and "q.set(k,extra[k])" in html
    assert "parks:META.parks,get:get" in html
    # the Identify box count reads the filters without the area
    assert "if(!side)u=r;return r.url;},match)" in html and "function match(i)" in html
    # the Identify button steps through an area's batches, with a ‹ › beside it to step by hand
    assert html.index("var IdStep=") < html.index("var AREA=MapArea(")
    assert 'id="idprev" class="idarrow" aria-label="Previous batch" hidden' in html
    assert 'id="idnext" class="idarrow" aria-label="Next batch" hidden' in html
    # beside several batches, the box around the shape is offered as one link in a new tab
    assert '<a class="abox" target="_blank" rel="noopener" hidden></a>' in html
    assert "if(k>1)BOX=boxlink(v,url,match,dated,true)" in html


_AREA_JS = r"""
const A = MapArea, out = {};
const r = [[-123.1234567, 49.2, -123, 49.3, -122.9, 49.2, -123.1234567, 49.2]];
out.code = A.encode(r);
out.back = A.decode(out.code);
const bow = A.index([[0, 0, 2, 2, 2, 0, 0, 2]]), donut = A.index([[0, 0, 10, 0, 10, 10, 0, 10],
  [3, 3, 7, 3, 7, 7, 3, 7]]);
out.bow = [[1.5, 1], [0.5, 1], [1, 1.5], [3, 1]].map(p => A.inside(bow, p[0], p[1]));
out.donut = [[1, 1], [5, 5]].map(p => A.inside(donut, p[0], p[1]));
const big = [];
for (let i = 0; i < 5000; i++) { const t = i / 5000 * 2 * Math.PI;
  big.push(-123 + Math.cos(t) * (1 + 0.01 * Math.sin(50 * t)), 50 + Math.sin(t) * 0.7); }
const islets = [];
for (let k = 0; k < 2000; k++) islets.push([-125 + k * 1e-3, 52, -125 + k * 1e-3 + 2e-4, 52,
  -125 + k * 1e-3, 52.0002]);
const s = A.simplify([big, ...islets], 1500);
out.simple = [s.length, s.reduce((a, x) => a + x.length / 2, 0)];
out.errors = [{type: "FeatureCollection", features: [], crs: {type: "name",
  properties: {name: "urn:ogc:def:crs:EPSG::3005"}}},
  {type: "Polygon", coordinates: [[[1200000, 500000], [1, 2], [3, 4]]]},
  {type: "Point", coordinates: [1, 2]}].map(g => { try { A.fromGeoJSON(g); return null; }
    catch (e) { return e.message; } });
out.multi = A.fromGeoJSON({type: "Feature", crs: {type: "name", properties: {name:
  "urn:ogc:def:crs:OGC:1.3:CRS84"}}, geometry: {type: "MultiPolygon", coordinates: [
  [[[0, 0], [1, 0], [1, 1], [0, 0]]], [[[5, 5], [6, 5], [6, 6], [5, 5]]]]}}).length;
try { A.decode("ab!"); } catch (e) { out.bad = e.message; }
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_area_filter_codes_tests_and_reads_polygons():
    js = "var document={};" + (_ASSETS / "map_area.js").read_text() + _AREA_JS
    run = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
    out = json.loads(run.stdout)
    # rounded to 0.0001 degrees, the closing corner dropped, and stable once rounded
    assert out["back"] == [[-123.1235, 49.2, -123, 49.3, -122.9, 49.2]]
    # even-odd: the two lobes of a bow tie are inside, a donut's hole is not
    assert out["bow"] == [1, 1, 0, 0] and out["donut"] == [1, 0]
    # capped at 1500 corners; islets too small to keep drop out and leave their share to the rest
    assert out["simple"][0] == 1 and 1400 <= out["simple"][1] <= 1500
    assert "EPSG:3005" in out["errors"][0] and "not longitude and latitude" in out["errors"][1]
    assert "no Polygon" in out["errors"][2]
    assert out["multi"] == 2 and out["bad"] == "bad character"


_BATCH_PARAMS = {
    "quality_grade": ["needs_id"],
    "reviewed": ["false"],
    "place_id": ["any"],
    "per_page": ["200"],
}


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
@pytest.mark.parametrize(
    "n,sizes", [(0, []), (1, [1]), (200, [200]), (201, [200, 1]), (10000, [200] * 50)]
)
def test_area_identify_batches_open_every_id_once(n, sizes):
    # 10-digit ids, the longest iNaturalist will reach for years, in no order
    ids = [9_999_999_999 - (i * 7919) % 1_000_003 * 1000 - i for i in range(n)]
    js = (
        "var document={};"
        + (_ASSETS / "map_area.js").read_text()
        + f"console.log(JSON.stringify(MapArea.batches({json.dumps(ids)})));"
    )
    # on stdin: 10,000 ids pass Linux's 128 KiB limit on one command-line argument
    run = subprocess.run(["node"], input=js, capture_output=True, text=True, check=True)
    out = json.loads(run.stdout)
    assert [b["n"] for b in out] == sizes
    got = []
    for b in out:
        u = urlparse(b["url"])
        assert u.netloc == "www.inaturalist.org" and u.path == "/observations/identify"
        assert len(b["url"]) < MAX_URL_LEN
        q = parse_qs(u.query)
        batch = [int(i) for i in q.pop("id")[0].split(",")]
        assert q == _BATCH_PARAMS
        # newest first, as Identify lists them, and the title's range is the batch's own
        assert batch == sorted(batch, reverse=True) and len(batch) == b["n"]
        assert (b["lo"], b["hi"]) == (batch[-1], batch[0])
        got += batch
    assert len(got) == len(ids) and set(got) == set(ids)


_STEP_JS = r"""
function el(text) {
  const on = {};
  return {textContent: text, href: "", hidden: false, disabled: false,
    addEventListener: (t, f) => { on[t] = f; }, click: () => on.click({button: 0})};
}
function store(data, broken) {
  const m = data || new Map(), no = () => { throw new Error("storage is off"); };
  if (broken) return () => { no(); };
  return () => ({get length() { return m.size; }, key: i => [...m.keys()][i],
    getItem: k => m.has(k) ? m.get(k) : null, setItem: (k, v) => m.set(k, String(v)),
    removeItem: k => m.delete(k)});
}
function make(build, st) {
  const o = {a: el("Identify these"), prev: el("‹"), next: el("›"), build: build, store: st};
  return Object.assign(IdStep(o), {o: o});
}
const list = k => Array.from({length: k}, (_, i) => ({url: "u" + (i + 1)}));
const tick = () => new Promise(r => setTimeout(r));
(async () => {
  const out = {}, data = new Map();
  let s = make("2026-10-05", store(data));
  out.one = [s.set("a", list(1), "u1"), s.o.a.textContent, s.o.prev.hidden, s.o.next.hidden];
  out.first = [s.set("a", list(3), "u1"), s.o.a.textContent, s.o.prev.disabled, s.o.next.disabled,
    s.o.prev.hidden];
  s.o.next.click(); s.o.next.click(); s.o.next.click();
  out.fwd = [s.at(), s.o.a.href, s.o.next.disabled, s.o.prev.disabled];
  s.o.prev.click();
  out.back = [s.at(), s.o.a.textContent];
  s.go(1);
  // a click opens the batch shown, then moves on once the browser has the href
  s.o.a.click(); out.opening = s.o.a.href; await tick();
  out.opened = [s.at(), s.o.a.textContent, s.o.a.href];
  s.o.a.click(); s.o.a.click(); await tick();
  out.stop = [s.at(), s.o.next.disabled, s.o.a.textContent, s.o.a.href];
  out.other = s.set("b", list(4), "u1") && s.at();
  // back to the first selection, batches 1 to 3 opened: it says so, and a click starts over
  out.again = [s.set("a", list(3), "u1"), s.at(), s.o.a.textContent];
  s.o.a.click(); await tick();
  out.restart = [s.at(), s.o.a.textContent, data.get("idstep:2026-10-05:a")];
  // a new page on the same build resumes at the first batch not opened
  s = make("2026-10-05", store(data));
  s.set("b", list(4), "u1"); s.o.a.click(); s.go(3); s.o.a.click();
  s = make("2026-10-05", store(data));
  out.resume = [s.set("b", list(4), "u1"), s.at()];
  // more records arriving keep that, until the viewer steps
  out.grow = s.set("b", list(6), "u1") && s.at();
  out.keys = [...data.keys()];
  s = make("2026-10-06", store(data));
  out.build = s.set("b", list(4), "u1") && s.at();
  out.pruned = [...data.keys()];
  // storage that throws: the stepper still steps and remembers in memory
  s = make("2026-10-06", store(null, true));
  s.set("c", list(3), "u1"); s.o.a.click(); await tick();
  out.broken = [s.at(), s.o.a.textContent];
  s.set("d", list(3), "u1"); out.brokenOther = s.at();
  s.set("c", list(3), "u1"); out.brokenBack = s.at();
  console.log(JSON.stringify(out));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_identify_button_steps_through_batches_and_remembers_them():
    js = (_ASSETS / "map_step.js").read_text() + _STEP_JS
    run = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
    out = json.loads(run.stdout)
    # one link: the button as it was, no arrows
    assert out["one"] == ["u1", "Identify these", True, True]
    assert out["first"] == ["u1", "Open in Identify · batch 1 of 3", True, False, False]
    # › stops at the last batch, ‹ steps back
    assert out["fwd"] == [[3, 3], "u3", True, False]
    assert out["back"] == [[2, 3], "Open in Identify · batch 2 of 3"]
    assert out["opening"] == "u1"
    assert out["opened"] == [[2, 3], "Open in Identify · batch 2 of 3", "u2"]
    # with every batch opened the button says so and offers batch 1 again
    assert out["stop"] == [[3, 3], True, "All 3 batches opened · start over", "u1"]
    # a new selection starts at batch 1
    assert out["other"] == [1, 4]
    assert out["again"] == ["u1", [3, 3], "All 3 batches opened · start over"]
    assert out["restart"] == [[2, 3], "Open in Identify · batch 2 of 3", "1"]
    assert out["resume"] == ["u2", [2, 4]] and out["grow"] == [2, 6]
    assert out["keys"] == ["idstep:2026-10-05:a", "idstep:2026-10-05:b"]
    # a new build starts over and drops what the old one remembered
    assert out["build"] == [1, 4] and out["pruned"] == []
    assert out["broken"] == [[2, 3], "Open in Identify · batch 2 of 3"]
    assert out["brokenOther"] == [1, 3] and out["brokenBack"] == [2, 3]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_area_box_link_cuts_to_view_counts_and_pages_like_the_batches():
    # records at x = 0..9 on y = 0; odd ones fail the filters
    P = {"n": 10, "pos": [c for i in range(10) for c in (i, 0)]}
    bb = [2, -1, 8, 1]
    js = (
        "var document={};"
        + (_ASSETS / "map_taxa.js").read_text()
        + (_ASSETS / "map_area.js").read_text()
        + f"const P={json.dumps(P)},bb={json.dumps(bb)},ok=i=>i%2===0,A=MapArea;"
        + "const cut=A.boxed({w:0,s:-5,e:5,n:5},bb,P,ok),"
        + "off=A.boxed({w:20,s:20,e:30,n:30},bb,P,ok);"
        + "const st={groups:[],all:3,up:false,d1:'',d2:'',months:[],taxa:null,not:[],only:{}};"
        + "const u=TX.identifyUrl(st,cut.box,{place_id:7085},A.BOXQ);"
        + "console.log(JSON.stringify({cut:cut,off:off,url:u.url}));"
    )
    run = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
    out = json.loads(run.stdout)
    # cut to the view, the box holds x = 2, 4 that pass; with no overlap, the whole box: 2, 4, 6, 8
    assert out["cut"] == {"box": {"w": 2, "s": -1, "e": 5, "n": 1}, "m": 2}
    assert out["off"] == {"box": {"w": 2, "s": -1, "e": 8, "n": 1}, "m": 4}
    q = parse_qs(urlparse(out["url"]).query)
    assert q == {
        "quality_grade": ["needs_id"],
        "place_id": ["7085"],
        "swlat": ["-1.0000"],
        "swlng": ["2.0000"],
        "nelat": ["1.0000"],
        "nelng": ["5.0000"],
        "per_page": ["200"],
        "reviewed": ["false"],
    }


def test_atlas_link_opt_in_preserves_blind_default_and_rejects_unsafe_urls(tmp_path):
    _, meta = encode_points(make_pool(10))
    default = render_map(meta, freeze=None)
    default_meta = json.loads(re.search(r"var META=(\{.*?\});var BASEMAPS", default).group(1))
    assert "atlas_url" not in default_meta
    enabled = render_map(meta, freeze=None, atlas_url="embeddings.html")
    enabled_meta = json.loads(re.search(r"var META=(\{.*?\});var BASEMAPS", enabled).group(1))
    assert enabled_meta["atlas_url"] == "embeddings.html"
    assert not [w for w in ARM_WORDS if w in enabled.lower()]
    write_map(tmp_path, make_pool(10), atlas_url="embeddings.html")
    assert '"atlas_url":"embeddings.html"' in (tmp_path / "map.html").read_text()
    with pytest.raises(ValueError, match="HTTPS"):
        render_map(meta, freeze=None, atlas_url="javascript:alert(1)")
