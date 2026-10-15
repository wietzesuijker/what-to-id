"""A map of the pool: every record as a point, filtered by group, taxon, status flags and date.

The page (map.html) loads MapLibre and deck.gl from a CDN and reads the points from gzipped column
files next to it. The files hold record ids, positions, dates, taxa and flags, sorted by id, and
nothing about lists, so the map cannot tell which list a record sits on. A viewer can also keep
only the records inside a polygon they draw or load from GeoJSON, or inside a BC park
(map_area.js); that runs in the page, and the build only copies the park list (map_parks.py) next
to it.

Each shard (pool-0.bin, pool-1.bin, ...), gzipped, little-endian, n records sorted by id,
columns back to back:
    uint32 id step | uint16 lon | uint16 lat | uint16 obs | uint16 up | uint16 taxon
    | uint8 group | uint8 ids | uint8 flags
Each wider column is stored byte by byte: the low byte of every value, then the next byte, and so
on, which gzip packs tighter. The id step is the id minus the previous record's, and the first
record's is the id itself. lon and lat are quantized over META.bbox; obs (observed) and up
(uploaded) count days from META.day0, with NO_DAY for a missing date; taxon indexes pool-taxa.bin
(gzipped JSON of [latin, common, iNaturalist taxon id, rank] rows, most recorded taxon first; rank
indexes META.ranks) and group META.groups, with 0xFFFF and 0xFF for none. A taxon's rank is the one
its first record by id carries. Built with the taxon tree, each row also names its parent's row and
ancestor rows follow the META.taxa_n record rows (see taxonomy.py); META.presets, the shortcut
chips, is set only when every record taxon has its whole line of ancestors. META.group_ids holds
the iNaturalist taxon of each group that has one, for the Identify link.
ids holds min(IDs, 15) in the low nibble and min(agreements, 15) in the high nibble. flags sets
FLAG_BITS for the flags META.flags names; a pool pulled before those columns existed has none.
Dates before DAY_FLOOR (1900-01-01) are stored at that day, so day0 is never earlier, and the
flags byte then also sets EARLY_OBS (16) or EARLY_UP (32) to say the stored day is a floor, not
the real date; the page shows "before 1900" for it.
Shard 0 holds records observed in the last RECENT_DAYS days of the map (and any without a date), so
the page draws them first; older records follow, newest first, in shards of at most SHARD_MAX.

META.totals, when the build fetched them, holds iNaturalist's count of every BC record with a photo
that needs an ID or is Research Grade, per group and month, by observed ("obs") and uploaded
("up") date: one list per META.groups entry, one count per month from META.totals.m0 to the
month of the last day, with earlier months added into the first. META.totals.only holds the same
counts for each status filter ("introduced", "threatened", "exact"), so the share still shows
under one of them. The page sets the pool against them to show the share still needing an ID.
"on" is the day they were fetched.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import time
from html import escape
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from what_to_id.batches import MAX_URL_LEN
from what_to_id.inat import BC_PLACE_ID, INAT, SLEEP, TIMEOUT, make_session
from what_to_id.map_parks import PARKS_NAME, parks_blob, parks_version
from what_to_id.map_projects import add_projects
from what_to_id.page import ARM_WORDS, group_name
from what_to_id.page_links import page_url
from what_to_id.taxonomy import GROUP_TAXA, NO_RANK, NO_TAXON, PRESETS, load_tree, taxa_table

MAP_NAME = "map.html"
TAXA_NAME = "pool-taxa.bin"
NO_DAY = 0xFFFF
_Q = 0xFFFF
BYTES_PER_RECORD = 17
RECENT_DAYS = 90
SHARD_MAX = 4_000_000
IMPRECISE_M = 1000
DAY_FLOOR = pd.Timestamp("1900-01-01")
EARLY_OBS = 16
EARLY_UP = 32
FLAG_BITS = {"introduced": 1, "threatened": 2, "obscured": 4, "imprecise": 8}
TOTAL_GRADES = "needs_id,research"
TOTAL_FIELDS = {"obs": "observed", "up": "created"}
# iNaturalist's search parameters for the page's status filters; "exact" drops obscured and
# imprecise records, as FLAG_BITS does
TOTAL_ONLY = {
    "introduced": {"introduced": "true"},
    "threatened": {"threatened": "true"},
    "exact": {"obscuration": "none", "acc_below_or_unknown": IMPRECISE_M + 1},
}
_COLS = ("id", "lat", "lon", "observed_on", "created_at", "iconic_taxon")
_ASSETS = files("what_to_id") / "map_assets"
# Inlined in this order; map_taxa.js and map_area.js define the taxon and area filters map.js calls,
# map_step.js the Identify button that steps through an area's batches, map_projects.js the
# project filter, and map_chips.js the click rule for the short-list chips and the selection bar.
MAP_CSS = ("map.css", "map_area.css")
MAP_JS = (
    "map_taxa.js",
    "map_step.js",
    "map_area.js",
    "map_projects.js",
    "map_chips.js",
    "map_links.js",
    "map.js",
)


def shard_name(k: int) -> str:
    return f"pool-{k}.bin"


def encode_points(pool: pd.DataFrame, tree: dict | None = None) -> tuple[dict[str, bytes], dict]:
    """Pack the pool into the gzipped files the page reads, and the META it needs to read them.

    ``tree`` is the taxon tree from taxonomy.load_tree; without it the taxa rows have no parents.
    """
    missing = [c for c in _COLS if c not in pool]
    if missing:
        raise ValueError(f"pool lacks columns {missing}")
    if pool.empty:
        raise ValueError("pool is empty; nothing to map")
    df = pool.sort_values("id").reset_index(drop=True)
    ids = df["id"].to_numpy(dtype=np.int64)
    if ids.min() < 0 or ids.max() > 0xFFFFFFFF:
        raise ValueError("record ids do not fit in uint32")
    lat = df["lat"].to_numpy(dtype=np.float64)
    lon = df["lon"].to_numpy(dtype=np.float64)
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("pool has records without coordinates")
    bbox = [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())]
    obs, up = _dates(df["observed_on"]), _dates(df["created_at"])
    if obs.isna().all():
        raise ValueError("no record has an observed date")
    early_obs, early_up = (obs < DAY_FLOOR).to_numpy(), (up < DAY_FLOOR).to_numpy()
    day0 = max(min(d for d in (obs.min(), up.min()) if pd.notna(d)), DAY_FLOOR)
    obs, up = obs.clip(lower=DAY_FLOOR), up.clip(lower=DAY_FLOOR)
    obs_off, up_off = (obs - day0).dt.days, (up - day0).dt.days
    last = int(max(obs_off.max(), up_off.max()))
    if last >= NO_DAY:
        raise ValueError("dates span more than 65,534 days")

    groups = sorted(df["iconic_taxon"].fillna("Unknown").astype(str).unique())
    gidx = pd.Categorical(df["iconic_taxon"].fillna("Unknown").astype(str), categories=groups)
    taxa, tidx, ranks, complete = taxa_table(df, tree)
    known, flags = _flags(df)
    flags = flags | np.where(early_obs, EARLY_OBS, 0).astype("u1")
    flags = flags | np.where(early_up, EARLY_UP, 0).astype("u1")
    obs_day = obs_off.fillna(NO_DAY).to_numpy().astype("<u2")
    cols = {
        "id": ids.astype("<u4"),
        "lon": _quant(lon, bbox[0], bbox[2]),
        "lat": _quant(lat, bbox[1], bbox[3]),
        "obs": obs_day,
        "up": up_off.fillna(NO_DAY).to_numpy().astype("<u2"),
        "taxon": tidx.astype("<u2"),
        "group": gidx.codes.astype("u1"),
        "ids": (_nibble(df, "ident_count") | _nibble(df, "agree") << 4).astype("u1"),
        "flags": flags,
    }
    # obs_day is NO_DAY for a missing date, so those records count as recent too
    recent = obs_day >= max(last - RECENT_DAYS + 1, 0)
    older = np.flatnonzero(~recent)
    older = older[np.lexsort((ids[older], -obs_day[older].astype(np.int64)))]
    parts = [np.flatnonzero(recent), *np.array_split(older, max(1, -(-len(older) // SHARD_MAX)))]
    out: dict[str, bytes] = {}
    shards = []
    for part in parts:
        if len(part):
            part = np.sort(part)
            name = shard_name(len(shards))
            out[name] = _gz(b"".join(_planes(_step(k, v[part])) for k, v in cols.items()))
            shards.append({"file": name, "n": len(part)})
    out[TAXA_NAME] = _gz(json.dumps(taxa, ensure_ascii=False, separators=(",", ":")).encode())
    h = hashlib.sha256()
    for name in sorted(out):
        h.update(name.encode() + out[name])
    meta = {
        "n": int(len(df)),
        "bbox": bbox,
        "day0": day0.strftime("%Y-%m-%d"),
        "days": last + 1,
        "hist0": _hist_start(obs_off, day0),
        "groups": groups,
        "names": {g: group_name(g) for g in groups},
        "group_ids": {g: GROUP_TAXA[g] for g in groups if g in GROUP_TAXA},
        "ranks": ranks,
        "flags": known,
        "shards": shards,
        "taxa": TAXA_NAME,
        "taxa_n": int(np.unique(tidx[tidx != NO_TAXON]).size),
        "sha256": h.hexdigest(),
    }
    if complete:
        meta["presets"] = PRESETS
    return out, meta


def decode_points(blobs: dict[str, bytes], meta: dict) -> pd.DataFrame:
    """Read the files back, as the page does; for tests and for a QGIS export."""
    frames = [_decode_shard(gzip.decompress(blobs[s["file"]]), int(s["n"])) for s in meta["shards"]]
    raw = {k: np.concatenate([f[k] for f in frames]) for k in frames[0]}
    x0, y0, x1, y1 = meta["bbox"]
    day0 = pd.Timestamp(meta["day0"])
    taxa = json.loads(gzip.decompress(blobs[meta["taxa"]]))

    def date(d: np.ndarray) -> pd.Series:
        return pd.Series(day0 + pd.to_timedelta(np.where(d == NO_DAY, 0, d), "D")).where(
            d != NO_DAY
        )

    def pick(table: list, idx: np.ndarray, none: int) -> list:
        return [None if i == none else table[i] for i in idx]

    names = pick(taxa, raw["taxon"], NO_TAXON)
    out = pd.DataFrame(
        {
            "id": raw["id"].astype(np.int64),
            "lon": x0 + raw["lon"] / _Q * (x1 - x0),
            "lat": y0 + raw["lat"] / _Q * (y1 - y0),
            "observed_on": date(raw["obs"]),
            "uploaded_on": date(raw["up"]),
            "iconic_taxon": np.asarray(meta["groups"], dtype=object)[raw["group"]],
            "taxon_id": pd.array(
                [t and t[2] for t in names],
                dtype="Int64",
            ),
            "taxon_name": [t and t[0] for t in names],
            "common_name": [t and t[1] for t in names],
            "rank": [None if not t or t[3] == NO_RANK else meta["ranks"][t[3]] for t in names],
            "ident_count": raw["ids"] & 15,
            "agree": raw["ids"] >> 4,
        }
    )
    out["observed_before_1900"] = (raw["flags"] & EARLY_OBS) > 0
    out["uploaded_before_1900"] = (raw["flags"] & EARLY_UP) > 0
    for name in meta["flags"]:
        out[name] = (raw["flags"] & FLAG_BITS[name]) > 0
    return out.sort_values("id").reset_index(drop=True)


def _decode_shard(blob: bytes, n: int) -> dict[str, np.ndarray]:
    if len(blob) != BYTES_PER_RECORD * n:
        raise ValueError(f"shard holds {len(blob)} bytes, expected {BYTES_PER_RECORD * n}")
    layout = [("id", "<u4"), ("lon", "<u2"), ("lat", "<u2"), ("obs", "<u2"), ("up", "<u2")]
    layout += [("taxon", "<u2"), ("group", "u1"), ("ids", "u1"), ("flags", "u1")]
    out, off = {}, 0
    for name, dt in layout:
        w = np.dtype(dt).itemsize
        planes = np.frombuffer(blob, "u1", w * n, off).reshape(w, n)
        out[name] = planes.T.copy().view(dt).ravel()
        off += w * n
    out["id"] = np.cumsum(out["id"], dtype=np.int64).astype("<u4")
    return out


def _step(name: str, v: np.ndarray) -> np.ndarray:
    """The id column as steps from the previous id; other columns as they are."""
    return np.diff(v, prepend=0).astype("<u4") if name == "id" else v


def _planes(v: np.ndarray) -> bytes:
    """A column byte by byte: every value's low byte, then the next byte, and so on."""
    return v.view("u1").reshape(-1, v.itemsize).T.tobytes()


def _quant(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = hi - lo or 1.0
    return np.rint((v - lo) / span * _Q).astype("<u2")


def _flags(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """The flag names this pool can fill, and each record's flag byte."""
    bits = np.zeros(len(df), dtype="u1")
    known = []
    for name in ("introduced", "threatened", "obscured"):
        if name in df:
            known.append(name)
            on = df[name].astype("boolean").fillna(False).to_numpy(dtype=bool)
            bits |= np.where(on, FLAG_BITS[name], 0).astype("u1")
    if "pos_acc" in df:
        known.append("imprecise")
        acc = pd.to_numeric(df["pos_acc"], errors="coerce").fillna(0).to_numpy()
        bits |= np.where(acc > IMPRECISE_M, FLAG_BITS["imprecise"], 0).astype("u1")
    return known, bits


def _nibble(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df:
        return np.zeros(len(df), dtype="u1")
    return np.clip(pd.to_numeric(df[col], errors="coerce").fillna(0), 0, 15).to_numpy("u1")


def _hist_start(obs_off: pd.Series, day0: pd.Timestamp) -> int:
    """The day the histogram starts: 1 January of the year holding the earliest 0.5% of records.

    Earlier records still count; the page draws them as one bar before the axis.
    """
    q = day0 + pd.Timedelta(days=int(obs_off.dropna().quantile(0.005)))
    return max(0, (pd.Timestamp(year=q.year, month=1, day=1) - day0).days)


def _gz(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9, mtime=0)


def _dates(col: pd.Series) -> pd.Series:
    """Calendar dates; created_at is UTC, observed_on is the observer's local date."""
    d = pd.to_datetime(col, errors="coerce", utc=True, format="mixed")
    return d.dt.tz_localize(None).dt.normalize()


def fetch_totals(groups: list[str], *, on: str, session=None, sleep: float = SLEEP) -> dict:
    """Monthly counts of every BC record with a photo that needs an ID or is Research Grade.

    One histogram request per group and date field, each covering every month on record, then
    the same again under each status filter in TOTAL_ONLY. The pool's "Unknown" group is
    iNaturalist's ``iconic_taxa=unknown``.
    """
    session = session or make_session()

    def fetch(extra: dict) -> dict:
        out: dict = {}
        for key, field in TOTAL_FIELDS.items():
            out[key] = {}
            for g in groups:
                params = {
                    "place_id": BC_PLACE_ID,
                    "quality_grade": TOTAL_GRADES,
                    "photos": "true",
                    "d1": DAY_FLOOR.strftime("%Y-%m-%d"),
                    "iconic_taxa": "unknown" if g == "Unknown" else g,
                    "date_field": field,
                    "interval": "month",
                    **extra,
                }
                r = session.get(f"{INAT}/histogram", params=params, timeout=TIMEOUT)
                r.raise_for_status()
                out[key][g] = {k[:7]: int(v) for k, v in r.json()["results"]["month"].items()}
                if sleep:
                    time.sleep(sleep)
        return out

    return {"on": on, **fetch({}), "only": {k: fetch(v) for k, v in TOTAL_ONLY.items()}}


def align_totals(raw: dict, meta: dict) -> dict:
    """The fetched totals as the page reads them: per group, a count per month of the map.

    A group the fetch lacks counts zero; months before the map's first fold into it, and months
    after its last day are dropped. A status filter the fetch lacks is left out.
    """
    day0 = pd.Timestamp(meta["day0"])
    last = day0 + pd.Timedelta(days=meta["days"] - 1)
    n = (last.year - day0.year) * 12 + last.month - day0.month + 1

    def rows(counts: dict) -> list[list[int]]:
        out = []
        for g in meta["groups"]:
            row = [0] * n
            for ym, v in counts.get(g, {}).items():
                k = (int(ym[:4]) - day0.year) * 12 + int(ym[5:7]) - day0.month
                if k < n:
                    row[max(k, 0)] += v
            out.append(row)
        return out

    out: dict = {"on": raw["on"], "m0": day0.strftime("%Y-%m")}
    out.update({key: rows(raw[key]) for key in TOTAL_FIELDS})
    out["only"] = {
        f: {key: rows(t[key]) for key in TOTAL_FIELDS} for f, t in raw.get("only", {}).items()
    }
    return out


def over_totals(pool: pd.DataFrame, meta: dict) -> int:
    """How many group-months hold more pool records than their totals.

    The totals are fetched minutes after the pool, so a few records can change grade in between;
    the page caps the share at 100%, and this count says how often it has to.
    """
    t = meta["totals"]
    m0 = pd.Timestamp(t["m0"] + "-01")
    over = 0
    for key, col in (("obs", "observed_on"), ("up", "created_at")):
        d = _dates(pool[col]).clip(lower=DAY_FLOOR)
        k = (d.dt.year - m0.year) * 12 + d.dt.month - m0.month
        g = pool["iconic_taxon"].fillna("Unknown").astype(str)
        counts = pd.DataFrame({"g": g, "k": k}).dropna().groupby(["g", "k"]).size()
        for (grp, month), c in counts.items():
            i, month = meta["groups"].index(grp), int(month)
            if 0 <= month < len(t[key][i]) and c > t[key][i][month]:
                over += 1
    return over


def render_map(
    meta: dict,
    *,
    freeze: str | None,
    back: str = "index.html",
    atlas_url: str | None = None,
) -> str:
    """The map page. META goes inline; the points and names come from the files at load."""
    when = ""
    if freeze:
        f = escape(freeze)
        when = f' on <time id="updated" datetime="{f}" title="{f}">{f}</time>'
    v = meta["sha256"][:12]
    page_meta = {
        **meta,
        "shards": [{**s, "file": f"{s['file']}?v={v}"} for s in meta["shards"]],
        "taxa": f"{meta['taxa']}?v={v}",
        "place_id": BC_PLACE_ID,
        "max_url": MAX_URL_LEN,
        "imprecise_m": IMPRECISE_M,
        "parks": f"{PARKS_NAME}?v={parks_version(parks_blob())}",
    }
    if atlas_url is not None:
        page_meta["atlas_url"] = page_url(atlas_url)
    fill = {
        "TITLE": "Records that need an ID in BC",
        "WHEN": when,
        "BACK": escape(back),
        "PLACE_ID": str(BC_PLACE_ID),
        "MAPLIBRE_CSS": MAPLIBRE_CSS,
        "MAPLIBRE_JS": MAPLIBRE_JS,
        "DECK_JS": DECK_JS,
        "CSS": _assets(MAP_CSS),
        "META": json.dumps(page_meta, sort_keys=True, separators=(",", ":")).replace("</", "<\\/"),
        "BASEMAPS": json.dumps(BASEMAPS, sort_keys=True),
        "JS": _assets(MAP_JS),
    }
    html = (_ASSETS / "map.html").read_text()
    for key, value in fill.items():
        html = html.replace("{{" + key + "}}", value)
    return html


def _assets(names: tuple[str, ...]) -> str:
    return "\n".join((_ASSETS / n).read_text() for n in names)


def write_map(
    out_dir: Path | str,
    pool: pd.DataFrame,
    *,
    freeze: str | None = None,
    totals: dict | None = None,
    tree: dict | None = None,
    projects: Path | None = None,
    atlas_url: str | None = None,
) -> list[Path]:
    """Write map.html and its data; totals from fetch_totals, projects from map_projects."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blobs, meta = encode_points(pool, tree)
    if tree is not None and "presets" not in meta:
        _warn("the taxon tree lacks some of the pool's taxa; the shortcut chips stay hidden")
    if totals is not None:
        meta["totals"] = align_totals(totals, meta)
        over = over_totals(pool, meta)
        if over:
            _warn(f"{over} group-months hold more records than their totals; the page caps them")
    add_projects(blobs, meta, projects)
    html = render_map(meta, freeze=freeze, atlas_url=atlas_url)
    low = html.lower()
    for w in ARM_WORDS:
        if w in low:
            raise ValueError(f"{MAP_NAME}: arm name {w!r} leaked into the map page")
    blobs[PARKS_NAME] = parks_blob()
    for name, blob in blobs.items():
        (out / name).write_bytes(blob)
    (out / MAP_NAME).write_text(html)
    return [out / MAP_NAME, *(out / name for name in blobs)]


MAPLIBRE_JS = "https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"
MAPLIBRE_CSS = "https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css"
DECK_JS = "https://unpkg.com/deck.gl@9.1.14/dist.min.js"
BASEMAPS = {
    "light": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "dark": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
}


def _warn(msg: str) -> None:
    logging.getLogger("what_to_id").warning(msg)
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::warning::{msg}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write map.html and its data from a pool parquet.")
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="site folder")
    ap.add_argument("--freeze", default=None, help="date the pool stands for, YYYY-MM-DD")
    ap.add_argument(
        "--totals",
        action="store_true",
        help="fetch all-record totals from iNaturalist (about 112 requests); on failure the map "
        "is built without them",
    )
    ap.add_argument("--tree", type=Path, help="taxon tree cache from what_to_id.taxonomy")
    ap.add_argument("--projects", type=Path, help="pool-projects.bin, from map_projects")
    ap.add_argument("--atlas-url", help="Optional image-sample explorer for this public map")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pool = pd.read_parquet(a.pool)
    tree = load_tree(a.tree) or None
    if a.tree and tree is None:
        _warn(f"no taxon tree at {a.tree}; building the map without clade search")
    totals = None
    if a.totals:
        groups = sorted(pool["iconic_taxon"].fillna("Unknown").astype(str).unique())
        on = a.freeze or pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
        try:
            totals = fetch_totals(groups, on=on)
        except (requests.RequestException, KeyError, ValueError) as e:
            _warn(f"could not fetch the all-record totals, building the map without them: {e}")
    for path in write_map(
        a.out,
        pool,
        freeze=a.freeze,
        totals=totals,
        tree=tree,
        projects=a.projects,
        atlas_url=a.atlas_url,
    ):
        logging.getLogger("what_to_id").info("wrote %s (%d bytes)", path, path.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
