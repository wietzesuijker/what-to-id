"""Outcome read-back for served records, run once the window after blitz end has closed.

``readback`` fetches the current state of served ids, ``outcomes`` joins that state to the
assignment arms and the pool and tallies per-arm shares. ``--dry-run`` skips the fetch and
reports the unengaged share per iconic group from the pool alone. Identifications keep their
timestamps, rank and ``current`` flag, so ``analysis`` can cut them at any date before the fetch.
Each identification row also carries ``observer_id``, the record's observer, so ``analysis`` can
leave out IDs an identifier made on their own record.
``reviewed_by`` lists every user who marked the record reviewed; it carries no timestamp. The
``--assign`` input accepts one or more single-build batches parquets (``id``, ``arm``) or
cumulative served-log parquets (``id``, ``label``, mapped through ``--label-map``);
``analysis.served_arms`` folds them into the union of every served id, one row each.

``outsiders`` is the check on other identifiers, stated before the blitz: per arm, the share of
served records that someone not in ``users`` identified, at any rank and within the window,
before any participant did, and that share minus the control's. Newest first is expected to lose
more records this way. It reads each identification's user and time and each record's observer
(``user_id`` in the obs table). Every identification the observer made on their own record is
left out before choosing who identified first, on both sides: the upload ID falls inside the
window for a record added during the blitz and would otherwise count as an outsider's, and a
participant's IDs on their own record do not count as a participant's. The read-back does not
keep the time a record reached Research Grade, so records that reached it first are not counted
on that ground.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from what_to_id import inat
from what_to_id.analysis import _users, _utc, served_arms

OBS_COLUMNS = (
    "id",
    "user_id",
    "quality_grade",
    "community_rank",
    "taxon_rank",
    "ident_count",
    "n_identifiers",
    "last_ident_at",
    "reviewed_by",
)
IDENT_COLUMNS = ("id", "user_id", "created_at", "taxon_id", "taxon_rank", "current", "observer_id")
OUTCOME_COLUMNS = (
    "arm",
    "n_served",
    "share_species",
    "share_engaged",
    "unengaged",
    "weighted_rg",
)


def summarise(obs: dict) -> dict:
    """Reduce one raw observation to its read-back fields plus its identification list."""
    idents = obs.get("identifications") or []
    taxon = obs.get("taxon") or {}
    community = obs.get("community_taxon") or {}
    flat = [
        {
            "user_id": (i.get("user") or {}).get("id"),
            "created_at": i.get("created_at"),
            "taxon_id": (i.get("taxon") or {}).get("id"),
            "taxon_rank": (i.get("taxon") or {}).get("rank"),
            "current": i.get("current"),
        }
        for i in idents
    ]
    users = {f["user_id"] for f in flat if f["user_id"] is not None}
    stamps = [f["created_at"] for f in flat if f["created_at"]]
    return {
        "id": int(obs["id"]),
        "user_id": (obs.get("user") or {}).get("id"),
        "quality_grade": obs.get("quality_grade"),
        "community_rank": community.get("rank") or None,
        "taxon_rank": taxon.get("rank") if taxon else None,
        "ident_count": int(obs.get("identifications_count") or len(flat)),
        "n_identifiers": len(users),
        "last_ident_at": max(stamps) if stamps else None,
        "reviewed_by": sorted(int(u) for u in obs.get("reviewed_by") or []),
        "identifications": flat,
    }


def readback(
    ids: Sequence[int],
    *,
    fetch: Callable[..., list[dict]] = inat.fetch_by_ids,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch served ids and return (one row per id, long identification table)."""
    ids = [int(i) for i in ids]
    summaries = {s["id"]: s for s in map(summarise, fetch(ids))}
    obs_rows: list[dict] = []
    ident_rows: list[dict] = []
    for i in ids:
        s = summaries.get(i)
        if s is None:
            obs_rows.append({"id": i, **{c: None for c in OBS_COLUMNS[1:]}})
            continue
        obs_rows.append({c: s[c] for c in OBS_COLUMNS})
        ident_rows.extend({"id": i, **f, "observer_id": s["user_id"]} for f in s["identifications"])
    obs_df = pd.DataFrame(obs_rows, columns=list(OBS_COLUMNS))
    for c in ("user_id", "ident_count", "n_identifiers"):
        obs_df[c] = obs_df[c].astype("Int64")
    idents_df = pd.DataFrame(ident_rows, columns=list(IDENT_COLUMNS))
    idents_df["observer_id"] = idents_df["observer_id"].astype("Int64")
    return obs_df, idents_df


def _arms(assign_df: pd.DataFrame | None, pool_df: pd.DataFrame) -> pd.DataFrame:
    if assign_df is None:
        return pd.DataFrame({"id": pool_df["id"].to_numpy(), "arm": "pool"})
    missing = [c for c in ("id", "arm") if c not in assign_df.columns]
    if missing:
        raise ValueError(f"assign_df missing columns {missing}")
    return assign_df[["id", "arm"]].drop_duplicates("id")


def outcomes(
    obs_df: pd.DataFrame | None,
    assign_df: pd.DataFrame | None,
    pool_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-arm outcome shares. ``obs_df=None`` falls back to the pool's own ident counts.

    ``weighted_rg`` uses the served log's ``cell_score`` when ``assign_df`` carries one, so each
    record is weighted by the score that ordered it; otherwise the pool's.
    """
    arms = _arms(assign_df, pool_df)
    scores = assign_df if assign_df is not None and "cell_score" in assign_df.columns else pool_df
    if obs_df is None:
        obs_df = pd.DataFrame(
            {
                "id": pool_df["id"],
                "quality_grade": "needs_id",
                "community_rank": None,
                "ident_count": pool_df["ident_count"],
            }
        )
    df = arms.merge(obs_df, on="id", how="left")
    score_col = "cell_score" if "cell_score" in scores.columns else None
    if score_col:
        df = df.merge(scores[["id", score_col]].drop_duplicates("id"), on="id", how="left")
    ident = pd.to_numeric(df["ident_count"], errors="coerce").fillna(0).astype(float)
    is_rg = df["quality_grade"].eq("research")
    df = df.assign(
        _species=(df["community_rank"].eq("species") | is_rg).astype(float),
        _engaged=(ident >= 2).astype(float),
        _unengaged=(ident <= 1).astype(float),
        _rg_score=np.where(is_rg, df[score_col].fillna(0.0) if score_col else 0.0, 0.0),
    )
    g = df.groupby("arm", sort=True)
    out = pd.DataFrame(
        {
            "n_served": g.size(),
            "share_species": g["_species"].mean(),
            "share_engaged": g["_engaged"].mean(),
            "unengaged": g["_unengaged"].mean(),
            "weighted_rg": g["_rg_score"].sum() if score_col else np.nan,
        }
    ).reset_index()
    return out[list(OUTCOME_COLUMNS)]


def outsiders(
    idents_df: pd.DataFrame,
    assign_df: pd.DataFrame,
    *,
    obs_df: pd.DataFrame,
    users: Sequence[int],
    start,
    cutoff,
    control: str,
) -> pd.DataFrame:
    """Per arm: served records a non-participant identified first in the window, and the share.

    ``obs_df`` gives each record's observer (``id``, ``user_id``). Every identification the
    observer made on their own record is dropped first, so neither side counts it: the upload ID
    does not make a new record an outsider's, and a participant's IDs on their own record are not
    participant IDs. ``diff`` is the arm's share minus the control's: its size and sign give the
    size and direction of the check. Identifications at any rank count, withdrawn ones too.
    """
    missing = [c for c in ("id", "user_id", "created_at") if c not in idents_df.columns]
    if missing:
        raise ValueError(f"idents missing columns {missing}")
    missing = [c for c in ("id", "arm") if c not in assign_df.columns]
    if missing:
        raise ValueError(f"assign_df missing columns {missing}")
    missing = [c for c in ("id", "user_id") if c not in obs_df.columns]
    if missing:
        raise ValueError(f"obs_df missing columns {missing}")
    arms = assign_df[["id", "arm"]].drop_duplicates("id")
    if control not in set(arms["arm"]):
        raise ValueError(f"control arm {control!r} not in {sorted(arms['arm'].unique())}")
    t0, t1 = _utc(start), _utc(cutoff)
    if t1 <= t0:
        raise ValueError(f"cutoff {cutoff} is not after start {start}")
    ts = pd.to_datetime(idents_df["created_at"], utc=True, errors="coerce")
    observer = obs_df.drop_duplicates("id").set_index("id")["user_id"]
    by = pd.to_numeric(idents_df["user_id"], errors="coerce")
    # A record with no known observer has no own IDs to drop, so every ID on it is kept.
    own = by.eq(idents_df["id"].map(pd.to_numeric(observer, errors="coerce")))
    own = own.fillna(False).astype(bool)
    keep = idents_df["id"].isin(arms["id"]) & by.notna() & ~own & ts.ge(t0) & ts.lt(t1)
    win = idents_df.loc[keep, ["id", "user_id"]].assign(ts=ts[keep])
    is_part = win["user_id"].isin({int(u) for u in users})
    first_part = win[is_part].groupby("id")["ts"].min()
    first_out = win[~is_part].groupby("id")["ts"].min()
    part = first_part.reindex(first_out.index)
    lost = set(first_out.index[part.isna() | (first_out < part)])
    df = arms.assign(_lost=arms["id"].isin(lost).astype(float))
    g = df.groupby("arm", sort=True)["_lost"]
    out = pd.DataFrame({"n_served": g.size(), "outsider": g.sum().astype(int), "share": g.mean()})
    out["diff"] = out["share"] - out.loc[control, "share"]
    return out.reset_index()


def to_markdown(outcomes_df: pd.DataFrame) -> str:
    """Simple pipe table, shares as percentages."""
    cols = list(outcomes_df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in outcomes_df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c == "n_served":
                cells.append(str(int(v)))
            elif c in ("share_species", "share_engaged", "unengaged"):
                cells.append(f"{100 * v:.1f}%")
            elif isinstance(v, float):
                cells.append("nan" if np.isnan(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dry_run(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Unengaged share per iconic group from the pool alone, no API calls."""
    assign = pd.DataFrame({"id": pool_df["id"], "arm": pool_df["iconic_taxon"].fillna("unknown")})
    return outcomes(None, assign, pool_df)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read back outcomes for served records.")
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument(
        "--assign",
        type=Path,
        nargs="+",
        help="assignment parquet(s) with id/arm, or served-log(s) with id/label plus --label-map",
    )
    ap.add_argument(
        "--label-map", type=Path, help="JSON {label: arm}, required if --assign files are logs"
    )
    ap.add_argument("--out", type=Path, help="outcomes parquet; idents written next to it")
    ap.add_argument("--dry-run", action="store_true", help="pool only, no fetch")
    ap.add_argument(
        "--users",
        type=Path,
        help="participant user ids; with --start/--cutoff adds the outsider check",
    )
    ap.add_argument("--start", help="blitz start timestamp, for the outsider check")
    ap.add_argument("--cutoff", help="outsider check counts identifications made before this")
    ap.add_argument("--control", default="recency", help="control arm for the outsider check")
    a = ap.parse_args(argv)
    if a.users and not (a.start and a.cutoff):
        ap.error("--users needs --start and --cutoff for the outsider check")
    pool = inat.load_pool(a.pool)
    if a.dry_run:
        res = dry_run(pool)
        print(to_markdown(res))
        return 0
    if a.assign is None or a.out is None:
        ap.error("--assign and --out are required unless --dry-run")
    assign_raw = pd.concat(
        [pd.read_parquet(p, engine="pyarrow") for p in a.assign], ignore_index=True
    )
    label_map = json.loads(a.label_map.read_text()) if a.label_map else None
    assign = served_arms(assign_raw, label_map)
    ids = assign["id"].drop_duplicates().tolist()
    obs_df, idents_df = readback(ids)
    res = outcomes(obs_df, assign, pool)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out, engine="pyarrow", index=False)
    obs_df.to_parquet(a.out.with_name(a.out.stem + "_obs.parquet"), engine="pyarrow", index=False)
    idents_df.to_parquet(
        a.out.with_name(a.out.stem + "_idents.parquet"), engine="pyarrow", index=False
    )
    print(to_markdown(res))
    if a.users:
        kw = {"start": a.start, "cutoff": a.cutoff, "control": a.control}
        check = outsiders(idents_df, assign, obs_df=obs_df, users=_users(a.users), **kw)
        print("outsiders: served records a non-participant identified before any participant")
        print(to_markdown(check))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
