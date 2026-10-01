"""Rest: move to the back records that sat at the front of a list for several builds unidentified.

Each daily build reorders the whole pool and every participant starts each list at its first
batch again, so without this a list would put its hardest records in front of everyone every
day. A record served in the first REST_FRONT batches of its list on REST_AFTER builds in a row
goes behind every other record of its list for the builds within REST_DAYS days of the last of
them, then takes its normal place again and its count starts over. A record at rest is still
served, only later, so every record stays servable. The rule reads only the served log's earlier
builds, batch positions and record ids, never list letters, so it treats every list the same and
a rebuild of a past day finds the same records at rest.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REST_FRONT = 2
REST_AFTER = 3
REST_DAYS = 7


def resting_ids(
    served: pd.DataFrame,
    build_date: str,
    *,
    front: int = REST_FRONT,
    after: int = REST_AFTER,
    days: int = REST_DAYS,
) -> set[int]:
    """Ids at rest on build_date, from the served log's builds before that date."""
    if front < 1 or after < 1 or days < 1:
        raise ValueError("rest needs front >= 1, after >= 1 and days >= 1")
    today = date.fromisoformat(build_date)
    earlier = served[served["build_date"].astype(str) < build_date]
    if earlier.empty:
        return set()
    dates = sorted(earlier["build_date"].astype(str).unique())
    up_front = earlier[earlier["batch_index"] < front]
    ids = np.sort(up_front["id"].astype("int64").unique())
    hit = np.zeros((len(ids), len(dates)), dtype=bool)
    hit[
        np.searchsorted(ids, up_front["id"].astype("int64").to_numpy()),
        np.searchsorted(dates, up_front["build_date"].astype(str).to_numpy()),
    ] = True
    streak = np.zeros(len(ids), dtype=np.int64)
    rest_until = np.full(len(ids), date.min.toordinal(), dtype=np.int64)
    for j, d in enumerate(dates):
        day = date.fromisoformat(d).toordinal()
        resting = day <= rest_until
        streak = np.where(hit[:, j] & ~resting, streak + 1, 0)
        full = streak >= after
        rest_until[full] = day + days
        streak[full] = 0
    return {int(i) for i in ids[today.toordinal() <= rest_until]}


def resting_from_log(
    path: Path | str,
    build_date: str,
    *,
    front: int = REST_FRONT,
    after: int = REST_AFTER,
    days: int = REST_DAYS,
) -> set[int]:
    """resting_ids from the served log at path; no log yet means nothing is at rest."""
    path = Path(path)
    if not path.exists():
        return set()
    # Only the front batches count, so read only their rows: with every list served in full the
    # log gains a row per pool record per build, and the front is a small slice of it.
    served = pd.read_parquet(
        path, columns=["build_date", "batch_index", "id"], filters=[("batch_index", "<", front)]
    )
    return resting_ids(served, build_date, front=front, after=after, days=days)
