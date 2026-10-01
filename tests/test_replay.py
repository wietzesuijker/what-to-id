import json

import pandas as pd
import pytest

from what_to_id.manifest import sha256_file
from what_to_id.replay import ReplayError, main, replay

from .conftest import make_pool
from .test_cli import _keyed_build

KEY = "ab" * 32
WRONG_KEY = "cd" * 32


def _two_day_builds(tmp_path, webapp_dir, monkeypatch):
    monkeypatch.setenv("WHAT_TO_ID_KEY", KEY)
    pool = make_pool(300, seed=5)
    day1, day2 = tmp_path / "pool1.parquet", tmp_path / "pool2.parquet"
    pool.iloc[:200].to_parquet(day1, index=False)
    pool.iloc[50:].to_parquet(day2, index=False)
    log_path = tmp_path / "state" / "served.parquet"
    o1, o2 = tmp_path / "o1", tmp_path / "o2"
    assert _keyed_build(day1, webapp_dir, o1, "2026-11-03", log_path) == 0
    assert _keyed_build(day2, webapp_dir, o2, "2026-11-04", log_path) == 0
    return day1, day2, o1, o2, log_path


def test_replay_matches_each_days_build(tmp_path, webapp_dir, monkeypatch, caplog):
    caplog.set_level("INFO", logger="what_to_id")
    day1, day2, o1, o2, log_path = _two_day_builds(tmp_path, webapp_dir, monkeypatch)
    key = bytes.fromhex(KEY)

    r1 = replay(day1, o1 / "build_record.json", log_path, webapp_dir, key=key)
    assert r1.ok and r1.expected_rows > 0 and r1.expected_rows == r1.rebuilt_rows

    r2 = replay(day2, o2 / "build_record.json", log_path, webapp_dir, key=key)
    assert r2.ok and r2.expected_rows > 0 and r2.expected_rows == r2.rebuilt_rows

    for word in ("recency", "gap_first", KEY):
        assert word not in caplog.text


def test_replay_refuses_wrong_pool(tmp_path, webapp_dir, monkeypatch):
    day1, day2, o1, o2, log_path = _two_day_builds(tmp_path, webapp_dir, monkeypatch)
    key = bytes.fromhex(KEY)
    with pytest.raises(ReplayError, match="pool_sha256"):
        replay(day2, o1 / "build_record.json", log_path, webapp_dir, key=key)


def test_replay_refuses_wrong_key(tmp_path, webapp_dir, monkeypatch):
    day1, day2, o1, o2, log_path = _two_day_builds(tmp_path, webapp_dir, monkeypatch)
    with pytest.raises(ReplayError, match="key_fingerprint"):
        replay(day1, o1 / "build_record.json", log_path, webapp_dir, key=bytes.fromhex(WRONG_KEY))


def test_replay_reports_mismatch_when_pool_edited_after_build(tmp_path, webapp_dir, monkeypatch):
    day1, day2, o1, o2, log_path = _two_day_builds(tmp_path, webapp_dir, monkeypatch)
    key = bytes.fromhex(KEY)

    edited = tmp_path / "pool1_edited.parquet"
    pd.read_parquet(day1).iloc[1:].to_parquet(edited, index=False)
    record_path = o1 / "build_record.json"
    record = json.loads(record_path.read_text())
    record["pool_sha256"] = sha256_file(edited)
    record_path.write_text(json.dumps(record))

    result = replay(edited, record_path, log_path, webapp_dir, key=key)
    assert not result.ok
    assert len(result.mismatches) > 0

    rc = main(
        [
            "--pool",
            str(edited),
            "--record",
            str(record_path),
            "--served-log",
            str(log_path),
            "--webapp-dir",
            str(webapp_dir),
            "--key-env",
            "WHAT_TO_ID_KEY",
        ]
    )
    assert rc == 1


def test_compare_refuses_duplicate_multiplicity():
    from what_to_id.replay import _compare
    from what_to_id.served_log import COLUMNS

    a = dict.fromkeys(COLUMNS, "a")
    b = dict.fromkeys(COLUMNS, "b")
    with pytest.raises(ReplayError, match="duplicate"):
        _compare(pd.DataFrame([a, a, b]), pd.DataFrame([a, b, b]))


def test_replay_rebuilds_a_day_with_records_at_rest(tmp_path, webapp_dir, monkeypatch):
    monkeypatch.setenv("WHAT_TO_ID_KEY", KEY)
    pool_path = tmp_path / "pool.parquet"
    make_pool(300, seed=5).to_parquet(pool_path, index=False)
    log_path = tmp_path / "state" / "served.parquet"
    rest = ["--rest-log", str(log_path), "--rest-front", "1"]
    days = ["2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"]
    for day in days:
        assert _keyed_build(pool_path, webapp_dir, tmp_path / day, day, log_path, rest) == 0

    served = pd.read_parquet(log_path)
    first = served[served["batch_index"] == 0]
    front = {d: set(first.loc[first["build_date"] == d, "id"]) for d in days}
    always = front[days[0]] & front[days[1]] & front[days[2]]
    assert always
    record = json.loads((tmp_path / days[3] / "build_record.json").read_text())
    assert record["rest"] == {"front": 1, "after": 3, "days": 7, "rows": len(always)}

    # Every record is still served on the rest day, the resting ones behind the rest of their list.
    last = served[served["build_date"] == days[3]]
    assert set(last["id"]) == set(served.loc[served["build_date"] == days[0], "id"])
    for _, lst in last.groupby(["label", "group"]):
        rank = lst.sort_values(["batch_index", "position"])["id"].isin(always).to_numpy()
        assert not (rank[:-1] & ~rank[1:]).any()

    result = replay(
        pool_path,
        tmp_path / days[3] / "build_record.json",
        log_path,
        webapp_dir,
        key=bytes.fromhex(KEY),
    )
    assert result.ok and result.expected_rows > 0
