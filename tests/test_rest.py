import pandas as pd
import pytest

from what_to_id.rest import resting_from_log, resting_ids


def _log(served, batch_index=0):
    """served: {build_date: [ids]} -> a served log with the columns rest reads."""
    rows = [(d, batch_index, i) for d, ids in served.items() for i in ids]
    return pd.DataFrame(rows, columns=["build_date", "batch_index", "id"])


def test_three_builds_in_a_row_rest_the_record_for_seven_days():
    log = _log({"2026-11-01": [1, 2], "2026-11-02": [1, 2], "2026-11-03": [1]})
    assert resting_ids(log, "2026-11-04") == {1}
    assert resting_ids(log, "2026-11-10") == {1}
    assert resting_ids(log, "2026-11-11") == set()


def test_a_gap_starts_the_count_over():
    log = _log({"2026-11-01": [1], "2026-11-02": [1], "2026-11-03": [2], "2026-11-04": [1]})
    assert resting_ids(log, "2026-11-05") == set()


def test_only_the_front_batches_count():
    log = _log({"2026-11-01": [1], "2026-11-02": [1], "2026-11-03": [1]}, batch_index=2)
    assert resting_ids(log, "2026-11-04", front=2) == set()
    assert resting_ids(log, "2026-11-04", front=3) == {1}


def test_builds_count_not_calendar_days():
    log = _log({"2026-11-01": [1], "2026-11-03": [1], "2026-11-06": [1]})
    assert resting_ids(log, "2026-11-07") == {1}


def test_the_count_starts_over_after_a_rest():
    days = {f"2026-11-{d:02d}": [1] for d in (1, 2, 3)}
    days.update({f"2026-11-{d:02d}": [2] for d in range(4, 11)})
    days.update({"2026-11-11": [1], "2026-11-12": [1]})
    log = _log(days)
    assert 1 not in resting_ids(log, "2026-11-13")
    log = _log({**days, "2026-11-13": [1]})
    assert 1 in resting_ids(log, "2026-11-14")


def test_only_earlier_builds_count():
    log = _log({"2026-11-01": [1], "2026-11-02": [1], "2026-11-03": [1]})
    assert resting_ids(log, "2026-11-03") == set()
    assert resting_ids(log, "2026-11-01") == set()


def test_empty_or_missing_log_rests_nothing(tmp_path):
    assert resting_ids(_log({}), "2026-11-04") == set()
    assert resting_from_log(tmp_path / "served.parquet", "2026-11-04") == set()


def test_reads_the_served_log_file(tmp_path):
    path = tmp_path / "served.parquet"
    _log({"2026-11-01": [7], "2026-11-02": [7], "2026-11-03": [7]}).to_parquet(path)
    assert resting_from_log(path, "2026-11-04", after=3, days=1) == {7}
    assert resting_from_log(path, "2026-11-05", after=3, days=1) == set()


@pytest.mark.parametrize("front,after,days", [(0, 3, 7), (2, 0, 7), (2, 3, 0)])
def test_rejects_an_empty_rule(front, after, days):
    with pytest.raises(ValueError, match="rest needs"):
        resting_ids(_log({}), "2026-11-04", front=front, after=after, days=days)
