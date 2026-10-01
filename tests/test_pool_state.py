import json

import pandas as pd
import pytest

from what_to_id import pool_state
from what_to_id.inat import COLUMNS, DTYPES

from .conftest import make_pool


def _new_rows(ids, created="2026-09-01T00:00:00+00:00"):
    df = make_pool(len(ids))
    df["id"] = pd.array(ids, dtype="int64")
    df["created_at"] = created
    return df


def test_merge_new_dedupes_new_wins_and_keeps_dtypes():
    pool = make_pool(3)
    pool.loc[:, "id"] = [1, 2, 3]
    new = _new_rows([2, 4])
    new.loc[new["id"] == 2, "taxon_name"] = "Updated"
    merged = pool_state.merge_new(pool, new)
    assert list(merged["id"]) == [1, 2, 3, 4]
    row2 = merged[merged["id"] == 2].iloc[0]
    assert row2["taxon_name"] == "Updated"
    for col, dtype in DTYPES.items():
        assert str(merged[col].dtype) == dtype
    assert list(merged.columns) == list(COLUMNS)


def test_merge_new_keeps_extras_when_the_old_pool_lacks_them():
    pool = make_pool(2)
    pool.loc[:, "id"] = [1, 2]
    new = _new_rows([3])
    new["introduced"] = pd.array([True], dtype="boolean")
    merged = pool_state.merge_new(pool, new)
    assert list(merged.columns) == [*COLUMNS, "introduced"]
    assert str(merged["introduced"].dtype) == "boolean"
    assert merged["introduced"].isna().tolist() == [True, True, False]


def test_merge_new_rejects_column_mismatch():
    pool = make_pool(2)
    bad = pool.drop(columns=["photo_url"])
    with pytest.raises(ValueError, match="photo_url"):
        pool_state.merge_new(pool, bad)


def test_drop_closed_keeps_unchecked_ids():
    pool = make_pool(4)
    pool.loc[:, "id"] = [10, 11, 12, 13]
    updated = pool_state.drop_closed(pool, checked=[10, 11], open_ids={10})
    assert sorted(updated["id"]) == [10, 12, 13]


def test_drop_closed_no_closures_returns_all():
    pool = make_pool(3)
    pool.loc[:, "id"] = [1, 2, 3]
    updated = pool_state.drop_closed(pool, checked=[1, 2, 3], open_ids={1, 2, 3})
    assert sorted(updated["id"]) == [1, 2, 3]


def test_last_created():
    pool = make_pool(5)
    ts = pool_state.last_created(pool)
    expected = pd.to_datetime(pool["created_at"], utc=True).max()
    assert ts == expected


def test_last_created_empty_pool_raises():
    pool = make_pool(0)
    with pytest.raises(ValueError, match="empty"):
        pool_state.last_created(pool)


def test_cmd_update_pulls_since_last_created_and_merges(tmp_path, monkeypatch):
    pool = make_pool(3)
    pool.loc[:, "id"] = [1, 2, 3]
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, index=False)

    captured = {}

    def fake_pull_pool(*, d1, freeze, out, created_d1, **kw):
        captured["created_d1"] = created_d1
        captured["d1"] = d1
        return _new_rows([4, 5])

    monkeypatch.setattr(pool_state, "pull_pool", fake_pull_pool)

    rc = pool_state.main(["update", "--pool", str(pool_path), "--overlap-hours", "6"])
    assert rc == 0

    expected_since = pool_state.last_created(pool) - pd.Timedelta(hours=6)
    assert captured["created_d1"] == expected_since
    assert captured["d1"] == pool_state.DEFAULT_D1

    result = pd.read_parquet(pool_path)
    assert sorted(result["id"]) == [1, 2, 3, 4, 5]


def test_cmd_update_uses_given_d1(tmp_path, monkeypatch):
    pool = make_pool(2)
    pool.loc[:, "id"] = [1, 2]
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, index=False)

    captured = {}

    def fake_pull_pool(*, d1, freeze, out, created_d1, **kw):
        captured["d1"] = d1
        return _new_rows([3])

    monkeypatch.setattr(pool_state, "pull_pool", fake_pull_pool)
    rc = pool_state.main(["update", "--pool", str(pool_path), "--d1", "2026-01-01"])
    assert rc == 0
    assert captured["d1"] == "2026-01-01"


def test_cmd_refresh_drops_closed_ids(tmp_path, monkeypatch):
    pool = make_pool(4)
    pool.loc[:, "id"] = [1, 2, 3, 4]
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, index=False)

    served = pd.DataFrame({"id": pd.array([1, 2, 3], dtype="int64")})
    served_path = tmp_path / "served.parquet"
    served.to_parquet(served_path, index=False)

    captured = {}

    def fake_still_open(ids, *, session=None):
        captured["ids"] = list(ids)
        return {1, 3}

    monkeypatch.setattr(pool_state, "still_open", fake_still_open)

    rc = pool_state.main(["refresh", "--pool", str(pool_path), "--ids", str(served_path)])
    assert rc == 0
    assert sorted(captured["ids"]) == [1, 2, 3]

    result = pd.read_parquet(pool_path)
    assert sorted(result["id"]) == [1, 3, 4]


def test_cmd_refresh_front_checks_only_the_first_batches(tmp_path, monkeypatch):
    pool = make_pool(4)
    pool.loc[:, "id"] = [1, 2, 3, 4]
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, index=False)
    batches = pd.DataFrame(
        {
            "batch_id": ["recency-Aves-000", "recency-Aves-001", "recency-Aves-1000", "x-Aves-002"],
            "id": pd.array([1, 2, 3, 4], dtype="int64"),
        }
    )
    batches_path = tmp_path / "batches.parquet"
    batches.to_parquet(batches_path, index=False)
    captured = {}

    def fake_still_open(ids, *, session=None):
        captured["ids"] = sorted(ids)
        return set()

    monkeypatch.setattr(pool_state, "still_open", fake_still_open)
    args = ["refresh", "--pool", str(pool_path), "--ids", str(batches_path), "--front", "2"]
    assert pool_state.main(args) == 0
    assert captured["ids"] == [1, 2]
    assert sorted(pd.read_parquet(pool_path)["id"]) == [3, 4]


@pytest.mark.parametrize("front", ["0", "-1"])
def test_cmd_refresh_rejects_an_empty_front(tmp_path, front):
    ids = tmp_path / "batches.parquet"
    pd.DataFrame({"batch_id": ["a-Aves-000"], "id": [1]}).to_parquet(ids)
    with pytest.raises(SystemExit, match="--front"):
        pool_state.main(["refresh", "--pool", str(ids), "--ids", str(ids), "--front", front])


def test_cmd_refresh_front_needs_batch_ids(tmp_path):
    ids = tmp_path / "ids.parquet"
    pd.DataFrame({"id": [1]}).to_parquet(ids)
    with pytest.raises(SystemExit, match="batch_id"):
        pool_state.main(["refresh", "--pool", str(ids), "--ids", str(ids), "--front", "1"])


def test_prepared_snapshot_retains_closures_across_days_and_reconciles_new_bundle(
    tmp_path, monkeypatch
):
    import json
    import shutil

    from what_to_id.arms import Recency
    from what_to_id.batches import build_batches

    original = _new_rows(range(1, 11))
    original["iconic_taxon"] = "Aves"
    source, pool = tmp_path / "prepared.parquet", tmp_path / "pool.parquet"
    state, ids = tmp_path / "bundle-eligibility.json", tmp_path / "ids.parquet"
    original.to_parquet(source, index=False)
    assignment = original[["id"]].assign(arm="recency")

    def prepare():
        assert (
            pool_state.main(
                [
                    "prepared",
                    "--prepared-pool",
                    str(source),
                    "--pool",
                    str(pool),
                    "--eligibility",
                    str(state),
                ]
            )
            == 0
        )

    def selection():
        return build_batches(
            pd.read_parquet(pool),
            assignment,
            {"recency": Recency()},
            size=2,
            seed=0,
            max_batches=1,
        )["id"].tolist()

    def refresh(closed):
        pd.DataFrame({"id": selection()}).to_parquet(ids, index=False)
        monkeypatch.setattr(pool_state, "still_open", lambda checked: set(checked) - closed)
        assert (
            pool_state.main(
                ["refresh", "--pool", str(pool), "--ids", str(ids), "--eligibility", str(state)]
            )
            == 0
        )

    prepare()
    assert selection() == [10, 9]
    refresh({10})
    assert selection() == [9, 8]
    # Only the small exclusion state is needed to reconstruct the next day's eligible pool.
    saved = tmp_path / "release-asset.json"
    shutil.copyfile(state, saved)
    pool.unlink()
    state.unlink()
    shutil.copyfile(saved, state)
    prepare()
    assert selection() == [9, 8]
    refresh({10, 8})
    assert selection() == [9, 7]
    assert json.loads(state.read_text())["closed_ids"] == [8, 10]

    # A newly prepared needs-ID snapshot is authoritative: include new IDs and reopened IDs.
    updated = pd.concat([original, _new_rows([11])], ignore_index=True)
    updated.to_parquet(source, index=False)
    prepare()
    assert set(pd.read_parquet(pool).id) == set(range(1, 12))
    assert json.loads(state.read_text())["closed_ids"] == []


@pytest.mark.parametrize("bad", ["{}", '{"schema_version":1}', "not-json"])
def test_corrupt_eligibility_fails_before_replacing_pool(tmp_path, bad):
    source, pool, state = (
        tmp_path / name for name in ("source.parquet", "pool.parquet", "state.json")
    )
    _new_rows([1, 2]).to_parquet(source, index=False)
    pool.write_bytes(b"previous pool")
    state.write_text(bad)
    with pytest.raises(ValueError):
        pool_state.main(
            [
                "prepared",
                "--prepared-pool",
                str(source),
                "--pool",
                str(pool),
                "--eligibility",
                str(state),
            ]
        )
    assert pool.read_bytes() == b"previous pool"


def test_apply_changes_drops_gone_and_upserts_changed():
    pool = make_pool(4)
    pool.loc[:, "id"] = [1, 2, 3, 4]
    changed = _new_rows([3, 5])
    changed.loc[changed["id"] == 3, "taxon_name"] = "Reidentified"
    updated = pool_state.apply_changes(pool, changed, gone={2, 99})
    assert updated["id"].tolist() == [1, 3, 4, 5]
    assert updated.loc[updated["id"] == 3, "taxon_name"].item() == "Reidentified"


def _sync(
    tmp_path, monkeypatch, *, state=None, since=None, api=4, newest="2026-09-30T01:00:00+00:00"
):
    pool = make_pool(4)
    pool.loc[:, "id"] = [1, 2, 3, 4]
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, index=False)
    state_path = tmp_path / "state" / "sync.json"
    if state is not None:
        state_path.parent.mkdir()
        state_path.write_text(json.dumps(state))
    asked = {}

    def fake_changed_since(start, *, d1):
        asked.update(start=start, d1=d1)
        return _new_rows([4, 7]), {2, 99}, newest

    def fake_total(params):
        asked["params"] = params
        return api

    monkeypatch.setattr(pool_state, "changed_since", fake_changed_since)
    monkeypatch.setattr(pool_state, "total_results", fake_total)
    argv = ["sync", "--pool", str(pool_path), "--state", str(state_path), "--d1", "1900-01-01"]
    if since:
        argv += ["--since", since]
    assert pool_state.main(argv) == 0
    return pd.read_parquet(pool_path), json.loads(state_path.read_text()), asked


def test_cmd_sync_first_run_uses_since_and_saves_the_newest_update(tmp_path, monkeypatch):
    pool, state, asked = _sync(tmp_path, monkeypatch, since="2026-09-29T16:00:00Z")
    assert pool["id"].tolist() == [1, 3, 4, 7]
    assert asked["start"] == pd.Timestamp("2026-09-29T14:00:00Z")
    assert asked["d1"] == "1900-01-01"
    assert "iconic_taxa" not in asked["params"] and "without_taxon_id" not in asked["params"]
    assert state["since"] == "2026-09-30T01:00:00+00:00"
    assert (state["added"], state["replaced"], state["dropped"]) == (1, 1, 1)
    assert (state["pool"], state["api_total"], state["drift"]) == (4, 4, 0)
    assert state["reconcile_due"] is False


def test_cmd_sync_reads_since_from_state_and_keeps_it_when_nothing_changed(tmp_path, monkeypatch):
    _, state, asked = _sync(
        tmp_path, monkeypatch, state={"since": "2026-09-30T01:00:00+00:00"}, newest=None
    )
    assert asked["start"] == pd.Timestamp("2026-09-29T23:00:00Z")
    assert state["since"] == "2026-09-30T01:00:00+00:00"


def test_cmd_sync_flags_drift(tmp_path, monkeypatch, capsys):
    _, state, _ = _sync(tmp_path, monkeypatch, since="2026-09-29", api=4 + 1001)
    assert state["drift"] == 1001 and state["reconcile_due"] is True
    assert "::warning::" in capsys.readouterr().out


def test_cmd_sync_needs_a_since(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="--since"):
        _sync(tmp_path, monkeypatch)


def test_cmd_reconcile_drops_gone_adds_missing_and_keeps_other_keys(tmp_path, monkeypatch):
    pool = make_pool(4)
    pool.loc[:, "id"] = [1, 2, 3, 4]
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, index=False)
    state_path = tmp_path / "sync.json"
    state_path.write_text(json.dumps({"since": "2026-09-30T01:00:00+00:00", "added": 3}))
    asked = {}

    def fake_reconcile(ids, *, d1, max_requests):
        asked.update(ids=sorted(ids), d1=d1, max_requests=max_requests)
        return {2}, {8, 9}, 17

    def fake_fetch(ids):
        asked["fetched"] = list(ids)
        return [{"id": 8, "quality_grade": "needs_id"}, {"id": 9, "quality_grade": "research"}]

    row = _new_rows([8]).iloc[0].to_dict()
    monkeypatch.setattr(pool_state, "reconcile_ids", fake_reconcile)
    monkeypatch.setattr(pool_state, "fetch_by_ids", fake_fetch)
    monkeypatch.setattr(pool_state, "flatten", lambda obs: row if obs["id"] == 8 else None)
    monkeypatch.setattr(pool_state, "total_results", lambda params: 4)
    argv = ["reconcile", "--pool", str(pool_path), "--state", str(state_path), "--d1", "1900-01-01"]
    assert pool_state.main([*argv, "--max-requests", "50"]) == 0
    assert asked["ids"] == [1, 2, 3, 4] and asked["max_requests"] == 50
    assert asked["d1"] == "1900-01-01" and asked["fetched"] == [8, 9]
    assert pd.read_parquet(pool_path)["id"].tolist() == [1, 3, 4, 8]
    state = json.loads(state_path.read_text())
    assert state["since"] == "2026-09-30T01:00:00+00:00" and state["added"] == 3
    assert (state["reconcile_gone"], state["reconcile_added"]) == (1, 1)
    assert state["reconcile_requests"] == 17 and "reconciled_at" in state
    assert (state["pool"], state["api_total"], state["drift"]) == (4, 4, 0)
    assert state["reconcile_due"] is False


def test_reconcile_failure_leaves_pool_and_state_untouched(tmp_path, monkeypatch):
    pool_path, state_path = tmp_path / "pool.parquet", tmp_path / "sync.json"
    make_pool(3).to_parquet(pool_path, index=False)
    state_path.write_text(json.dumps({"since": "2026-09-01T00:00:00+00:00"}))
    before = (pool_path.read_bytes(), state_path.read_bytes())

    def boom(*args, **kwargs):
        raise RuntimeError("reconcile needs more than 1 requests; raise the cap")

    monkeypatch.setattr(pool_state, "reconcile_ids", boom)
    argv = ["reconcile", "--pool", str(pool_path), "--state", str(state_path)]
    with pytest.raises(RuntimeError, match="raise the cap"):
        pool_state.main(argv)
    assert (pool_path.read_bytes(), state_path.read_bytes()) == before
    assert not list(tmp_path.glob("*.tmp"))
