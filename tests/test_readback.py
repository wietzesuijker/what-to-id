import pandas as pd
import pytest

from what_to_id import analysis, readback


def _ident(user, ts, taxon, rank="species", current=True):
    return {
        "user": {"id": user},
        "created_at": ts,
        "taxon": {"id": taxon, "rank": rank},
        "current": current,
    }


OBS = {
    "id": 11,
    "quality_grade": "research",
    "community_taxon": {"id": 1, "rank": "species"},
    "taxon": {"id": 1, "rank": "species"},
    "identifications_count": 3,
    "user": {"id": 1},
    "reviewed_by": [3, 1, 2],
    "identifications": [
        _ident(1, "2026-09-12T00:00:00Z", 1),
        _ident(2, "2026-09-20T00:00:00Z", 2, rank="genus", current=False),
        _ident(2, "2026-09-25T00:00:00Z", 1),
    ],
}


def test_summarise():
    s = readback.summarise(OBS)
    assert s["id"] == 11
    assert s["user_id"] == 1
    assert s["quality_grade"] == "research"
    assert s["community_rank"] == "species"
    assert s["taxon_rank"] == "species"
    assert s["ident_count"] == 3
    assert s["n_identifiers"] == 2
    assert s["last_ident_at"] == "2026-09-25T00:00:00Z"
    assert s["reviewed_by"] == [1, 2, 3]
    assert len(s["identifications"]) == 3
    assert s["identifications"][0] == {
        "user_id": 1,
        "created_at": "2026-09-12T00:00:00Z",
        "taxon_id": 1,
        "taxon_rank": "species",
        "current": True,
    }
    assert s["identifications"][1]["taxon_rank"] == "genus"
    assert s["identifications"][1]["current"] is False


def test_summarise_sparse():
    s = readback.summarise({"id": 5, "quality_grade": "needs_id", "community_taxon": None})
    assert s["community_rank"] is None and s["taxon_rank"] is None
    assert s["user_id"] is None
    assert s["ident_count"] == 0 and s["n_identifiers"] == 0
    assert s["last_ident_at"] is None
    assert s["identifications"] == []
    assert s["reviewed_by"] == []


def test_readback_missing_ids_get_none_row():
    obs_df, idents_df = readback.readback([11, 99], fetch=lambda ids: [OBS])
    assert list(obs_df.columns) == list(readback.OBS_COLUMNS)
    assert obs_df["id"].tolist() == [11, 99]
    assert obs_df.loc[0, "user_id"] == 1 and pd.isna(obs_df.loc[1, "user_id"])
    assert pd.isna(obs_df.loc[1, "quality_grade"])
    assert pd.isna(obs_df.loc[1, "ident_count"])
    assert obs_df.loc[0, "n_identifiers"] == 2
    assert obs_df.loc[0, "reviewed_by"] == [1, 2, 3]
    assert obs_df.loc[1, "reviewed_by"] is None
    assert list(idents_df.columns) == list(readback.IDENT_COLUMNS)
    assert len(idents_df) == 3
    assert (idents_df["id"] == 11).all()


@pytest.fixture
def tiny():
    pool = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "iconic_taxon": ["Aves", "Aves", "Insecta", "Insecta"],
            "ident_count": [1, 0, 2, 1],
            "cell_score": [0.5, float("nan"), 2.0, 1.0],
        }
    )
    assign = pd.DataFrame({"id": [1, 2, 3, 4], "arm": ["A", "A", "B", "B"]})
    obs = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "quality_grade": ["research", "research", "needs_id", None],
            "community_rank": ["species", None, "species", None],
            "ident_count": [3, 2, 2, None],
        }
    )
    return pool, assign, obs


def test_outcomes(tiny):
    pool, assign, obs = tiny
    out = readback.outcomes(obs, assign, pool).set_index("arm")
    assert list(out.columns) == list(readback.OUTCOME_COLUMNS[1:])
    assert out.loc["A", "n_served"] == 2 and out.loc["B", "n_served"] == 2
    assert out.loc["A", "share_species"] == 1.0
    assert out.loc["B", "share_species"] == 0.5
    assert out.loc["A", "share_engaged"] == 1.0
    assert out.loc["B", "share_engaged"] == 0.5
    assert out.loc["B", "unengaged"] == 0.5
    assert out.loc["A", "weighted_rg"] == 0.5
    assert out.loc["B", "weighted_rg"] == 0.0


def test_outcomes_weights_by_served_score(tiny):
    pool, assign, obs = tiny
    served = assign.assign(cell_score=[3.0, 0.0, 2.0, 1.0])
    out = readback.outcomes(obs, served, pool).set_index("arm")
    # Record 1 is research grade: the served score 3.0 wins over the pool's 0.5.
    assert out.loc["A", "weighted_rg"] == 3.0


def test_outcomes_without_cell_score(tiny):
    pool, assign, obs = tiny
    out = readback.outcomes(obs, assign, pool.drop(columns="cell_score"))
    assert out["weighted_rg"].isna().all()


def test_outcomes_no_assign_single_arm(tiny):
    pool, _, obs = tiny
    out = readback.outcomes(obs, None, pool)
    assert out["arm"].tolist() == ["pool"]
    assert out.loc[0, "n_served"] == 4


def test_dry_run_per_group(tiny):
    pool, _, _ = tiny
    out = readback.dry_run(pool).set_index("arm")
    assert out.loc["Aves", "unengaged"] == 1.0
    assert out.loc["Insecta", "unengaged"] == 0.5
    assert out["share_species"].eq(0.0).all()


def test_to_markdown(tiny):
    pool, assign, obs = tiny
    md = readback.to_markdown(readback.outcomes(obs, assign, pool))
    lines = md.splitlines()
    assert lines[0].startswith("| arm |")
    assert "| A |" in lines[2] and "| B |" in lines[3]
    assert "100.0%" in lines[2]


def test_cli_dry_run(tmp_path, capsys):
    from what_to_id import inat

    pool = pd.DataFrame({c: [None] for c in inat.COLUMNS}).assign(
        id=[1], lat=[0.0], lon=[0.0], ident_count=[0], agree=[0], iconic_taxon=["Aves"]
    )
    p = tmp_path / "pool.parquet"
    pool.to_parquet(p)
    assert readback.main(["--pool", str(p), "--dry-run"]) == 0
    assert "| Aves |" in capsys.readouterr().out


def test_readback_idents_feed_analysis():
    _, idents_df = readback.readback([11], fetch=lambda ids: [OBS])
    served = pd.DataFrame({"id": [11], "arm": ["A"]})
    assert (idents_df["observer_id"] == 1).all()
    counts = analysis.identifier_counts(idents_df, served, start="2026-09-01", cutoff="2026-10-01")
    # User 1 observed record 11, so their own ID on it does not count.
    assert counts["A"].to_dict() == {2: 1}


def test_readback_from_served_log_union_with_label_map(tiny):
    """served_arms folds two daily served logs into the assign frame readback/outcomes expect."""
    pool, _, _ = tiny
    day1 = pd.DataFrame({"build_date": ["2026-09-01"] * 2, "label": ["A"] * 2, "id": [1, 2]})
    day2 = pd.DataFrame({"build_date": ["2026-09-02"] * 2, "label": ["B"] * 2, "id": [3, 4]})
    log = pd.concat([day1, day2], ignore_index=True)
    assign = analysis.served_arms(log, {"A": "recency", "B": "gap_first"})
    ids = assign["id"].drop_duplicates().tolist()
    assert sorted(ids) == [1, 2, 3, 4]

    def fake_fetch(ids_, **kw):
        return [{"id": i, "quality_grade": "needs_id", "identifications_count": 0} for i in ids_]

    obs_df, idents_df = readback.readback(ids, fetch=fake_fetch)
    assert sorted(obs_df["id"]) == [1, 2, 3, 4]
    assert idents_df.empty
    out = readback.outcomes(obs_df, assign, pool).set_index("arm")
    assert out.loc["recency", "n_served"] == 2
    assert out.loc["gap_first", "n_served"] == 2


def test_readback_reviewed_by_survives_parquet(tmp_path):
    obs_df, _ = readback.readback([11, 99], fetch=lambda ids: [OBS])
    p = tmp_path / "obs.parquet"
    obs_df.to_parquet(p, engine="pyarrow", index=False)
    back = pd.read_parquet(p, engine="pyarrow")
    assert list(back.loc[0, "reviewed_by"]) == [1, 2, 3]
    assert back.loc[1, "reviewed_by"] is None


def _window_idents(rows):
    return pd.DataFrame(rows, columns=["id", "user_id", "created_at"])


OUT_ASSIGN = pd.DataFrame({"id": [1, 2, 3, 4], "arm": ["recency"] * 2 + ["gap_first"] * 2})
OUT_OBS = pd.DataFrame({"id": [1, 2, 3, 4], "user_id": pd.array([90, 91, 92, 93], "Int64")})
OUT_KW = {
    "obs_df": OUT_OBS,
    "users": [10],
    "start": "2026-11-01",
    "cutoff": "2026-12-01",
    "control": "recency",
}


def test_outsiders_counts_records_a_non_participant_identified_first():
    idents = _window_idents(
        [
            (1, 50, "2026-11-02T00:00:00Z"),  # outsider before the participant: lost
            (1, 10, "2026-11-03T00:00:00Z"),
            (2, 51, "2026-11-04T00:00:00Z"),  # outsider only: lost
            (3, 10, "2026-11-02T00:00:00Z"),  # same time: not lost
            (3, 50, "2026-11-02T00:00:00Z"),
            (4, 50, "2026-10-15T00:00:00Z"),  # outsider before the window: ignored
            (4, 10, "2026-11-02T00:00:00Z"),
            (4, 52, "2026-12-02T00:00:00Z"),  # after the cutoff: ignored
        ]
    )
    out = readback.outsiders(idents, OUT_ASSIGN, **OUT_KW).set_index("arm")
    assert out.loc["recency", ["n_served", "outsider"]].tolist() == [2, 2]
    assert out.loc["recency", "share"] == 1.0 and out.loc["gap_first", "share"] == 0.0
    assert out.loc["gap_first", "diff"] == -1.0 and out.loc["recency", "diff"] == 0.0


def test_outsiders_leaves_out_the_observers_own_ids():
    obs = pd.DataFrame({"id": [1, 2, 3, 4], "user_id": [60, 61, 10, 10]})
    idents = _window_idents(
        [
            (1, 60, "2026-11-02T00:00:00Z"),  # upload ID in the window, observer only: not lost
            (1, 60, "2026-11-05T00:00:00Z"),  # the observer's later ID: also left out
            (2, 61, "2026-11-02T00:00:00Z"),  # upload ID, left out
            (2, 50, "2026-11-03T00:00:00Z"),  # outsider before the participant: lost
            (2, 10, "2026-11-04T00:00:00Z"),
            (3, 10, "2026-11-02T00:00:00Z"),  # participant's upload of their own record: left out
            (3, 50, "2026-11-03T00:00:00Z"),  # so the outsider is first: lost
            (4, 10, "2026-11-02T00:00:00Z"),  # participant observer, left out
            (4, 11, "2026-11-03T00:00:00Z"),  # another participant first: not lost
            (4, 50, "2026-11-04T00:00:00Z"),
        ]
    )
    kw = {**OUT_KW, "obs_df": obs, "users": [10, 11]}
    out = readback.outsiders(idents, OUT_ASSIGN, **kw).set_index("arm")
    assert out.loc["recency", "outsider"] == 1 and out.loc["gap_first", "outsider"] == 1
    assert out.loc["recency", "share"] == 0.5 and out.loc["gap_first", "diff"] == 0.0


def test_outsiders_keeps_every_id_on_a_record_with_no_known_observer():
    obs = pd.DataFrame({"id": [1, 2, 3, 4], "user_id": pd.array([None, 91, 92, 93], "Int64")})
    idents = _window_idents(
        [
            (1, 50, "2026-11-02T00:00:00Z"),  # observer unknown: the outsider ID still counts
            (1, 10, "2026-11-03T00:00:00Z"),
            (3, 10, "2026-11-02T00:00:00Z"),
        ]
    )
    kw = {**OUT_KW, "obs_df": obs, "users": [10]}
    out = readback.outsiders(idents, OUT_ASSIGN, **kw).set_index("arm")
    assert out.loc["recency", "outsider"] == 1 and out.loc["gap_first", "outsider"] == 0
    missing = obs.iloc[1:]  # record 1 absent from obs_df altogether: same rule
    out = readback.outsiders(idents, OUT_ASSIGN, **{**kw, "obs_df": missing}).set_index("arm")
    assert out.loc["recency", "outsider"] == 1


def test_outsiders_rejects_bad_input():
    idents = _window_idents([(1, 50, "2026-11-02T00:00:00Z")])
    with pytest.raises(ValueError, match="obs_df missing"):
        readback.outsiders(idents, OUT_ASSIGN, **{**OUT_KW, "obs_df": OUT_OBS[["id"]]})
    with pytest.raises(ValueError, match="control arm"):
        readback.outsiders(idents, OUT_ASSIGN, **{**OUT_KW, "control": "nope"})
    with pytest.raises(ValueError, match="not after start"):
        readback.outsiders(idents, OUT_ASSIGN, **{**OUT_KW, "cutoff": "2026-10-01"})
    with pytest.raises(ValueError, match="idents missing"):
        readback.outsiders(idents.drop(columns="created_at"), OUT_ASSIGN, **OUT_KW)


def test_cli_prints_the_outsider_check(tmp_path, capsys, monkeypatch):
    from what_to_id import inat

    pool = pd.DataFrame({c: [None] * 4 for c in inat.COLUMNS}).assign(
        id=[1, 2, 3, 4], lat=0.0, lon=0.0, ident_count=0, agree=0, iconic_taxon="Aves"
    )
    pool.to_parquet(tmp_path / "pool.parquet")
    OUT_ASSIGN.to_parquet(tmp_path / "a.parquet")
    (tmp_path / "u.txt").write_text("10\n")
    idents = _window_idents([(1, 50, "2026-11-02T00:00:00Z")])
    obs = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "user_id": [90, 91, 92, 93],
            "quality_grade": "needs_id",
            "community_rank": None,
            "ident_count": 1,
        }
    )
    monkeypatch.setattr(readback, "readback", lambda ids: (obs, idents))
    args = ["--pool", str(tmp_path / "pool.parquet"), "--assign", str(tmp_path / "a.parquet")]
    args += ["--out", str(tmp_path / "o.parquet"), "--users", str(tmp_path / "u.txt")]
    assert readback.main([*args, "--start", "2026-11-01", "--cutoff", "2026-12-01"]) == 0
    out = capsys.readouterr().out
    assert "outsiders: served records a non-participant identified before any participant" in out
    assert "| gap_first | 2 | 0 | 0.000 | -0.500 |" in out
    with pytest.raises(SystemExit):
        readback.main(args)
