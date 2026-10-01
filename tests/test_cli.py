import json

import numpy as np
import pandas as pd
import pytest

from what_to_id.cli import main
from what_to_id.manifest import validate_manifest

from .conftest import make_pool


def test_build_end_to_end(tmp_path, webapp_dir):
    pool = make_pool(300, seed=5)
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, index=False)
    out = tmp_path / "out"
    rc = main(
        [
            "build",
            "--pool",
            str(pool_path),
            "--freeze",
            "2026-09-01",
            "--d1",
            "2026-09-15",
            "--seed",
            "3",
            "--batch-size",
            "25",
            "--arms",
            "recency,gap_first",
            "--webapp-dir",
            str(webapp_dir),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    d = json.loads((out / "manifest.json").read_text())
    validate_manifest(d)
    assert d["pool_rows"] == 300 and d["seed"] == 3 and d["arms"] == ["recency", "gap_first"]
    assert sum(len(b["ids"]) for b in d["batches"].values()) == 300
    a = pd.read_parquet(out / "assign.parquet")
    b = pd.read_parquet(out / "batches.parquet")
    assert len(a) == 300 and len(b) == 300 and b["id"].is_unique
    for name in ["index.html", "arm_A.html", "arm_B.html"]:
        assert (out / "site" / name).exists()
    assert "recency" not in (out / "site" / "index.html").read_text()


def test_build_max_batches(tmp_path, webapp_dir):
    pool = make_pool(300, seed=5)
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, index=False)
    out = tmp_path / "out"
    rc = main(
        [
            "build",
            "--pool",
            str(pool_path),
            "--freeze",
            "2026-09-01",
            "--d1",
            "2026-09-15",
            "--seed",
            "3",
            "--batch-size",
            "25",
            "--arms",
            "recency,gap_first",
            "--webapp-dir",
            str(webapp_dir),
            "--max-batches",
            "2",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    d = json.loads((out / "manifest.json").read_text())
    validate_manifest(d)
    assert d["max_batches"] == 2
    b = pd.read_parquet(out / "batches.parquet")
    assert d["served_rows"] == len(b)
    nb = b.groupby(["arm", "group"])["batch_id"].nunique()
    assert (nb <= 2).all()


def test_build_design_rotation_writes_only_index(tmp_path, webapp_dir):
    pool = make_pool(300, seed=5)
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, index=False)
    out = tmp_path / "out"
    rc = main(
        [
            "build",
            "--pool",
            str(pool_path),
            "--freeze",
            "2026-09-01",
            "--d1",
            "2026-09-15",
            "--seed",
            "3",
            "--batch-size",
            "25",
            "--arms",
            "recency,gap_first",
            "--webapp-dir",
            str(webapp_dir),
            "--design",
            "rotation",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    d = json.loads((out / "manifest.json").read_text())
    validate_manifest(d)
    assert d["design"] == "rotation"
    site = out / "site"
    assert sorted(p.name for p in site.iterdir()) == [
        "index.html",
        "map.html",
        "pool-0.bin",
        "pool-taxa.bin",
    ]
    html = (site / "index.html").read_text().lower()
    assert 'href="map.html"' in html
    for word in ("recency", "gap_first", "novelty", "batch_id"):
        assert word not in html


def test_build_surprise_arm_reads_scores_and_stays_blind(tmp_path, webapp_dir):
    pool = make_pool(300, seed=5)
    pool_path, scores = tmp_path / "pool.parquet", tmp_path / "surprise.parquet"
    pool.to_parquet(pool_path, index=False)
    s = np.linspace(0, 1, 300)
    pd.DataFrame({"id": pool["id"], "surprise": s}).to_parquet(scores, index=False)
    args = ["build", "--pool", str(pool_path), "--freeze", "2026-09-01", "--d1", "2026-09-15"]
    args += ["--batch-size", "25", "--arms", "recency,surprise", "--design", "rotation"]
    args += ["--webapp-dir", str(webapp_dir), "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="surprise-scores"):
        main(args)
    assert main([*args, "--surprise-scores", str(scores)]) == 0
    b = pd.read_parquet(tmp_path / "out" / "batches.parquet")
    first = b[b["batch_id"].str.endswith("-000") & (b["arm"] == "surprise") & (b["position"] == 0)]
    by_id = dict(zip(pool["id"], s, strict=True))
    for _, row in first.iterrows():
        rest = b[(b["arm"] == "surprise") & (b["group"] == row["group"])]
        assert by_id[row["id"]] == max(by_id[i] for i in rest["id"])
    html = (tmp_path / "out" / "site" / "index.html").read_text().lower()
    assert "surprise" not in html and "unexpected sightings first" in html


def test_build_surprise_arm_accepts_scores_with_n_ref(tmp_path, webapp_dir):
    pool = make_pool(300, seed=5)
    pool_path, scores = tmp_path / "pool.parquet", tmp_path / "surprise.parquet"
    pool.to_parquet(pool_path, index=False)
    s = np.linspace(0, 1, 300)
    n_ref = np.arange(300, dtype=float)
    pd.DataFrame({"id": pool["id"], "surprise": s, "n_ref": n_ref}).to_parquet(scores, index=False)
    args = ["build", "--pool", str(pool_path), "--freeze", "2026-09-01", "--d1", "2026-09-15"]
    args += ["--batch-size", "25", "--arms", "recency,surprise", "--design", "rotation"]
    args += ["--webapp-dir", str(webapp_dir), "--out", str(tmp_path / "out")]
    args += ["--surprise-scores", str(scores)]
    assert main(args) == 0


def test_build_surprise_arm_accepts_sinr_scores(tmp_path, webapp_dir):
    pool = make_pool(300, seed=5)
    pool_path = tmp_path / "pool.parquet"
    surprise_scores = tmp_path / "surprise.parquet"
    sinr_scores = tmp_path / "sinr.parquet"
    pool.to_parquet(pool_path, index=False)
    pd.DataFrame({"id": pool["id"], "surprise": np.linspace(0, 1, 300)}).to_parquet(
        surprise_scores, index=False
    )
    pd.DataFrame({"id": pool["id"], "sinr_rel": np.linspace(0, 2, 300)}).to_parquet(
        sinr_scores, index=False
    )
    args = ["build", "--pool", str(pool_path), "--freeze", "2026-09-01", "--d1", "2026-09-15"]
    args += ["--batch-size", "25", "--arms", "recency,surprise", "--design", "rotation"]
    args += ["--webapp-dir", str(webapp_dir), "--out", str(tmp_path / "out")]
    args += ["--surprise-scores", str(surprise_scores), "--sinr-scores", str(sinr_scores)]
    assert main(args) == 0


def test_build_sinr_scores_requires_id_and_sinr_rel(tmp_path, webapp_dir):
    pool = make_pool(300, seed=5)
    pool_path = tmp_path / "pool.parquet"
    surprise_scores = tmp_path / "surprise.parquet"
    sinr_scores = tmp_path / "sinr.parquet"
    pool.to_parquet(pool_path, index=False)
    pd.DataFrame({"id": pool["id"], "surprise": np.linspace(0, 1, 300)}).to_parquet(
        surprise_scores, index=False
    )
    pd.DataFrame({"id": pool["id"], "rel": np.linspace(0, 2, 300)}).to_parquet(
        sinr_scores, index=False
    )
    args = ["build", "--pool", str(pool_path), "--freeze", "2026-09-01", "--d1", "2026-09-15"]
    args += ["--batch-size", "25", "--arms", "recency,surprise", "--design", "rotation"]
    args += ["--webapp-dir", str(webapp_dir), "--out", str(tmp_path / "out")]
    args += ["--surprise-scores", str(surprise_scores), "--sinr-scores", str(sinr_scores)]
    with pytest.raises(SystemExit, match="sinr_rel"):
        main(args)


def test_build_default_design_is_sets(tmp_path, webapp_dir):
    pool = make_pool(300, seed=5)
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, index=False)
    out = tmp_path / "out"
    main(
        [
            "build",
            "--pool",
            str(pool_path),
            "--freeze",
            "2026-09-01",
            "--d1",
            "2026-09-15",
            "--seed",
            "3",
            "--batch-size",
            "25",
            "--arms",
            "recency,gap_first",
            "--webapp-dir",
            str(webapp_dir),
            "--out",
            str(out),
        ]
    )
    d = json.loads((out / "manifest.json").read_text())
    assert d["design"] == "sets"
    for name in ["index.html", "arm_A.html", "arm_B.html"]:
        assert (out / "site" / name).exists()


def _keyed_build(pool_path, webapp_dir, out, freeze, log_path, extra=()):
    return main(
        [
            "build",
            "--pool",
            str(pool_path),
            "--freeze",
            freeze,
            "--d1",
            "2026-11-01",
            "--batch-size",
            "25",
            "--webapp-dir",
            str(webapp_dir),
            "--design",
            "rotation",
            "--key-env",
            "WHAT_TO_ID_KEY",
            "--served-log",
            str(log_path),
            "--out",
            str(out),
            *extra,
        ]
    )


def test_keyed_daily_builds_keep_lists_and_log_letters_only(
    tmp_path, webapp_dir, monkeypatch, caplog
):
    monkeypatch.setenv("WHAT_TO_ID_KEY", "ab" * 32)
    caplog.set_level("INFO", logger="what_to_id")
    pool = make_pool(300, seed=5)
    day1, day2 = tmp_path / "pool1.parquet", tmp_path / "pool2.parquet"
    pool.iloc[:200].to_parquet(day1, index=False)
    pool.iloc[50:].to_parquet(day2, index=False)
    log_path = tmp_path / "state" / "served.parquet"
    assert _keyed_build(day1, webapp_dir, tmp_path / "o1", "2026-11-03", log_path) == 0
    assert _keyed_build(day2, webapp_dir, tmp_path / "o2", "2026-11-04", log_path) == 0

    d = json.loads((tmp_path / "o2" / "manifest.json").read_text())
    validate_manifest(d)
    assert d["assignment"] == "keyed" and len(d["key_fingerprint"]) == 12
    a1 = pd.read_parquet(tmp_path / "o1" / "assign.parquet").set_index("id")["arm"]
    a2 = pd.read_parquet(tmp_path / "o2" / "assign.parquet").set_index("id")["arm"]
    shared = a1.index.intersection(a2.index)
    assert len(shared) == 150 and (a1[shared] == a2[shared]).all()

    served = pd.read_parquet(log_path)
    assert sorted(served["build_date"].unique()) == ["2026-11-03", "2026-11-04"]
    assert set(served["label"]) == set(d["arm_labels"].values())
    text = caplog.text + served.astype(str).to_csv()
    for word in ("recency", "gap_first", "ab" * 32):
        assert word not in text


def test_build_records_the_grid_it_read(tmp_path, webapp_dir, monkeypatch):
    monkeypatch.setenv("WHAT_TO_ID_KEY", "ab" * 32)
    pool_path = tmp_path / "pool.parquet"
    make_pool(100, seed=5).to_parquet(pool_path, index=False)
    log_path = tmp_path / "served.parquet"
    assert _keyed_build(pool_path, webapp_dir, tmp_path / "o1", "2026-11-03", log_path) == 0
    assert (
        json.loads((tmp_path / "o1" / "manifest.json").read_text())["where_to_blitz_grid"] is None
    )
    (webapp_dir / "provenance.json").write_text(json.dumps({"manifest_hash": "f67acf37"}))
    assert _keyed_build(pool_path, webapp_dir, tmp_path / "o2", "2026-11-04", log_path) == 0
    d = json.loads((tmp_path / "o2" / "manifest.json").read_text())
    assert d["where_to_blitz_grid"] == "f67acf37"


def test_build_writes_the_code_commit_into_the_manifest(tmp_path, webapp_dir, monkeypatch):
    monkeypatch.setenv("WHAT_TO_ID_KEY", "ab" * 32)
    pool_path = tmp_path / "pool.parquet"
    make_pool(100, seed=5).to_parquet(pool_path, index=False)
    out = tmp_path / "o1"
    assert _keyed_build(pool_path, webapp_dir, out, "2026-11-03", tmp_path / "s.parquet") == 0
    manifest = json.loads((out / "manifest.json").read_text())
    record = json.loads((out / "build_record.json").read_text())
    assert manifest["code_commit"] == record["code_commit"]


def test_build_writes_a_public_build_record(tmp_path, webapp_dir, monkeypatch):
    monkeypatch.setenv("WHAT_TO_ID_KEY", "ab" * 32)
    pool_path = tmp_path / "pool.parquet"
    make_pool(100, seed=5).to_parquet(pool_path, index=False)
    log_path = tmp_path / "served.parquet"
    out = tmp_path / "o1"
    assert _keyed_build(pool_path, webapp_dir, out, "2026-11-03", log_path) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    record = json.loads((out / "build_record.json").read_text())

    assert "arm_labels" not in record
    assert "batches" not in record
    assert manifest["arm_labels"] not in record.values()
    assert record["pool_sha256"] == manifest["pool_sha256"]
    assert record["arms"] == manifest["arms"]
    for key in (
        "freeze",
        "d1",
        "seed",
        "batch_size",
        "max_batches",
        "design",
        "assignment",
        "arms",
        "key_fingerprint",
        "pool_sha256",
        "pool_rows",
        "served_rows",
        "where_to_blitz_ref",
        "where_to_blitz_grid",
        "embeddings_sha256",
        "reference_sha256",
        "created_at",
        "code_commit",
    ):
        assert key in record


def test_build_rejects_bad_date(tmp_path):
    with pytest.raises(SystemExit):
        main(["build", "--freeze", "2026-13-01", "--d1", "2026-09-15", "--pool", "x"])


def test_novelty_requires_reference(tmp_path, webapp_dir):
    pool_path = tmp_path / "pool.parquet"
    pool = make_pool(20)
    pool.to_parquet(pool_path, index=False)
    emb = tmp_path / "emb.npz"
    rng = np.random.default_rng(0)
    E = rng.normal(size=(20, 8)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    np.savez(emb, ids=pool["id"].to_numpy(), E=E, backbone="synthetic")
    args = [
        "build",
        "--pool",
        str(pool_path),
        "--freeze",
        "2026-09-01",
        "--d1",
        "2026-09-15",
        "--arms",
        "recency,novelty",
        "--embeddings",
        str(emb),
        "--webapp-dir",
        str(webapp_dir),
        "--out",
        str(tmp_path / "o"),
    ]
    with pytest.raises(SystemExit, match="reference"):
        main(args)
    main([*args, "--reference-embeddings", str(emb)])
    d = json.loads((tmp_path / "o" / "manifest.json").read_text())
    assert d["arms"] == ["recency", "novelty"]
    assert d["reference_sha256"] == d["embeddings_sha256"] and d["backbone"] == "synthetic"
    for b in d["batches"].values():
        assert b["n_embedded"] == len(b["ids"])
        assert 0.0 <= b["novelty"] <= 2.0
        if b["n_embedded"] >= 2:
            assert 0.0 <= b["cohesion"] <= 1.0
        else:
            assert b["cohesion"] is None
    label = {v: k for k, v in d["arm_labels"].items()}
    for lab in label:
        html = (tmp_path / "o" / "site" / f"arm_{lab}.html").read_text()
        assert "look-alike 0." in html and "unfamiliar 0." in html and "Look-alike</b>" in html


def test_similarity_requires_embeddings(tmp_path, webapp_dir):
    pool_path = tmp_path / "pool.parquet"
    make_pool(20).to_parquet(pool_path, index=False)
    with pytest.raises((SystemExit, FileNotFoundError)):
        main(
            [
                "build",
                "--pool",
                str(pool_path),
                "--freeze",
                "2026-09-01",
                "--d1",
                "2026-09-15",
                "--arms",
                "recency,similarity",
                "--webapp-dir",
                str(webapp_dir),
                "--out",
                str(tmp_path / "o"),
            ]
        )


def test_embeddings_directory_finds_embed_step_output(tmp_path):
    from what_to_id.artifacts import embedding_paths

    (tmp_path / "emb_Aves_bioclip25.npz").touch()
    (tmp_path / "emb_Insecta.npz").touch()
    out = embedding_paths(str(tmp_path), ["Aves", "Insecta", "Fungi"])
    assert out == {
        "Aves": tmp_path / "emb_Aves_bioclip25.npz",
        "Insecta": tmp_path / "emb_Insecta.npz",
    }
    (tmp_path / "emb_Aves_dinov2.npz").touch()
    with pytest.raises(SystemExit, match="several embedding files for Aves"):
        embedding_paths(str(tmp_path), ["Aves"])
    with pytest.raises(FileNotFoundError):
        embedding_paths(str(tmp_path / "empty"), ["Aves"])
