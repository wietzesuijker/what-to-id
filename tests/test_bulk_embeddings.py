"""Bulk workers checkpoint successful rows and reject ambiguous cache identities."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from what_to_id import bulk_embeddings as bulk


def test_csv_cluster_imports_work_without_pyarrow():
    code = """
import importlib.abc
import sys
class NoArrow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'pyarrow' or fullname.startswith('pyarrow.'):
            raise ModuleNotFoundError('PyArrow intentionally unavailable in CSV workers')
sys.meta_path.insert(0, NoArrow())
from what_to_id import bulk_embeddings, staging
assert 'pyarrow' not in sys.modules
"""
    root = Path(__file__).parents[1]
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def setup(tmp_path, monkeypatch, *, count=6, shard_size=3):
    source = tmp_path / "pool.csv"
    pd.DataFrame(
        {
            "id": list(range(count, 0, -1)),
            "photo_url": [f"https://example.test/{i}.jpg" for i in range(count, 0, -1)],
        }
    ).to_csv(source, index=False)
    manifest = bulk.prepare(source, tmp_path / "inputs", shard_size)
    work, out = tmp_path / "work", tmp_path / "out"
    calls = {"load": 0, "encoded": []}
    monkeypatch.setattr(bulk, "_verify_snapshot", lambda path: bulk.MODEL_HASHES)

    def loader(snapshot, device):
        calls["load"] += 1

        def preprocess(path):
            value = int(Path(path).read_text())
            if value < 0:
                raise ValueError("corrupt image")
            return np.full(1024, value, dtype=np.float32)

        def model(items):
            calls["encoded"].extend(int(row[0]) for row in items)
            return items

        return model, preprocess

    def stage(rows, cache, **kwargs):
        paths = []
        for row in rows.itertuples():
            path = bulk.embed.photo_path(cache, int(row.id), row.photo_url)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(str(row.id))
            paths.append(str(path))
        return rows.assign(local_path=paths)

    monkeypatch.setattr(bulk.embed, "stage_images", stage)
    return manifest, work, out, calls, loader


def run(manifest, work, out, loader, **kwargs):
    return bulk.run(
        manifest,
        [0, 1],
        Path("frozen-model"),
        work,
        out,
        device="cpu",
        batch=1,
        _loader=loader,
        **kwargs,
    )


def test_manifest_covers_exact_sorted_rows_in_bounded_shards(tmp_path, monkeypatch):
    manifest, *_ = setup(tmp_path, monkeypatch)
    data = json.loads(manifest.read_text())
    assert data["row_count"] == 6
    assert [part["ids"] for part in data["shards"]] == [[1, 2, 3], [4, 5, 6]]
    for part in data["shards"]:
        assert bulk.sha256_file(manifest.parent / part["file"]) == part["sha256"]


def test_worker_loads_once_and_cached_rerun_does_no_inference(tmp_path, monkeypatch):
    manifest, work, out, calls, loader = setup(tmp_path, monkeypatch)
    first = run(manifest, work, out, loader)
    assert first["completed_shards"] == [0, 1] and first["encoded_count"] == 6
    assert calls == {"load": 1, "encoded": [1, 2, 3, 4, 5, 6]}
    cached = run(manifest, work, out, loader)
    assert cached["reused_count"] == 6 and cached["encoded_count"] == 0
    assert cached["model_loaded"] is False
    for directory in out.glob("shard-*"):
        state = json.loads((directory / "state.json").read_text())
        assert state["status"] == "complete" and len(state["photos_sha256"]) == 3
        with np.load(directory / state["vectors_file"], allow_pickle=False) as vectors:
            assert vectors["E"].shape == (3, 1024)
            assert np.isfinite(vectors["E"]).all()


def test_soft_limit_checkpoints_then_resume_skips_successes(tmp_path, monkeypatch):
    manifest, work, out, calls, loader = setup(tmp_path, monkeypatch)
    ticks = iter([0, 0, 0, 1, 2, 2])
    stopped = run(manifest, work, out, loader, max_seconds=1.5, _clock=lambda: next(ticks))
    assert stopped["soft_limit_reached"] and stopped["encoded_count"] == 1
    resumed = run(manifest, work, out, loader)
    assert resumed["reused_count"] == 1 and resumed["encoded_count"] == 5
    assert calls["encoded"] == list(range(1, 7))


def test_offline_missing_and_corrupt_rows_retry_only_failures(tmp_path, monkeypatch):
    manifest, work, out, calls, loader = setup(tmp_path, monkeypatch)
    rows = pd.read_csv(manifest.parent / "input-00000.csv")
    for row in rows.iloc[:2].itertuples():
        path = bulk.embed.photo_path(work, int(row.id), row.photo_url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("-1" if row.id == 2 else "1")
    monkeypatch.setattr(bulk.embed, "stage_images", lambda *a, **k: pytest.fail("offline network"))
    result = bulk.run(
        manifest,
        [0],
        Path("model"),
        work,
        out,
        device="cpu",
        batch=1,
        offline_photos=True,
        _loader=loader,
    )
    assert result["partial_shards"] == [0] and result["encoded_count"] == 1
    state = json.loads((out / "shard-00000/state.json").read_text())
    assert set(state["failures"]) == {"2", "3"} and state["status"] == "partial"
    for row in rows.iloc[1:].itertuples():
        bulk.embed.photo_path(work, int(row.id), row.photo_url).write_text(str(row.id))
    result = bulk.run(
        manifest,
        [0],
        Path("model"),
        work,
        out,
        device="cpu",
        batch=1,
        offline_photos=True,
        _loader=loader,
    )
    assert result["completed_shards"] == [0] and result["reused_count"] == 1
    assert calls["encoded"] == [1, 2, 3]


@pytest.mark.parametrize("mutation", ["shard", "photo", "input", "runtime"])
def test_corruption_or_changed_identity_fails_before_reload(tmp_path, monkeypatch, mutation):
    manifest, work, out, calls, loader = setup(tmp_path, monkeypatch)
    run(manifest, work, out, loader)
    if mutation == "shard":
        (out / "shard-00000/vectors.npz").write_bytes(b"bad")
    elif mutation == "photo":
        row = pd.read_csv(manifest.parent / "input-00000.csv").iloc[0]
        bulk.embed.photo_path(work, int(row.id), row.photo_url).write_text("changed")
    elif mutation == "input":
        with (manifest.parent / "input-00000.csv").open("a") as stream:
            stream.write("7,https://example.test/7.jpg\n")
    else:
        monkeypatch.setattr(bulk, "_runtime", lambda device: {"changed": True})
    with pytest.raises(ValueError):
        run(manifest, work, out, loader)
    assert calls["load"] == 1


def test_writer_collision_fails_without_touching_lock(tmp_path, monkeypatch):
    manifest, work, out, calls, loader = setup(tmp_path, monkeypatch)
    lock = out / "shard-00000/.writer-lock"
    lock.mkdir(parents=True)
    with pytest.raises(ValueError, match="writer lock"):
        run(manifest, work, out, loader)
    assert lock.is_dir() and calls["load"] == 0


@pytest.mark.parametrize(
    "ids,urls", [([1, 1], ["https://x/a", "https://x/b"]), ([1], [None]), ([0], ["https://x/a"])]
)
def test_prepare_rejects_invalid_primary_rows(tmp_path, ids, urls):
    source = tmp_path / "bad.csv"
    pd.DataFrame({"id": ids, "photo_url": urls}).to_csv(source, index=False)
    with pytest.raises(ValueError):
        bulk.prepare(source, tmp_path / "inputs")


def test_invalid_batch_rejected_before_model_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(bulk, "_verify_snapshot", lambda path: pytest.fail("must fail early"))
    with pytest.raises(ValueError, match="batch"):
        bulk.run(tmp_path / "missing.json", [0], tmp_path / "model", tmp_path, tmp_path, batch=0)


def test_stop_request_commits_batch_before_returning(tmp_path, monkeypatch):
    manifest, work, out, calls, loader = setup(tmp_path, monkeypatch)
    stopped = run(manifest, work, out, loader, _stop_requested=lambda: len(calls["encoded"]) >= 1)
    assert stopped["soft_limit_reached"] and stopped["encoded_count"] == 1
    state = json.loads((out / "shard-00000/state.json").read_text())
    assert state["completed_count"] == 1 and state["status"] == "partial"
    resumed = run(manifest, work, out, loader)
    assert resumed["encoded_count"] == 5 and calls["encoded"] == list(range(1, 7))


def test_prepare_preserves_unknown_and_null_groups(tmp_path):
    pool = tmp_path / "pool.csv"
    pd.DataFrame(
        {
            "id": [1, 2],
            "photo_url": ["https://x/1", "https://x/2"],
            "iconic_taxon": [None, "Chromista"],
        }
    ).to_csv(pool, index=False)
    manifest = bulk.prepare(pool, tmp_path / "inputs")
    result = pd.read_csv(manifest.parent / "input-00000.csv")
    assert len(result) == 2 and result.iconic_taxon.isna().sum() == 1
    assert result.iconic_taxon.iloc[1] == "Chromista"
