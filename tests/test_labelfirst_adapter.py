import json
from importlib import metadata
from types import SimpleNamespace

import numpy as np
import pytest

from what_to_id import labelfirst_adapter
from what_to_id.arms import Novelty, load_embeddings


@pytest.mark.parametrize(
    ("ids", "E", "message"),
    [
        ([1, 1], np.eye(2), "duplicate ids"),
        ([1.5, 2.0], np.eye(2), "integer array"),
        (["1", "2"], np.eye(2), "integer array"),
        (np.array([2**63], dtype=np.uint64), [[1, 0]], "int64 range"),
        ([[1, 2]], np.eye(2), "integer array"),
        ([1, 2], np.ones((1, 2)), "does not match"),
        ([1, 2], np.empty((2, 0)), "at least one dimension"),
        ([1, 2], [[np.nan, 0], [0, 1]], "finite"),
        ([1, 2], [[np.inf, 0], [0, 1]], "finite"),
        ([1, 2], [[0, 0], [0, 1]], "unit norm"),
        ([1, 2], [[2, 0], [0, 1]], "unit norm"),
    ],
)
def test_embedding_cache_rejects_invalid_arrays(tmp_path, ids, E, message):
    path = tmp_path / "e.npz"
    np.savez(path, ids=ids, E=E)
    with pytest.raises(ValueError, match=message):
        load_embeddings(path)


def test_empty_embedding_cache_is_valid(tmp_path):
    path = tmp_path / "e.npz"
    np.savez(path, ids=np.array([], dtype=np.int64), E=np.empty((0, 3)))
    ids, E, backbone = load_embeddings(path)
    assert ids.shape == (0,) and E.shape == (0, 3) and backbone == ""


def test_novelty_rejects_different_backbones(tmp_path, pool):
    for name in ("candidate", "reference"):
        np.savez(tmp_path / f"{name}.npz", ids=[1, 2], E=np.eye(2), backbone=name)
    with pytest.raises(ValueError, match="backbones differ"):
        Novelty(tmp_path / "candidate.npz", tmp_path / "reference.npz").order(pool, seed=0)


@pytest.mark.parametrize("source", [None, "invalid-json", json.dumps({"url": "wheel"})])
def test_provenance_does_not_invent_revision(monkeypatch, source):
    dist = SimpleNamespace(version="0.11.0", read_text=lambda _: source)
    monkeypatch.setattr(labelfirst_adapter.metadata, "distribution", lambda _: dist)
    assert labelfirst_adapter.labelfirst_provenance() == {
        "version": "0.11.0",
        "commit": None,
        "dirty": None,
    }


def test_provenance_reads_resolved_git_commit(monkeypatch):
    source = {"vcs_info": {"vcs": "git", "commit_id": "a" * 40, "requested_revision": "main"}}
    dist = SimpleNamespace(version="0.11.0", read_text=lambda _: json.dumps(source))
    monkeypatch.setattr(labelfirst_adapter.metadata, "distribution", lambda _: dist)
    assert labelfirst_adapter.labelfirst_provenance()["commit"] == "a" * 40


def test_provenance_without_optional_package(monkeypatch):
    def absent(_):
        raise metadata.PackageNotFoundError("labelfirst")

    monkeypatch.setattr(labelfirst_adapter.metadata, "distribution", absent)
    assert labelfirst_adapter.labelfirst_provenance() is None


@pytest.mark.parametrize("dirty", [b"", b" M changed.py\n"])
def test_editable_provenance_reports_checkout_state(monkeypatch, dirty):
    source = {"dir_info": {"editable": True}, "url": "file:///tmp/source%20checkout"}
    dist = SimpleNamespace(version="0.11.0", read_text=lambda _: json.dumps(source))
    monkeypatch.setattr(labelfirst_adapter.metadata, "distribution", lambda _: dist)

    def git(args, **_):
        assert args[:3] == ["git", "-C", "/tmp/source checkout"]
        return b"abc123\n" if args[3] == "rev-parse" else dirty

    monkeypatch.setattr(labelfirst_adapter.subprocess, "check_output", git)
    assert labelfirst_adapter.labelfirst_provenance() == {
        "version": "0.11.0",
        "commit": "abc123",
        "dirty": bool(dirty),
    }
