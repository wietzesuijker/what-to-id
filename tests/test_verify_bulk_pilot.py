import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from what_to_id.manifest import sha256_file

spec = importlib.util.spec_from_file_location(
    "bulk_gate", Path(__file__).parents[1] / "benchmarks/verify_bulk_pilot.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def pilot(tmp_path):
    ids = np.arange(1, 513)
    vectors = np.full((512, 1024), 1 / 32, dtype=np.float32)
    reference = tmp_path / "reference.npz"
    np.savez(reference, ids=ids, E=vectors)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"shards": [{"index": 0, "ids": ids.tolist(), "count": 512, "sha256": "source"}]}
        )
    )
    photos = tmp_path / "photos.json"
    photos.write_text(json.dumps({str(value): "a" * 64 for value in ids}))
    output = tmp_path / "results"
    directory = output / "shard-00000"
    directory.mkdir(parents=True)
    path = directory / "vectors.npz"
    np.savez(path, ids=ids[::-1], E=vectors[::-1])
    state = {
        "status": "complete",
        "failures": {},
        "identity": {
            "manifest_sha256": sha256_file(manifest),
            "shard_input_sha256": "source",
            "model_revision": gate.MODEL_REVISION,
            "model_sha256": gate.MODEL_HASHES,
        },
        "vectors_file": path.name,
        "vectors_sha256": sha256_file(path),
        "photos_sha256": json.loads(photos.read_text()),
        "batches": [{"elapsed_seconds": 10}],
    }
    (directory / "state.json").write_text(json.dumps(state))
    return manifest, output, reference, photos


def test_gate_aligns_ids_and_accepts_matching_vectors(tmp_path):
    result = gate.verify(*pilot(tmp_path))
    assert result["status"] == "passes"
    assert result["count"] == 512
    assert result["minimum_cosine"] == pytest.approx(1)


@pytest.mark.parametrize(
    "mutation", ["photo", "vector", "source", "incomplete", "zero_cpu", "slow"]
)
def test_gate_blocks_bulk_for_incompatible_or_incomplete_evidence(tmp_path, mutation):
    manifest, output, reference, photos = pilot(tmp_path)
    pointer = output / "shard-00000/state.json"
    state = json.loads(pointer.read_text())
    if mutation == "photo":
        state["photos_sha256"]["1"] = "b" * 64
    elif mutation == "source":
        state["identity"]["manifest_sha256"] = "wrong"
    elif mutation == "incomplete":
        state["status"] = "partial"
    elif mutation == "slow":
        state["batches"][0]["elapsed_seconds"] = 200
    else:
        path = reference if mutation == "zero_cpu" else output / "shard-00000/vectors.npz"
        with np.load(path) as saved:
            ids, vectors = saved["ids"], saved["E"]
        vectors[0] = 0 if mutation == "zero_cpu" else -vectors[0]
        np.savez(path, ids=ids, E=vectors)
        state["vectors_sha256"] = sha256_file(output / "shard-00000/vectors.npz")
    pointer.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        gate.verify(manifest, output, reference, photos)
