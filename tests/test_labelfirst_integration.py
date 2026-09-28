"""Required real-library checks, run with the similarity extra and -m labelfirst."""

import numpy as np
import pytest

from what_to_id.arms import Similarity
from what_to_id.embed import separability_report

from .conftest import make_pool

pytestmark = pytest.mark.labelfirst


def test_similarity_preserves_ids_when_embedding_cache_is_reordered(tmp_path):
    import labelfirst

    assert callable(labelfirst.CoreSet)
    pool = make_pool(8, groups=["Aves"])
    ids = pool["id"].to_numpy()[:6]
    E = np.random.default_rng(7).normal(size=(6, 8)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    orders = []
    for permutation in (np.arange(6), np.array([5, 3, 1, 2, 4, 0])):
        path = tmp_path / "e.npz"
        np.savez(path, ids=ids[permutation], E=E[permutation])
        arm = Similarity(path, batch_size=2)
        order = arm.order(pool, seed=9)
        np.testing.assert_array_equal(arm.order(pool, seed=9), order)
        assert sorted(order.tolist()) == list(range(8))
        assert set(order[-2:]) == {6, 7}
        orders.append(order)
    np.testing.assert_array_equal(*orders)


def test_separability_uses_same_known_rows_as_real_library():
    import labelfirst

    E = np.array([[1, 0], [1, 0.01], [0, 1], [0.01, 1], [-1, 0], [0, -1], [-1, -1]])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    report = separability_report(E, ["a", "a", "b", "b", None, float("nan"), ""])
    assert report["n"] == 4
    assert report["separability_pct"] == labelfirst.separability_score(E[:4], ["a", "a", "b", "b"])
    assert report["separability_pct"] == 100.0
