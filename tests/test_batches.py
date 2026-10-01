import pytest

from what_to_id.arms import GapFirst, Recency
from what_to_id.assign import assign
from what_to_id.batches import IDENTIFY_URL, batch_key, build_batches, cut_batches, identify_url


def test_batch_key_sorts_batch_numbers_as_numbers():
    ids = ["recency-Aves-1000", "recency-Aves-101", "gap_first-All_biodiversity-002"]
    assert sorted(ids, key=batch_key) == [
        "gap_first-All_biodiversity-002",
        "recency-Aves-101",
        "recency-Aves-1000",
    ]


def test_cut_batches():
    assert cut_batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert cut_batches([], 3) == []
    with pytest.raises(ValueError):
        cut_batches([1], 0)


def test_identify_url_shape_and_escaping():
    url = identify_url([3, 1, 2])
    assert url.startswith(IDENTIFY_URL + "?")
    assert "id=3,1,2" in url
    assert "place_id=7085" in url
    url = identify_url([1], extra={"taxon name": "a&b"})
    assert "taxon+name=a%26b" in url
    with pytest.raises(ValueError):
        identify_url([])


def test_identify_url_length_guard():
    with pytest.raises(ValueError, match="chars"):
        identify_url(list(range(10**9, 10**9 + 1000)))


def test_build_batches(pool):
    pool = pool.copy()
    pool["cell_score"] = 0.5
    arms = {"recency": Recency(), "gap_first": GapFirst()}
    a = assign(pool, list(arms), seed=0)
    b = build_batches(pool, a, arms, size=7, seed=0)
    assert list(b.columns) == ["batch_id", "arm", "group", "position", "id"]
    assert b["id"].is_unique and set(b["id"]) == set(pool["id"])
    # Every id is in the arm it was assigned to.
    m = b.merge(a, on="id", suffixes=("", "_a"))
    assert (m["arm"] == m["arm_a"]).all()
    for bid, sub in b.groupby("batch_id"):
        assert sorted(sub["position"]) == list(range(len(sub)))
        assert len(sub) <= 7
        arm, group, i = bid.rsplit("-", 2)
        assert arm == sub["arm"].iloc[0] and group == sub["group"].iloc[0] and len(i) == 3
    assert b.groupby(["arm", "group"])["batch_id"].nunique().gt(0).all()


def test_build_batches_validation(pool):
    a = assign(pool, ["recency"], seed=0)
    with pytest.raises(ValueError, match="Arm object"):
        build_batches(pool, a, {}, size=5, seed=0)
    with pytest.raises(ValueError, match="no assignment"):
        build_batches(pool, a.head(10), {"recency": Recency()}, size=5, seed=0)
    with pytest.raises(ValueError):
        build_batches(pool, a, {"recency": Recency()}, size=0, seed=0)


def test_build_batches_max_batches(pool):
    pool = pool.copy()
    pool["cell_score"] = 0.5
    arms = {"recency": Recency(), "gap_first": GapFirst()}
    a = assign(pool, list(arms), seed=0)
    full = build_batches(pool, a, arms, size=7, seed=0)
    capped = build_batches(pool, a, arms, size=7, seed=0, max_batches=2)
    nb = capped.groupby(["arm", "group"])["batch_id"].nunique()
    assert (nb <= 2).all()
    # The kept batches are the highest-priority prefix of the full ordering.
    for (arm, group), sub in capped.groupby(["arm", "group"]):
        full_sub = full[(full["arm"] == arm) & (full["group"] == group)]
        kept_ids = set(sub["id"])
        full_ids_in_order = full_sub.sort_values(["batch_id", "position"])["id"].tolist()
        assert kept_ids == set(full_ids_in_order[: len(kept_ids)])
    unlimited = build_batches(pool, a, arms, size=7, seed=0, max_batches=None)
    assert unlimited["id"].tolist() == full["id"].tolist()
    with pytest.raises(ValueError):
        build_batches(pool, a, arms, size=7, seed=0, max_batches=0)


def test_ordering_cache_reuses_only_exact_subsets(pool, tmp_path):
    import pandas as pd

    class CountingRecency(Recency):
        calls = 0

        def order(self, pool, *, seed):
            self.calls += 1
            return super().order(pool, seed=seed)

    arm = CountingRecency()
    assignment = assign(pool, ["recency"], seed=0)
    kwargs = dict(size=7, seed=0, cache_dir=tmp_path, cache_context="inputs-v1")
    expected = build_batches(pool, assignment, {"recency": Recency()}, size=7, seed=0)
    first = build_batches(pool, assignment, {"recency": arm}, **kwargs)
    calls = arm.calls
    second = build_batches(pool, assignment, {"recency": arm}, **kwargs)
    assert arm.calls == calls
    pd.testing.assert_frame_equal(first, expected)
    pd.testing.assert_frame_equal(second, expected)
    reduced = pool.iloc[1:].reset_index(drop=True)
    result = build_batches(reduced, assignment, {"recency": arm}, **kwargs)
    assert arm.calls == calls + 1
    pd.testing.assert_frame_equal(
        result, build_batches(reduced, assignment, {"recency": Recency()}, size=7, seed=0)
    )
    # A new cap cuts the same cached full ordering, without reordering.
    calls = arm.calls
    capped = build_batches(pool, assignment, {"recency": arm}, max_batches=1, **kwargs)
    assert arm.calls == calls
    pd.testing.assert_frame_equal(
        capped,
        build_batches(pool, assignment, {"recency": Recency()}, size=7, seed=0, max_batches=1),
    )


@pytest.mark.parametrize("change", ["rows", "score", "seed", "size", "context", "dtype"])
def test_ordering_cache_invalidates_changed_inputs(pool, tmp_path, change):
    from unittest.mock import Mock

    pool = pool.assign(cell_score=0.5)
    assignment = assign(pool, ["recency"], seed=0)
    arm = Mock(name="arm")
    arm.name = "recency"
    arm.order.side_effect = Recency().order
    kwargs = dict(size=7, seed=0, cache_dir=tmp_path, cache_context="inputs-v1")
    build_batches(pool, assignment, {"recency": arm}, **kwargs)
    previous = arm.order.call_count
    if change == "rows":
        pool = pool.iloc[::-1]
    elif change == "score":
        pool = pool.assign(cell_score=0.7)
    elif change == "dtype":
        pool["cell_score"] = pool["cell_score"].astype("float32")
    else:
        key = "cache_context" if change == "context" else change
        kwargs[key] = "inputs-v2" if change == "context" else kwargs[key] + 1
    build_batches(pool, assignment, {"recency": arm}, **kwargs)
    assert arm.order.call_count == 2 * previous


@pytest.mark.parametrize("damage", ["truncated", "zip", "npy", "duplicate", "float", "checksum"])
def test_ordering_cache_recovers_corruption(pool, tmp_path, damage):
    import numpy as np
    import pandas as pd

    assignment = assign(pool, ["recency"], seed=0)
    kwargs = dict(size=7, seed=0, cache_dir=tmp_path, cache_context="inputs-v1")
    expected = build_batches(pool, assignment, {"recency": Recency()}, **kwargs)
    for path in tmp_path.glob("*.npz"):
        if damage in {"truncated", "zip"}:
            path.write_bytes(b"broken" if damage == "truncated" else path.read_bytes()[:40])
        elif damage == "npy":
            with path.open("wb") as stream:
                np.save(stream, np.arange(3))
        else:
            with np.load(path) as saved:
                order = saved["order"]
                checksum = saved["sha256"]
            if damage == "duplicate":
                order[:] = 0
            elif damage == "float":
                order = order.astype(float)
            else:
                order = order[::-1]
            np.savez(path, order=order, sha256=checksum)
    result = build_batches(pool, assignment, {"recency": Recency()}, **kwargs)
    pd.testing.assert_frame_equal(result, expected)
