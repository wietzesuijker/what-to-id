"""Tests for what_to_id.embed that run without torch, open_clip or PIL."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from what_to_id import embed

DIM = 8


def _pool(ids, taxon="Aves"):
    return pd.DataFrame(
        {
            "id": np.asarray(ids, dtype=np.int64),
            "iconic_taxon": taxon,
            "photo_url": [f"https://example.test/{i}.jpg" for i in ids],
            "lat": [49.0 + i * 0.01 for i in ids],
            "lon": [-123.0 - i * 0.01 for i in ids],
        }
    )


class _Resp:
    def __init__(self, content=b"", status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(embed.time, "sleep", slept.append)
    return slept


def _fake_get(script):
    """script: url -> outcomes consumed in order: bytes, an int status or an Exception."""
    calls = {}

    def get(url, timeout=None, headers=None):
        assert headers["User-Agent"] == embed.USER_AGENT
        calls[url] = calls.get(url, 0) + 1
        outcome = script[url].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, int):
            return _Resp(status=outcome)
        return _Resp(outcome)

    get.calls = calls
    return get


# ---- staging ----------------------------------------------------------------------------
def test_stage_images_success_retry_and_permanent_failure(tmp_path, monkeypatch, no_sleep):
    pool = _pool([1, 2, 3])
    script = {
        "https://example.test/1.jpg": [b"one"],
        "https://example.test/2.jpg": [ConnectionError("flaky"), b"two"],
        "https://example.test/3.jpg": [TimeoutError("t"), TimeoutError("t"), TimeoutError("t")],
    }
    get = _fake_get(script)
    monkeypatch.setattr(embed.requests, "get", get)
    logs = []

    out = embed.stage_images(pool, tmp_path, workers=2, retries=3, log=logs.append)

    assert list(out["local_path"]) == [
        str(embed.photo_path(tmp_path, i, f"https://example.test/{i}.jpg")) for i in (1, 2, 3)
    ]
    assert embed.photo_path(tmp_path, 1, "https://example.test/1.jpg").read_bytes() == b"one"
    assert embed.photo_path(tmp_path, 2, "https://example.test/2.jpg").read_bytes() == b"two"
    assert not embed.photo_path(tmp_path, 3, "https://example.test/3.jpg").exists()
    assert not list((tmp_path / "photos").glob("*.part"))
    assert get.calls["https://example.test/2.jpg"] == 2
    assert get.calls["https://example.test/3.jpg"] == 3
    failed = out.attrs["failed"]
    assert [i for i, _ in failed] == [3]
    assert "TimeoutError" in failed[0][1]
    # backoff 0.5, 1 for id 3 plus 0.5 for id 2; the last attempt never sleeps
    assert sorted(no_sleep) == [0.5, 0.5, 1.0]
    assert any("1 failed" in line for line in logs)


STATIC = "https://static.inaturalist.org/photos/9/medium.jpg"
S3 = "https://inaturalist-open-data.s3.amazonaws.com/photos/9/medium.jpg"


def _stage_one(tmp_path, monkeypatch, url, script, retries=3):
    pool = _pool([9])
    pool["photo_url"] = url
    get = _fake_get(script)
    monkeypatch.setattr(embed.requests, "get", get)
    out = embed.stage_images(pool, tmp_path, workers=1, retries=retries, log=lambda _: None)
    return out, get


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (STATIC, S3),
        (S3, STATIC),
        (STATIC + "?v=2", S3 + "?v=2"),
        ("https://example.test/photos/9/medium.jpg", None),
        ("https://static.inaturalist.org.evil.test/photos/9/medium.jpg", None),
    ],
)
def test_alternate_photo_url(url, expected):
    assert embed._alternate_photo_url(url) == expected


@pytest.mark.parametrize(("url", "other", "status"), [(STATIC, S3, 403), (S3, STATIC, 404)])
def test_stage_images_moved_inat_photo_uses_other_host(
    tmp_path, monkeypatch, no_sleep, url, other, status
):
    out, get = _stage_one(tmp_path, monkeypatch, url, {url: [status], other: [b"jpg"]})

    assert out.attrs["failed"] == []
    assert embed.photo_path(tmp_path, 9, url).read_bytes() == b"jpg"
    assert get.calls == {url: 1, other: 1}
    assert no_sleep == []


def test_stage_images_inat_both_hosts_fail_names_both(tmp_path, monkeypatch, no_sleep):
    script = {STATIC: [403, 403], S3: [403, 403]}
    out, get = _stage_one(tmp_path, monkeypatch, STATIC, script, retries=2)

    ((obs_id, err),) = out.attrs["failed"]
    assert obs_id == 9
    assert STATIC in err and S3 in err
    assert not embed.photo_path(tmp_path, 9, STATIC).exists()
    assert get.calls == {STATIC: 2, S3: 2}


def test_stage_images_non_inat_403_makes_no_alternate_request(tmp_path, monkeypatch, no_sleep):
    url = "https://example.test/photos/9/medium.jpg"
    out, get = _stage_one(tmp_path, monkeypatch, url, {url: [403, 403, 403]})

    assert [i for i, _ in out.attrs["failed"]] == [9]
    assert get.calls == {url: 3}


def test_stage_images_inat_500_uses_normal_retries(tmp_path, monkeypatch, no_sleep):
    out, get = _stage_one(tmp_path, monkeypatch, STATIC, {STATIC: [500, b"jpg"]})

    assert out.attrs["failed"] == []
    assert get.calls == {STATIC: 2}
    assert no_sleep == [0.5]


def test_stage_images_skips_existing_and_bad_url(tmp_path, monkeypatch, no_sleep):
    (tmp_path / "photos").mkdir()
    embed.photo_path(tmp_path, 7, "https://example.test/7.jpg").write_bytes(b"cached")
    pool = _pool([7, 8])
    pool.loc[pool["id"] == 8, "photo_url"] = None
    get = _fake_get({})
    monkeypatch.setattr(embed.requests, "get", get)

    out = embed.stage_images(pool, tmp_path, log=lambda _: None)

    assert get.calls == {}
    assert embed.photo_path(tmp_path, 7, "https://example.test/7.jpg").read_bytes() == b"cached"
    assert out.attrs["failed"] == [(8, "no photo_url")]


@pytest.mark.parametrize("workers", [0, 1, 3])
def test_stage_images_bounds_queued_downloads_and_preserves_failure_order(
    tmp_path, monkeypatch, workers
):
    release = Event()
    executor = embed.ThreadPoolExecutor
    limit = 2 * max(1, workers)
    outstanding = 0
    peak = 0

    class TrackedExecutor(executor):
        def submit(self, *args, **kwargs):
            nonlocal outstanding, peak
            outstanding += 1
            peak = max(peak, outstanding)
            if outstanding > limit:
                release.set()
                pytest.fail("downloads queued beyond the bounded window")
            future = super().submit(*args, **kwargs)

            def result():
                nonlocal outstanding
                release.set()
                value = future.result(timeout=5)
                outstanding -= 1
                return value

            return SimpleNamespace(result=result)

    def fetch(obs_id, *args, **kwargs):
        assert release.wait(timeout=5)
        return obs_id, "failed"

    monkeypatch.setattr(embed, "ThreadPoolExecutor", TrackedExecutor)
    monkeypatch.setattr(embed, "_fetch_one", fetch)
    ids = list(range(17, 0, -1))
    out = embed.stage_images(_pool(ids), tmp_path, workers=workers, log=lambda _: None)

    assert peak == limit
    assert outstanding == 0
    assert out.attrs["failed"] == [(i, "failed") for i in ids]


def test_stage_images_empty_pool(tmp_path, monkeypatch):
    def fetch(*args, **kwargs):
        pytest.fail("empty input must not download photos")

    monkeypatch.setattr(embed, "_fetch_one", fetch)
    out = embed.stage_images(_pool([]), tmp_path, log=lambda _: None)
    assert out.empty
    assert out.attrs["failed"] == []


# ---- embedding core ---------------------------------------------------------------------
def _fake_loader(backbone, device):
    """numpy stand-in: preprocess reads the file, model_fn hashes bytes into a DIM vector."""
    rng_dim = DIM

    def preprocess(path):
        data = Path(path).read_bytes()
        if not data:
            raise ValueError("empty image")
        seed = int.from_bytes(data[:4].ljust(4, b"\0"), "little")
        return np.random.default_rng(seed).normal(size=rng_dim) * 3.0

    def model_fn(x):
        return x

    return model_fn, preprocess


def test_embed_paths_drops_failures_and_normalises(tmp_path):
    good = tmp_path / "a.jpg"
    good.write_bytes(b"abcd")
    bad = tmp_path / "missing.jpg"
    model_fn, preprocess = _fake_loader("bioclip25", "cpu")
    E, kept, failed = embed._embed_paths(
        [str(good), str(bad), str(good)],
        embed._numpy_batch_fn(model_fn),
        preprocess,
        batch=1,
        open_image=embed._passthrough_path,
        log=lambda _: None,
    )
    assert kept == [0, 2]
    assert [i for i, _ in failed] == [1]
    assert E.shape == (2, DIM) and E.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-6)


def test_embed_paths_empty():
    E, kept, failed = embed._embed_paths([], embed._numpy_batch_fn(lambda x: x), lambda p: p)
    assert E.shape[0] == 0 and kept == [] and failed == []


def _stage_bytes(monkeypatch, tmp_path, ids, empty=()):
    script = {
        f"https://example.test/{i}.jpg": [b"" if i in empty else i.to_bytes(4, "little")]
        for i in ids
    }
    monkeypatch.setattr(embed.requests, "get", _fake_get(script))


def test_embed_group_round_trip_and_sidecar(tmp_path, monkeypatch, no_sleep):
    ids = [10, 11, 12]
    _stage_bytes(monkeypatch, tmp_path, ids)
    pool = _pool(ids, taxon="Insecta")

    out = embed.embed_group(
        pool, cache_dir=tmp_path, batch=2, _loader=_fake_loader, log=lambda _: None
    )

    assert out == tmp_path / "emb_Insecta_bioclip25.npz"
    z = np.load(out)
    assert set(z.files) == {"ids", "E", "lat", "lon", "backbone", "emb_device"}
    assert z["ids"].dtype == np.int64 and list(z["ids"]) == ids
    assert z["E"].dtype == np.float32 and z["E"].shape == (3, DIM)
    np.testing.assert_allclose(np.linalg.norm(z["E"], axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(z["lat"], pool["lat"])
    assert str(z["backbone"]) == "bioclip25"
    assert str(z["emb_device"]) == "fake"
    side = json.loads(out.with_suffix(".json").read_text())
    assert side["n"] == 3
    assert side["model_id"] == "hf-hub:imageomics/bioclip-2.5-vith14"
    assert side["failed_ids"] == []
    assert side["created_at"]
    loaded_ids, loaded_E = embed.load_embeddings(out)
    np.testing.assert_array_equal(loaded_ids, ids)
    np.testing.assert_array_equal(loaded_E, z["E"])


def test_embed_group_skip_if_covered_and_append_missing(tmp_path, monkeypatch, no_sleep):
    _stage_bytes(monkeypatch, tmp_path, [1, 2])
    first = embed.embed_group(
        _pool([1, 2]), cache_dir=tmp_path, _loader=_fake_loader, log=lambda _: None
    )
    ids0, E0 = embed.load_embeddings(first)

    # covered: no network call, no rewrite
    def boom(*a, **k):
        raise AssertionError("must not download when cache covers the group")

    monkeypatch.setattr(embed.requests, "get", boom)
    mtime = first.stat().st_mtime_ns
    logs = []
    again = embed.embed_group(
        _pool([2, 1]), cache_dir=tmp_path, _loader=_fake_loader, log=logs.append
    )
    assert again == first and first.stat().st_mtime_ns == mtime
    assert any("skipping" in line for line in logs)

    # append: only id 3 is fetched (4 fails to open and lands in failed_ids)
    _stage_bytes(monkeypatch, tmp_path, [3, 4], empty={4})
    out = embed.embed_group(
        _pool([1, 2, 3, 4]), cache_dir=tmp_path, _loader=_fake_loader, log=lambda _: None
    )
    ids1, E1 = embed.load_embeddings(out)
    assert list(ids1) == [1, 2, 3]
    np.testing.assert_array_equal(E1[:2], E0)
    z = np.load(out)
    assert len(z["lat"]) == 3 and len(z["lon"]) == 3
    side = json.loads(out.with_suffix(".json").read_text())
    assert side["n"] == 3 and side["failed_ids"] == [4]

    # a later pass retries the failure, then clears it from the sidecar
    _stage_bytes(monkeypatch, tmp_path, [4])
    out = embed.embed_group(
        _pool([1, 2, 3, 4]), cache_dir=tmp_path, _loader=_fake_loader, log=lambda _: None
    )
    ids2, _ = embed.load_embeddings(out)
    assert list(ids2) == [1, 2, 3, 4]
    assert json.loads(out.with_suffix(".json").read_text())["failed_ids"] == []


def test_embed_group_unknown_backbone(tmp_path):
    with pytest.raises(KeyError, match="unknown backbone"):
        embed.embed_group(_pool([1]), backbone="resnet", cache_dir=tmp_path, _loader=_fake_loader)


# ---- backbones and availability ---------------------------------------------------------
def test_backbone_table():
    assert embed.DEFAULT_BACKBONE == "bioclip25"
    assert embed.BACKBONES["bioclip25"] == (
        "open_clip",
        "hf-hub:imageomics/bioclip-2.5-vith14",
        1024,
    )
    assert embed.BACKBONES["bioclip2"] == ("open_clip", "hf-hub:imageomics/bioclip-2", 768)
    assert embed.BACKBONES["dinov2"] == ("torchhub", "facebookresearch/dinov2:dinov2_vits14", 384)
    assert embed.emb_cache_path("data", "Aves", "dinov2") == Path("data/emb_Aves_dinov2.npz")


def test_load_backbone_without_deps_points_at_extra(monkeypatch):
    monkeypatch.setattr(embed, "embeddings_available", lambda: False)
    with pytest.raises(ImportError, match=r'uv pip install -e "\.\[embed\]"'):
        embed.load_backbone("bioclip25", "cpu")
    with pytest.raises(ImportError, match="optional"):
        embed.pick_device()
    with pytest.raises(KeyError):
        embed.load_backbone("nope", "cpu")


def test_l2_normalise_handles_zero_rows():
    E = embed._l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(E[0], [0.6, 0.8])
    np.testing.assert_array_equal(E[1], [0.0, 0.0])


# ---- separability -----------------------------------------------------------------------
def test_separability_report_wraps_labelfirst(monkeypatch):
    pytest.importorskip("labelfirst")
    import labelfirst

    seen = {}

    def fake_score(X, labels, **kw):
        seen["shape"] = X.shape
        seen["labels"] = labels
        return 87.5

    monkeypatch.setattr(labelfirst, "separability_score", fake_score)
    E = np.eye(4, dtype=np.float32)
    rep = embed.separability_report(E, ["a", "b", None, "a"])
    assert rep == {"separability_pct": 87.5, "n": 3}
    assert seen["shape"] == (4, 4) and seen["labels"] == ["a", "b", None, "a"]
    with pytest.raises(ValueError, match="align"):
        embed.separability_report(E, ["a"])


def test_changed_url_fetches_new_bytes_despite_old_photo(tmp_path, monkeypatch):
    pool = _pool([1])
    old = embed.photo_path(tmp_path, 1, pool.photo_url.iloc[0])
    old.parent.mkdir()
    old.write_bytes(b"old")
    pool.loc[0, "photo_url"] = "https://example.test/new.jpg"
    get = _fake_get({pool.photo_url.iloc[0]: [b"new"]})
    monkeypatch.setattr(embed.requests, "get", get)
    staged = embed.stage_images(pool, tmp_path)
    assert Path(staged.local_path.iloc[0]).read_bytes() == b"new"
    assert old.read_bytes() == b"old"
    assert get.calls == {pool.photo_url.iloc[0]: 1}


def test_all_failed_images_can_retry(tmp_path, monkeypatch, no_sleep):
    pool = _pool([1])
    monkeypatch.setattr(embed.requests, "get", _fake_get({pool.photo_url.iloc[0]: [b""] * 3}))
    with pytest.raises(ValueError, match="no images embedded"):
        embed.embed_group(pool, cache_dir=tmp_path, _loader=_fake_loader)
    assert not embed.emb_cache_path(tmp_path, "Aves", "bioclip25").exists()
    monkeypatch.setattr(embed.requests, "get", _fake_get({pool.photo_url.iloc[0]: [b"good"]}))
    out = embed.embed_group(pool, cache_dir=tmp_path, _loader=_fake_loader)
    assert embed.load_embeddings(out)[0].tolist() == [1]


def test_corrupt_download_is_removed_before_archiving_then_refetched(tmp_path, monkeypatch):
    import io

    Image = pytest.importorskip("PIL.Image")
    jpeg = io.BytesIO()
    Image.new("RGB", (2, 2), "green").save(jpeg, format="JPEG")
    pool = _pool([1])
    get = _fake_get({pool.photo_url.iloc[0]: [b"not a JPEG", jpeg.getvalue()]})
    monkeypatch.setattr(embed.requests, "get", get)
    staged = embed.stage_images(pool, tmp_path)
    assert embed.validate_staged_images(staged) == [(1, "UnidentifiedImageError")]
    assert not Path(staged.local_path.iloc[0]).exists()
    staged = embed.stage_images(pool, tmp_path)
    assert embed.validate_staged_images(staged) == []
    assert get.calls == {pool.photo_url.iloc[0]: 2}


def test_save_embeddings_keeps_previous_cache_when_compression_fails(tmp_path, monkeypatch):
    target = tmp_path / "emb_Aves_bioclip25.npz"
    args = dict(
        ids=np.array([1]),
        E=np.array([[1.0, 0.0]]),
        lat=np.array([0.0]),
        lon=np.array([0.0]),
        backbone="bioclip25",
        emb_device="fixture",
        failed_ids=[],
    )
    embed.save_embeddings(target, **args)
    previous = target.read_bytes()
    previous_metadata = target.with_suffix(".json").read_bytes()

    def broken_save(destination, **kwargs):
        if hasattr(destination, "write"):
            destination.write(b"incomplete zip")
        else:
            Path(destination).write_bytes(b"incomplete zip")
        raise OSError("interrupted compression")

    monkeypatch.setattr(embed.np, "savez_compressed", broken_save)
    with pytest.raises(OSError, match="interrupted compression"):
        embed.save_embeddings(target, **args)
    assert target.read_bytes() == previous
    assert target.with_suffix(".json").read_bytes() == previous_metadata
    np.testing.assert_array_equal(embed.load_embeddings(target)[0], [1])
    assert set(tmp_path.iterdir()) == {target, target.with_suffix(".json")}


@pytest.mark.parametrize("batch", [0, -1])
def test_embed_group_rejects_batch_before_staging(tmp_path, monkeypatch, batch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid batch must fail before staging or loading")

    monkeypatch.setattr(embed, "stage_images", unexpected)
    with pytest.raises(ValueError, match="batch size must be positive"):
        embed.embed_group(_pool([1]), cache_dir=tmp_path, batch=batch, _loader=unexpected)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "output",
    [
        np.ones(8),
        np.ones((2, 8)),
        np.ones((1, 0)),
        np.full((1, 8), np.nan),
        np.full((1, 8), np.inf),
        np.zeros((1, 8)),
    ],
)
def test_invalid_encoder_output_fails_first_batch_without_cache(tmp_path, monkeypatch, output):
    _stage_bytes(monkeypatch, tmp_path, [1, 2])
    calls = []

    def loader(backbone, device):
        def model(items):
            calls.append(len(items))
            return output

        return model, lambda path: np.ones(8)

    with pytest.raises(ValueError, match="encoder output"):
        embed.embed_group(_pool([1, 2]), cache_dir=tmp_path, batch=1, _loader=loader)
    assert calls == [1]
    assert not list(tmp_path.glob("emb_*.npz"))
    assert not list(tmp_path.glob("emb_*.json"))


def test_embed_paths_rejects_changed_width_before_third_batch():
    calls = []

    def batch_fn(items):
        calls.append(len(items))
        return np.ones((1, 8 if len(calls) == 1 else 7))

    with pytest.raises(ValueError, match="width changed"):
        embed._embed_paths(
            ["a", "b", "c"],
            batch_fn,
            lambda path: path,
            batch=1,
            open_image=lambda path: path,
        )
    assert calls == [1, 1]
