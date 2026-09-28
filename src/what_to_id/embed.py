"""Photo staging and image embeddings for the similarity arm.

Ported from the shape of where-to-blitz/research/exp_discovery_offline.py (stage_images,
_load_bioclip, _load_dinov2, embed_images and the npz cache), not the experiment itself.

torch, open_clip and PIL are optional; install with ``uv pip install -e ".[embed]"``.
Everything heavy is imported lazily inside functions so the rest of the package stays light.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from what_to_id.inat import USER_AGENT

INSTALL_HINT = 'uv pip install -e ".[embed]"'

# name -> (loader kind, model id, embedding dim)
BACKBONES: dict[str, tuple[str, str, int]] = {
    # default, ViT-H/14
    "bioclip25": ("open_clip", "hf-hub:imageomics/bioclip-2.5-vith14", 1024),
    # ViT-L/14, lighter
    "bioclip2": ("open_clip", "hf-hub:imageomics/bioclip-2", 768),
    # for the separability comparison against where-to-blitz national caches
    "dinov2": ("torchhub", "facebookresearch/dinov2:dinov2_vits14", 384),
    # gated on the Hub: needs a token that accepted the DINOv3 licence (HF_HOME/token or HF_TOKEN)
    "dinov3": ("transformers", "facebook/dinov3-vitl16-pretrain-lvd1689m", 1024),
    "dinov3b": ("transformers", "facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
}
DEFAULT_BACKBONE = "bioclip25"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---- availability and device ------------------------------------------------------------
def embeddings_available() -> bool:
    """True when torch, open_clip and PIL can all be imported."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "open_clip", "PIL"))


def _require_embed_deps() -> None:
    if not embeddings_available():
        raise ImportError(
            "image embeddings need torch, open_clip_torch and pillow, which are optional. "
            f"Install them with: {INSTALL_HINT}"
        )


def pick_device() -> str:
    """'cuda' if available, else 'mps' if available, else 'cpu'."""
    _require_embed_deps()
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label(device: str) -> str:
    """Human-readable label of where embeddings were computed (GPU name when on cuda)."""
    if device == "cuda":
        import torch

        return torch.cuda.get_device_name(0)
    return device


# ---- backbones --------------------------------------------------------------------------
def load_backbone(name: str, device: str) -> tuple[Callable, Callable]:
    """Return (model_fn, preprocess). model_fn(batch_tensor) -> feature tensor."""
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    _require_embed_deps()
    kind, model_id, _dim = BACKBONES[name]
    import torch

    if kind == "open_clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(model_id)
        model = model.to(device).eval()
        return model.encode_image, preprocess

    if kind in ("torchhub", "transformers"):
        import torchvision.transforms as T

        preprocess = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
        if kind == "torchhub":
            repo, entry = model_id.split(":", 1)
            model = torch.hub.load(repo, entry).to(device).eval()
            return model, preprocess

        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_id).to(device).eval()

        def cls_token(x):
            # DINOv3 exposes the layer-normed CLS token as pooler_output; same 224 ImageNet
            # preprocessing as the DINOv2 hub models, so the two stay comparable.
            return model(pixel_values=x).pooler_output

        return cls_token, preprocess

    raise ValueError(f"unknown loader kind {kind!r} for backbone {name!r}")


def _torch_batch_fn(model_fn: Callable, device: str) -> Callable[[list], np.ndarray]:
    """Wrap a tensor model_fn so it takes a list of preprocessed tensors and returns numpy."""
    import torch

    def run(items: list) -> np.ndarray:
        with torch.no_grad():
            x = torch.stack(items).to(device)
            return model_fn(x).float().cpu().numpy()

    return run


def _numpy_batch_fn(model_fn: Callable) -> Callable[[list], np.ndarray]:
    """Wrap a numpy model_fn (used by tests and any torch-free loader)."""

    def run(items: list) -> np.ndarray:
        return np.asarray(model_fn(np.stack(items)), dtype=np.float32)

    return run


def _open_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _passthrough_path(path: str) -> str:
    """Stand-in for _open_image when a test loader's preprocess reads the path itself."""
    return path


# ---- staging ----------------------------------------------------------------------------
def photo_path(cache_dir: Path, obs_id: int) -> Path:
    return Path(cache_dir) / "photos" / f"{int(obs_id)}.jpg"


def _fetch_one(
    obs_id: int,
    url: str,
    dest: Path,
    *,
    timeout: float,
    retries: int,
    sleep: Callable[[float], None],
) -> tuple[int, str | None]:
    """Download url to dest with exponential backoff. Returns (id, error or None)."""
    if dest.exists():
        return obs_id, None
    if not isinstance(url, str) or not url:
        return obs_id, "no photo_url"
    err = "no attempts"
    for attempt in range(max(1, retries)):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            if not r.content:
                raise ValueError("empty body")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return obs_id, None
        except Exception as exc:  # network and disk errors are both retryable here
            err = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < retries:
                sleep(0.5 * (2**attempt))
    return obs_id, err


def stage_images(
    pool_df: pd.DataFrame,
    cache_dir: Path,
    *,
    workers: int = 4,
    timeout: float = 30,
    retries: int = 3,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Download photos to <cache_dir>/photos/<id>.jpg, skipping files that already exist.

    Returns a copy of pool_df with a ``local_path`` column. Single failures never raise;
    they are collected as (id, error) tuples in ``result.attrs["failed"]``.
    """
    cache_dir = Path(cache_dir)
    (cache_dir / "photos").mkdir(parents=True, exist_ok=True)
    out = pool_df.copy()
    out["local_path"] = [str(photo_path(cache_dir, i)) for i in out["id"]]
    jobs = [
        (int(i), u, Path(p))
        for i, u, p in zip(out["id"], out["photo_url"], out["local_path"], strict=True)
    ]
    t0 = time.time()
    failed: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [
            ex.submit(_fetch_one, i, u, p, timeout=timeout, retries=retries, sleep=time.sleep)
            for i, u, p in jobs
        ]
        for k, fut in enumerate(futures, 1):
            obs_id, err = fut.result()
            if err is not None:
                failed.append((obs_id, err))
            if k % 200 == 0:
                log(f"  [stage] {k}/{len(jobs)} ({len(failed)} failed)")
    log(
        f"[stage] {len(jobs) - len(failed)}/{len(jobs)} photos in {cache_dir / 'photos'} "
        f"({len(failed)} failed, {time.time() - t0:.0f}s)"
    )
    out.attrs["failed"] = failed
    return out


# ---- embedding --------------------------------------------------------------------------
def _l2_normalise(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float32)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return E / norms


def _embed_paths(
    paths: Sequence[str],
    batch_fn: Callable[[list], np.ndarray],
    preprocess: Callable,
    batch: int = 32,
    *,
    open_image: Callable[[str], object] = _open_image,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, list[int], list[tuple[int, str]]]:
    """Embed images at ``paths``. Returns (E rows L2-normalised, kept row indices, failures).

    ``batch_fn`` takes a list of preprocessed items and returns a (k, d) numpy array.
    Images that fail to open or preprocess are dropped and reported as (index, error).
    """
    embs: list[np.ndarray] = []
    kept: list[int] = []
    failed: list[tuple[int, str]] = []
    buf: list = []
    idxs: list[int] = []

    def flush() -> None:
        if buf:
            embs.append(np.asarray(batch_fn(buf), dtype=np.float32))
            kept.extend(idxs)
            buf.clear()
            idxs.clear()

    for i, p in enumerate(paths):
        try:
            buf.append(preprocess(open_image(p)))
            idxs.append(i)
        except Exception as exc:
            failed.append((i, f"{type(exc).__name__}: {exc}"))
            continue
        if len(buf) >= batch:
            flush()
        if (i + 1) % 500 == 0:
            log(f"  [embed] {i + 1}/{len(paths)}")
    flush()
    E = np.concatenate(embs, 0) if embs else np.zeros((0, 0), dtype=np.float32)
    return _l2_normalise(E), kept, failed


def emb_cache_path(cache_dir: Path, group: str, backbone: str) -> Path:
    safe = str(group).replace(" ", "_").replace("/", "_")
    return Path(cache_dir) / f"emb_{safe}_{backbone}.npz"


def save_embeddings(
    path: Path,
    *,
    ids: np.ndarray,
    E: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    backbone: str,
    emb_device: str,
    failed_ids: Sequence[int],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=np.int64),
        E=np.asarray(E, dtype=np.float32),
        lat=np.asarray(lat, dtype=float),
        lon=np.asarray(lon, dtype=float),
        backbone=backbone,
        emb_device=emb_device,
    )
    model_id = BACKBONES[backbone][1] if backbone in BACKBONES else backbone
    sidecar = {
        "n": int(len(ids)),
        "backbone": backbone,
        "model_id": model_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "failed_ids": [int(i) for i in failed_ids],
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1))


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids int64, E float32) from an npz written by save_embeddings."""
    from what_to_id.arms import load_embeddings as load_validated_embeddings

    ids, E, _ = load_validated_embeddings(path)
    return ids, E


def embed_group(
    pool_df_group: pd.DataFrame,
    *,
    backbone: str = DEFAULT_BACKBONE,
    cache_dir: Path,
    batch: int = 32,
    device: str | None = None,
    log: Callable[[str], None] = print,
    _loader: Callable[[str, str], tuple[Callable, Callable]] | None = None,
) -> Path:
    """Stage and embed one group; write ``emb_<group>_<backbone>.npz`` plus a json sidecar.

    Skips work when the npz already covers every id in the group and appends only the
    missing ids otherwise. ``_loader`` swaps in a numpy (model_fn, preprocess) pair for tests.
    """
    if backbone not in BACKBONES:
        raise KeyError(f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")
    cache_dir = Path(cache_dir)
    group = str(pool_df_group["iconic_taxon"].iloc[0]) if len(pool_df_group) else "empty"
    out = emb_cache_path(cache_dir, group, backbone)
    ids_all = pool_df_group["id"].to_numpy(dtype=np.int64)

    old_ids = np.zeros(0, dtype=np.int64)
    old_E = None
    if out.exists():
        old_ids, old_E = load_embeddings(out)
        missing = ~np.isin(ids_all, old_ids)
        if not missing.any():
            log(f"[embed] {group}: cache covers all {len(ids_all)} ids, skipping ({out})")
            return out
        todo = pool_df_group.loc[missing]
        log(f"[embed] {group}: appending {len(todo)} missing of {len(ids_all)} to {out}")
    else:
        todo = pool_df_group

    staged = stage_images(todo, cache_dir, log=log)
    stage_failed = {int(i) for i, _ in staged.attrs.get("failed", [])}

    if _loader is not None:
        model_fn, preprocess = _loader(backbone, device or "cpu")
        batch_fn = _numpy_batch_fn(model_fn)
        open_image: Callable[[str], object] = _passthrough_path
        emb_device = device or "fake"
    else:
        device = device or pick_device()
        model_fn, preprocess = load_backbone(backbone, device)
        batch_fn = _torch_batch_fn(model_fn, device)
        open_image = _open_image
        emb_device = device_label(device)

    t0 = time.time()
    E_new, kept, failed_rows = _embed_paths(
        list(staged["local_path"]), batch_fn, preprocess, batch, open_image=open_image, log=log
    )
    new_ids = staged["id"].to_numpy(dtype=np.int64)
    failed_ids = sorted(stage_failed | {int(new_ids[i]) for i, _ in failed_rows})
    kept_ids = new_ids[kept]
    kept_lat = staged["lat"].to_numpy(dtype=float)[kept]
    kept_lon = staged["lon"].to_numpy(dtype=float)[kept]

    if old_E is not None and len(old_ids):
        z = np.load(out)
        ids_out = np.concatenate([old_ids, kept_ids])
        E_out = np.concatenate([old_E, E_new]) if len(kept_ids) else old_E
        lat_out = np.concatenate([z["lat"], kept_lat])
        lon_out = np.concatenate([z["lon"], kept_lon])
        # keep earlier failures unless this pass embedded them after all
        failed_ids = sorted((set(_read_failed_ids(out)) | set(failed_ids)) - set(ids_out.tolist()))
    else:
        ids_out, E_out, lat_out, lon_out = kept_ids, E_new, kept_lat, kept_lon

    save_embeddings(
        out,
        ids=ids_out,
        E=E_out,
        lat=lat_out,
        lon=lon_out,
        backbone=backbone,
        emb_device=emb_device,
        failed_ids=failed_ids,
    )
    log(
        f"[embed] {group}: {len(kept_ids)} new rows with {backbone} "
        f"({len(failed_ids)} failed) in {time.time() - t0:.0f}s -> {out}"
    )
    return out


def _read_failed_ids(npz_path: Path) -> list[int]:
    sidecar = Path(npz_path).with_suffix(".json")
    if not sidecar.exists():
        return []
    try:
        return [int(i) for i in json.loads(sidecar.read_text()).get("failed_ids", [])]
    except (ValueError, AttributeError):
        return []


# ---- separability -----------------------------------------------------------------------
def separability_report(E: np.ndarray, labels: Sequence[str | None]) -> dict:
    """kNN leave-one-out separability (%) via labelfirst; ``n`` counts labelled rows."""
    import labelfirst

    labels = [None if pd.isna(v) or v == "" else v for v in labels]
    if len(labels) != len(E):
        raise ValueError(f"labels ({len(labels)}) must align with E rows ({len(E)})")
    pct = float(labelfirst.separability_score(np.asarray(E, dtype=np.float32), labels))
    return {"separability_pct": pct, "n": int(sum(v is not None for v in labels))}


# ---- CLI --------------------------------------------------------------------------------
def _cli(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m what_to_id.embed",
        description="Embed photos of the records assigned to one arm, one npz per group.",
    )
    ap.add_argument("--pool", required=True, help="pool parquet (id, iconic_taxon, photo_url, ...)")
    ap.add_argument(
        "--assign", default=None, help="assignment parquet (id, arm); omit to embed the whole pool"
    )
    ap.add_argument("--arm", default="similarity", help="arm to embed when --assign is given")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    ap.add_argument("--cache", default="data/", help="cache dir for photos and npz files")
    ap.add_argument("--groups", default=None, help="comma-separated iconic_taxon subset")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    pool = pd.read_parquet(args.pool)
    sel = pool
    scope = "pool"
    if args.assign:
        assign = pd.read_parquet(args.assign)
        if "arm" not in assign.columns or "id" not in assign.columns:
            raise SystemExit(f"--assign needs columns id and arm, got {list(assign.columns)}")
        ids = assign.loc[assign["arm"] == args.arm, "id"].astype("int64")
        sel = pool[pool["id"].astype("int64").isin(ids)]
        scope = f"arm={args.arm}"
    if args.groups:
        wanted = [g.strip() for g in args.groups.split(",") if g.strip()]
        sel = sel[sel["iconic_taxon"].isin(wanted)]
    print(f"[embed] {scope} records={len(sel)} groups={sel['iconic_taxon'].nunique()}")

    t_all = time.time()
    for group, df in sel.groupby("iconic_taxon", sort=True):
        t0 = time.time()
        out = embed_group(
            df,
            backbone=args.backbone,
            cache_dir=Path(args.cache),
            batch=args.batch,
            device=args.device,
        )
        n_ids, _ = load_embeddings(out)
        print(f"  {group}: {len(df)} assigned, {len(n_ids)} embedded, {time.time() - t0:.0f}s")
    print(f"[embed] done in {time.time() - t_all:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
