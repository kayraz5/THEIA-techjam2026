"""Frozen-feature extraction with on-disk caching.

Cache key = (backbone name, image size, sample path, degradation plan). Features are stored per
(dataset, condition) as .npz so re-training the linear head is near-instant.

    python -m src.features --config configs/frozen_siglip2_giant.yaml --split train --epochs 3
    python -m src.features --config ... --split val --conditions all
"""
from __future__ import annotations
import argparse, hashlib, os, time
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from src.common import load_config, get_device, squish_resize, to_tensor, build_backbone, set_seed
from src.degradation import RandomDegradation, apply_transform, EVAL_CONDITIONS
from src.data import Sample


class ImageDS(Dataset):
    def __init__(self, samples, size, mean, std, degrade: RandomDegradation | None = None,
                 fixed: tuple | None = None, seed: int = 0):
        self.s, self.size, self.mean, self.std, self.degrade, self.fixed, self.seed = samples, size, mean, std, degrade, fixed, seed
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s = self.s[i]
        try:
            img = s.load()
        except Exception as e:
            img = Image.new("RGB", (self.size, self.size)); print(f"[warn] unreadable {s.path}: {e}")
        plan = []
        if self.degrade is not None:
            self.degrade.reseed(self.seed * 1_000_003 + i)   # deterministic per (epoch-seed, index)
            img, plan = self.degrade(img)
        elif self.fixed is not None and self.fixed[0] != "clean":
            img = apply_transform(img, self.fixed[0], self.fixed[1], rng_seed=self.seed * 1_000_003 + i)
        x = to_tensor(squish_resize(img, self.size), self.mean, self.std)
        return x, s.label, i, "|".join(f"{n}:{l}" for n, l in plan)


def cache_path(cache_dir, backbone_name, size, tag):
    key = hashlib.md5(f"{backbone_name}|{size}|{tag}".encode()).hexdigest()[:10]
    safe = backbone_name.replace("/", "_")
    return os.path.join(cache_dir, f"{safe}_{size}px_{tag}_{key}.npz")


@torch.no_grad()
def extract(backbone, samples, size, mean, std, device, batch_size=32, degrade=None, fixed=None, seed=0, workers=2, desc=""):
    # NOTE: >2 spawned workers on this 64 GB MPS box starves the GPU (memory pressure): 6 workers -> ~1 img/s, 2 -> ~6 img/s.
    ds = ImageDS(samples, size, mean, std, degrade, fixed, seed)
    workers = min(workers, max(0, len(ds) // 200))
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers, shuffle=False, persistent_workers=False, prefetch_factor=(4 if workers else None))
    feats = np.zeros((len(ds), backbone.feature_dim), np.float32); labels = np.zeros(len(ds), np.int64)
    plans = [""] * len(ds); t0 = time.time(); n = 0
    dtype = next(backbone.parameters()).dtype
    for x, y, idx, plan in tqdm(dl, desc=desc or "extract", leave=False):
        f = backbone(x.to(device, dtype)).float().cpu().numpy()
        feats[idx.numpy()] = f; labels[idx.numpy()] = y.numpy(); n += len(y)
        for j, p in zip(idx.tolist(), plan): plans[j] = p
    print(f"[features] {desc}: {n} imgs in {time.time()-t0:.0f}s ({n/max(time.time()-t0,1e-6):.1f} img/s)")
    return feats, labels, np.array(plans), np.array([s.path for s in samples])


def get_or_extract(cfg, backbone, samples, tag, device, degrade=None, fixed=None, seed=0):
    p = cache_path(cfg["features"]["cache_dir"], backbone.info.name, backbone.info.image_size, tag)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        if len(z["paths"]) == len(samples) and (z["paths"] == np.array([s.path for s in samples])).all():
            return z["feats"], z["labels"], z["plans"], z["paths"]
        print(f"[features] cache {p} stale (sample list changed) -> re-extracting")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    i = backbone.info
    feats, labels, plans, paths = extract(backbone, samples, i.image_size, i.mean, i.std, device,
                                          cfg["eval"]["batch_size"], degrade, fixed, seed, desc=tag)
    np.savez(p, feats=feats, labels=labels, plans=plans, paths=paths)
    return feats, labels, plans, paths
