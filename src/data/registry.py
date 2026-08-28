"""Dataset registry: every source yields a list of Sample(path_or_bytes, label, source, group).

label: 0 = real, 1 = AI-generated.  Sources:
  sid_set          HF saberzl/SID_Set parquet shards (real = OpenImages, fake = label 1 full synthetic)
  wildfake:<name>  a WildFake label CSV, resolved against data/wildfake/images (only downloaded files)
  cifake           32x32 Kaggle set, smoke test only (config flag, default off)
Validation:
  wildfake_coco_dalle3   4,998 COCO val2017 reals + 8,843 DALL-E Advanced (dalle3) fakes  [organisers' subset]
  alt real set           held-out non-COCO reals (default: WildFake imagenet slice) for the shortcut check
"""
from __future__ import annotations

import glob
import io
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image

from src.degradation import jpeg_compress

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
HARMONIZE_JPEG_QUALITY = 90   # train-only re-encode target; kills the file-format-vs-label shortcut (see registry.py:build_train)


@dataclass
class Sample:
    path: str            # file path, or "<parquet>#<row>" for parquet-backed samples
    label: int
    source: str
    meta: dict = field(default_factory=dict)
    _bytes: bytes | None = field(default=None, repr=False)
    harmonize: bool = False   # re-encode through one common JPEG pass on load (train sources only)

    def load(self) -> Image.Image:
        if self._bytes is not None:
            img = Image.open(io.BytesIO(self._bytes)).convert("RGB")
        else:
            img = Image.open(self.path).convert("RGB")
        return jpeg_compress(img, HARMONIZE_JPEG_QUALITY) if self.harmonize else img


# ---------------------------------------------------------------- WildFake -----------------------
def wildfake_csv(root: str, csv_name: str, max_n: int = 0, seed: int = 0, path_filter: str | None = None,
                 require_exists: bool = True) -> list[Sample]:
    df = pd.read_csv(os.path.join(root, "csv", f"{csv_name}.csv"))
    if path_filter:
        df = df[df.Image_path.str.contains(path_filter, regex=False)]
    samples = []
    for r in df.itertuples(index=False):
        p = os.path.join(root, "images", r.Image_path.lstrip("./"))
        if require_exists and not os.path.exists(p):
            continue
        samples.append(Sample(p, int(r.IsFake), f"wildfake:{csv_name}", {"advanced": int(r.IsAdvanced)}))
    if max_n and len(samples) > max_n:
        random.Random(seed).shuffle(samples)
        samples = sorted(samples[:max_n], key=lambda s: s.path)
    return samples


def wildfake_validation(root: str) -> list[Sample]:
    """Organisers' designated demo subset."""
    real = wildfake_csv(root, "real_coco", path_filter="/val2017/")
    fake = wildfake_csv(root, "dalle3", path_filter="/Advanced/")
    for s in real + fake:
        s.source = "val:" + s.source
    return real + fake


def wildfake_alt_real(root: str, csv_name: str = "real_imagenet", n: int = 2000, seed: int = 0) -> list[Sample]:
    s = wildfake_csv(root, csv_name, max_n=n, seed=seed)
    for x in s:
        x.source = "altreal:" + x.source
    return s


# ---------------------------------------------------------------- SID_Set ------------------------
def sid_set(root: str, max_n: int = 0, include_tampered: bool = False, seed: int = 0) -> list[Sample]:
    """Loads locally downloaded parquet shards (data/sid_set/data/train-*.parquet). Bytes kept in memory
    are avoided: we store shard path + row and read lazily via pyarrow row groups."""
    import pyarrow.parquet as pq
    shards = sorted(glob.glob(os.path.join(root, "data", "train-*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no SID_Set shards under {root}/data")
    samples: list[Sample] = []
    for sh in shards:
        t = pq.read_table(sh, columns=["img_id", "label"])
        labels = t.column("label").to_pylist()
        ids = t.column("img_id").to_pylist()
        for i, (iid, lab) in enumerate(zip(ids, labels)):
            if lab == 2 and not include_tampered:
                continue
            samples.append(Sample(f"{sh}#{i}", 1 if lab in (1, 2) else 0, "sid_set", {"img_id": iid, "raw_label": lab}))
    if max_n and len(samples) > max_n:
        rng = random.Random(seed)
        # stratified subsample keeps the real/fake ratio of the shards
        real = [s for s in samples if s.label == 0]; fake = [s for s in samples if s.label == 1]
        k_real = round(max_n * len(real) / len(samples))
        rng.shuffle(real); rng.shuffle(fake)
        samples = real[:k_real] + fake[: max_n - k_real]
    return samples


def materialise_parquet_samples(samples: list[Sample], out_dir: str) -> list[Sample]:
    """Write parquet-backed samples to disk once (so hashing/exclusion and dataloaders see plain files)."""
    import pyarrow.parquet as pq
    from collections import defaultdict
    os.makedirs(out_dir, exist_ok=True)
    by_shard = defaultdict(list)
    for s in samples:
        if "#" in s.path and s.path.endswith(tuple("0123456789")):
            sh, row = s.path.rsplit("#", 1); by_shard[sh].append((int(row), s))
    for sh, rows in by_shard.items():
        need = [(r, s) for r, s in rows if not os.path.exists(os.path.join(out_dir, f"{s.meta['img_id']}.png"))]
        if not need:
            for r, s in rows: s.path = os.path.join(out_dir, f"{s.meta['img_id']}.png")
            continue
        pf = pq.ParquetFile(sh)
        idx = {r: s for r, s in rows}
        offset = 0
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["image"])
            n = tbl.num_rows
            col = tbl.column("image")
            for r in range(offset, offset + n):
                if r in idx:
                    d = col[r - offset].as_py()
                    p = os.path.join(out_dir, f"{idx[r].meta['img_id']}.png")
                    if not os.path.exists(p):
                        Image.open(io.BytesIO(d["bytes"])).convert("RGB").save(p, format="PNG", compress_level=1)
                    idx[r].path = p
            offset += n
    return samples


# ---------------------------------------------------------------- CIFAKE -------------------------
def cifake(root: str, max_n: int = 0, seed: int = 0) -> list[Sample]:
    """32x32 images. Smoke test ONLY (see README §limitations)."""
    samples = []
    for split in ("train", "test"):
        for lab, sub in ((0, "REAL"), (1, "FAKE")):
            for p in glob.glob(os.path.join(root, split, sub, "*")):
                samples.append(Sample(p, lab, "cifake"))
    if max_n and len(samples) > max_n:
        random.Random(seed).shuffle(samples); samples = samples[:max_n]
    return samples


# ---------------------------------------------------------------- assembly -----------------------
def build_train(cfg: dict) -> list[Sample]:
    d = cfg["data"]
    out: list[Sample] = []
    if "sid_set" in d["sources"]:
        s = sid_set("data/sid_set", d["sid_set"].get("max_train", 0), d["sid_set"].get("include_tampered", False), cfg["seed"])
        out += materialise_parquet_samples(s, "data/sid_set/images")
    if "wildfake" in d["sources"]:
        w = d["wildfake"]
        for name in w["train_subsets"]:
            # never touch the validation sources: COCO val2017 and dalle3/Advanced are excluded structurally here
            # AND by hash in the exclusion check.
            flt = None
            if name == "real_coco": flt = "/test2017/"
            if name == "dalle3": raise ValueError("dalle3 (DALL-E Advanced) is the held-out validation fake set; refusing to train on it")
            out += wildfake_csv(w["root"], name, w.get("max_per_subset", 0), cfg["seed"], path_filter=flt)
    if d.get("cifake", {}).get("enabled", False):
        print("[data] WARNING: CIFAKE (32x32) enabled — smoke test only, numbers are not meaningful at 384px")
        out += cifake(d["cifake"]["root"], d["cifake"].get("max_n", 0), cfg["seed"])
    # Sources differ in native format by construction (WildFake dalle2 fakes are 512px PNG,
    # real_laion5b reals are JPEG) — a linear head can learn "PNG -> fake" instead of anything
    # about AI generation. Harmonize every training image through one common JPEG pass so file
    # format stops correlating with label. Validation/alt-real stay untouched (organisers' format).
    for s in out:
        s.harmonize = True
    return out


def summarize(samples: list[Sample], title: str) -> None:
    from collections import Counter
    c = Counter((s.source, s.label) for s in samples)
    n = len(samples); nf = sum(s.label for s in samples)
    print(f"[data] {title}: {n} samples, real={n-nf} fake={nf} (fake frac {nf/max(n,1):.3f})")
    for (src, lab), k in sorted(c.items()):
        print(f"         {src:32s} label={lab} n={k}")


def eval_subset(val_s: list[Sample], cfg: dict) -> list[Sample]:
    """Seeded fixed subset of the validation set used for the per-condition grid (compute budget)."""
    n = cfg["eval"].get("n_eval_max", 0)
    if not n or len(val_s) <= n:
        return val_s
    v = list(val_s); random.Random(cfg["seed"]).shuffle(v)
    return sorted(v[:n], key=lambda s: s.path)
