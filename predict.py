#!/usr/bin/env python
"""python predict.py --image_dir <dir> --output <path.json> [--config configs/frozen_siglip2_giant.yaml] [--checkpoint ...]

Emits [{"image_path": ..., "pred": p_fake}, ...]. Recurses into subdirectories; jpg/jpeg/png/webp/bmp;
corrupt files are skipped with a warning; batched inference with a progress bar and img/s throughput."""
import argparse, json, os, sys, time, warnings
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common import load_config, get_device, build_backbone, squish_resize, to_tensor
from src.data import IMG_EXT
from src.evaluate import load_scorer

class DirDS(Dataset):
    def __init__(self, paths, size, mean, std): self.p, self.size, self.mean, self.std = paths, size, mean, std
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        try:
            img = Image.open(self.p[i]); img.load()
            return to_tensor(squish_resize(img, self.size), self.mean, self.std), i, True
        except Exception as e:
            warnings.warn(f"skipping corrupt/unreadable {self.p[i]}: {e}")
            return torch.zeros(3, self.size, self.size), i, False

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--image_dir", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="configs/frozen_siglip2_giant.yaml"); ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch_size", type=int, default=32); ap.add_argument("--workers", type=int, default=2); a = ap.parse_args()
    cfg = load_config(a.config); device = get_device()
    backbone = build_backbone(cfg, device)
    for p in backbone.parameters(): p.requires_grad_(False)
    ckpt = a.checkpoint or os.path.join(cfg["output_dir"], "head_best.pt" if cfg["mode"] == "frozen" else "lora_best.pt")
    score = load_scorer(cfg, backbone, device, ckpt)
    paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(a.image_dir) for f in fs if os.path.splitext(f)[1].lower() in IMG_EXT)
    print(f"[predict] {len(paths)} images under {a.image_dir}")
    i = backbone.info; dl = DataLoader(DirDS(paths, i.image_size, i.mean, i.std), batch_size=a.batch_size, num_workers=a.workers)
    dtype = next(backbone.parameters()).dtype; out = {}; t0 = time.time(); n_ok = 0
    with torch.no_grad():
        for x, idx, ok in tqdm(dl, unit="batch"):
            feats = backbone(x.to(device, dtype)).float().cpu().numpy(); p = score(feats)
            for j, k, o in zip(idx.tolist(), p.tolist(), ok.tolist()):
                if o: out[j] = float(min(max(k, 0.0), 1.0)); n_ok += 1
    dt = time.time() - t0
    res = [{"image_path": paths[j], "pred": out[j]} for j in sorted(out)]
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True); json.dump(res, open(a.output, "w"), indent=1)
    print(f"[predict] wrote {len(res)} predictions ({len(paths)-n_ok} skipped) to {a.output} | {n_ok/dt:.1f} img/s")

if __name__ == "__main__":
    main()
