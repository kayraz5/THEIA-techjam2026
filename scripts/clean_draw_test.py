"""Issue #8: does adding a CLEAN training draw help, given distortion_prob=1.0 means the head
currently never sees an undamaged image?

Extracts two new caches over the training set:
  train_clean   — one undegraded pass
  train_deg_e2  — a third degraded draw, as the fair control ("more data" vs "clean data")

then trains three heads on cached features and scores them on the standard eval grid:
  A: 2 degraded draws                (current baseline)
  B: 2 degraded + 1 clean            (the proposal)
  C: 3 degraded draws                (control: same data volume as B)

  python scripts/clean_draw_test.py --config configs/frozen_siglip2_giant_sidonly.yaml
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import load_config, set_seed, get_device, build_backbone, metrics
from src.data import build_train, summarize
from src.degradation import RandomDegradation
from src.features import get_or_extract, cache_path
from src.head_utils import train_head_on, eval_grid   # thin shared helpers

BB, SZ = "google/siglip2-giant-opt-patch16-384", 384


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = load_config(a.config); set_seed(cfg["seed"]); device = get_device()
    dc, tc = cfg["degradation"], cfg["train"]
    need = [t for t in ("train_clean", "train_deg_e2")
            if not os.path.exists(cache_path(cfg["features"]["cache_dir"], BB, SZ, t))]
    if need:
        backbone = build_backbone(cfg, device)
        for p in backbone.parameters(): p.requires_grad_(False)
        train_s = build_train(cfg); summarize(train_s, "train")
        aug = RandomDegradation(dc["prob"], dc["min_transforms"], dc["max_transforms"], dc["hflip"], dc["seed"])
        if "train_clean" in need:
            get_or_extract(cfg, backbone, train_s, "train_clean", device)          # no degradation
        if "train_deg_e2" in need:
            get_or_extract(cfg, backbone, train_s, "train_deg_e2", device, degrade=aug, seed=cfg["seed"] * 100 + 2)
        del backbone
        if device.type == "mps": torch.mps.empty_cache()

    def load(tag):
        z = np.load(cache_path(cfg["features"]["cache_dir"], BB, SZ, tag), allow_pickle=True)
        return z["feats"], z["labels"]
    e0, e1, e2, cl = load("train_deg_e0"), load("train_deg_e1"), load("train_deg_e2"), load("train_clean")
    arms = {
        "A: 2 degraded (baseline)":      [e0, e1],
        "B: 2 degraded + 1 clean":       [e0, e1, cl],
        "C: 3 degraded (volume control)":[e0, e1, e2],
    }
    rows = []
    for name, parts in arms.items():
        X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
        head, rob, score = train_head_on(cfg, X, y, device)
        m = eval_grid(cfg, score)
        rows.append({"arm": name, "n_train": len(X), "robust_ckpt": rob, **m})
        print(f"  {name:32s} n={len(X):6d} clean {m['clean_auc_full']:.4f}  worst {m['worst_auc']:.4f} "
              f"({m['worst_cell']})  maxdelta {m['max_delta']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(cfg["output_dir"], "clean_draw_test.csv"); df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}")); print(f"\n-> {out}")


if __name__ == "__main__":
    main()
