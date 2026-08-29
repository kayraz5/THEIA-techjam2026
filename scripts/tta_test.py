"""Issue #11: test-time augmentation via horizontal-flip view averaging.

Flipping the normalised 384x384 tensor horizontally is exactly equivalent to flipping the image
before the squish resize (squish is separable per axis; normalisation is per channel), so the flipped
view needs no separate image pipeline — only a second backbone pass.

Validates on the cheap subset the issue recommends (clean + the two historically weakest cells)
before anyone pays for a full grid re-run.

  python scripts/tta_test.py --config configs/frozen_siglip2_giant_sidonly.yaml
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import load_config, set_seed, get_device, build_backbone, metrics
from src.data import wildfake_validation, wildfake_alt_real, eval_subset
from src.features import ImageDS, cache_path
from src.head_utils import load_cached, train_head_on

BB, SZ = "google/siglip2-giant-opt-patch16-384", 384
CONDS = [("clean", None), ("jpeg", 30), ("resize", 0.25)]


@torch.no_grad()
def extract_flipped(backbone, samples, cfg, cond, seed, device):
    """Features for the horizontally-flipped view of each image under condition `cond`."""
    i = backbone.info
    ds = ImageDS(samples, i.image_size, i.mean, i.std, fixed=cond, seed=seed)
    dl = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], num_workers=2, shuffle=False)
    feats = np.zeros((len(ds), backbone.feature_dim), np.float32); labels = np.zeros(len(ds), np.int64)
    dtype = next(backbone.parameters()).dtype
    for x, y, idx, _ in dl:
        x = torch.flip(x, dims=[-1])                      # the TTA view
        feats[idx.numpy()] = backbone(x.to(device, dtype)).float().cpu().numpy()
        labels[idx.numpy()] = y.numpy()
    return feats, labels


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = load_config(a.config); set_seed(cfg["seed"]); device = get_device()
    root = cfg["data"]["wildfake"]["root"]
    val = eval_subset(wildfake_validation(root), cfg)
    alt = wildfake_alt_real(root, n=cfg["data"]["alt_real"]["n"], seed=cfg["seed"])
    seed = cfg["degradation"]["seed"]

    # extract flipped views (cached so re-runs are free)
    todo = [(n, lv) for n, lv in CONDS
            if not os.path.exists(cache_path(cfg["features"]["cache_dir"], BB, SZ,
                                             f"valsub_flip_{'clean' if n=='clean' else f'{n}_{lv}'}"))]
    if todo:
        backbone = build_backbone(cfg, device)
        for p in backbone.parameters(): p.requires_grad_(False)
        for n, lv in todo:
            tag = "clean" if n == "clean" else f"{n}_{lv}"
            t0 = time.time()
            for name, samples in (("valsub_flip", val), ("altsub_flip", alt)):
                f, y = extract_flipped(backbone, samples, cfg, (n, lv), seed, device)
                np.savez(cache_path(cfg["features"]["cache_dir"], BB, SZ, f"{name}_{tag}"),
                         feats=f, labels=y, plans=np.array([""] * len(f)),
                         paths=np.array([s.path for s in samples]))
            print(f"[tta] flipped views for {tag}: {(len(val)+len(alt))/(time.time()-t0):.1f} img/s", flush=True)
        del backbone
        if device.type == "mps": torch.mps.empty_cache()

    # train the arm's head, then compare single-view vs 2-view averaged scoring
    Xs, ys = [], []
    for k in range(cfg["train"].get("feature_epochs", 2)):
        X, y = load_cached(cfg, f"train_deg_e{k}"); Xs.append(X); ys.append(y)
    head, rob, score = train_head_on(cfg, np.concatenate(Xs), np.concatenate(ys), device)

    rows = []
    for n, lv in CONDS:
        tag = "clean" if n == "clean" else f"{n}_{lv}"
        Xv, yv = load_cached(cfg, f"valsub_{tag}"); Xa, _ = load_cached(cfg, f"altsub_{tag}")
        Xvf, _ = load_cached(cfg, f"valsub_flip_{tag}"); Xaf, _ = load_cached(cfg, f"altsub_flip_{tag}")
        for label, (sv, sa) in (("single view", (score(Xv), score(Xa))),
                                ("TTA (orig+hflip mean)", ((score(Xv) + score(Xvf)) / 2, (score(Xa) + score(Xaf)) / 2))):
            fake, coco = sv[yv == 1], sv[yv == 0]
            rows.append({"condition": tag, "scoring": label,
                         "auc_coco_real": metrics(np.r_[np.zeros(len(coco)), np.ones(len(fake))], np.r_[coco, fake])["auc"],
                         "auc_alt_real": metrics(np.r_[np.zeros(len(sa)), np.ones(len(fake))], np.r_[sa, fake])["auc"]})
    df = pd.DataFrame(rows)
    piv = df.pivot(index="condition", columns="scoring", values="auc_coco_real")
    piv["delta"] = piv["TTA (orig+hflip mean)"] - piv["single view"]
    out = os.path.join(cfg["output_dir"], "tta_test.csv"); df.to_csv(out, index=False)
    print("\n" + piv.to_string(float_format=lambda v: f"{v:.4f}")); print(f"\n-> {out}")


if __name__ == "__main__":
    main()
