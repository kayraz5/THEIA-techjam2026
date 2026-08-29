"""Data ablation on cached frozen features: retrain the head on a SUBSET of the training sources.

Uses the identical head-training procedure as src/train.py::train_frozen (same optimiser, schedule,
loss, sampler, best-robust checkpointing), so the only variable is which training sources are kept.
Costs seconds — no feature extraction, everything comes from data/features/.

  python scripts/data_ablation.py --config configs/frozen_siglip2_giant_harmonized.yaml --keep sid_set
  python scripts/data_ablation.py --config ... --keep sid_set --keep wildfake   # (i.e. the full mix)
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import load_config, set_seed, get_device, metrics
from src.degradation import EVAL_CONDITIONS
from src.features import cache_path
from src.models import LinearHead
from src.train import make_loss, cosine_warmup, sampler_for

BB, SZ = "google/siglip2-giant-opt-patch16-384", 384


def load(cfg, tag):
    p = cache_path(cfg["features"]["cache_dir"], BB, SZ, tag)
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing cache for tag '{tag}' ({p}) — run src.train / src.evaluate first")
    z = np.load(p, allow_pickle=True)
    return z["feats"], z["labels"], z["paths"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--keep", action="append", required=True,
                    help="substring(s) of the sample path identifying sources to KEEP (e.g. sid_set, laion, dalle)")
    ap.add_argument("--tag", default=None, help="output name; defaults to the joined --keep values")
    ap.add_argument("--save_head", action="store_true",
                    help="persist the trained head so it can be scored on external sets (scripts/eval_external.py)")
    ap.add_argument("--subsample", type=int, default=0,
                    help="keep only N images per degradation draw (stratified by label) — for data-size curves")
    a = ap.parse_args()
    cfg = load_config(a.config); set_seed(cfg["seed"]); device = get_device()
    tc = cfg["train"]; name = a.tag or "_".join(a.keep)

    # ---- assemble the filtered training features from the cached degradation draws
    Xs, ys = [], []
    for k in range(tc.get("feature_epochs", 2)):
        X, y, paths = load(cfg, f"train_deg_e{k}")
        m = np.array([any(s in p for s in a.keep) for p in paths])
        Xs.append(X[m]); ys.append(y[m])
    if a.subsample:
        rng = np.random.default_rng(cfg["seed"])
        for k in range(len(Xs)):
            y = ys[k]; keep = []
            for lab in (0, 1):                      # stratified: preserve the label ratio
                idx = np.flatnonzero(y == lab)
                keep.append(rng.choice(idx, min(len(idx), round(a.subsample * len(idx) / len(y))), replace=False))
            sel = np.sort(np.concatenate(keep)); Xs[k], ys[k] = Xs[k][sel], ys[k][sel]
    Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
    print(f"[ablation] keep={a.keep}: {len(Xtr)} train features ({int((ytr==0).sum())} real / {int(ytr.sum())} fake) "
          f"from {tc.get('feature_epochs',2)} degradation draws")
    if len(Xtr) == 0:
        raise SystemExit("no training samples matched --keep")

    # ---- head training: identical procedure to src/train.py::train_frozen
    head = LinearHead(Xtr.shape[1]).to(device)
    Xt = torch.from_numpy(Xtr).to(device); yt = torch.from_numpy(ytr).to(device)
    spe = math.ceil(len(Xt) / tc["batch_size"]); total = spe * tc["epochs"]
    opt = torch.optim.AdamW(head.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = cosine_warmup(opt, total, spe * tc["warmup_epochs"]); loss_fn = make_loss(tc)
    sampler = sampler_for(ytr, tc["weighted_sampler"])
    def score(X):
        with torch.no_grad():
            return torch.softmax(head(torch.from_numpy(X).to(device)).float(), -1)[:, 1].cpu().numpy()
    Xvd, yvd, _ = load(cfg, "valsub_deg")          # robust checkpoint metric, same as the main arm
    best, best_sd = -1, None
    for ep in range(tc["epochs"]):
        head.train()
        idx = torch.as_tensor(list(sampler)) if sampler else torch.randperm(len(Xt))
        for b in range(spe):
            bi = idx[b * tc["batch_size"]:(b + 1) * tc["batch_size"]].to(device)
            loss = loss_fn(head(Xt[bi]), yt[bi]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        head.eval()
        r = metrics(yvd, score(Xvd))["auc"]
        if r > best: best, best_sd = r, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_sd); head.eval()
    print(f"[ablation] best robust (degraded val subset) AUC {best:.4f}")

    # ---- score every eval condition from cache
    rows = []
    for cname, lv in EVAL_CONDITIONS:
        t = "clean" if cname == "clean" else f"{cname}_{lv}"
        Xv, yv, _ = load(cfg, f"valsub_{t}"); Xa, _, _ = load(cfg, f"altsub_{t}")
        sv, sa = score(Xv), score(Xa); fake = sv[yv == 1]; coco = sv[yv == 0]
        m_coco = metrics(np.r_[np.zeros(len(coco)), np.ones(len(fake))], np.r_[coco, fake])
        m_alt = metrics(np.r_[np.zeros(len(sa)), np.ones(len(fake))], np.r_[sa, fake])
        rows.append({"transform": cname, "level": "" if lv is None else lv,
                     "auc_coco_real": m_coco["auc"], "auc_alt_real": m_alt["auc"],
                     "err_coco_real": m_coco["err"], "shortcut_gap": m_coco["auc"] - m_alt["auc"]})
    df = pd.DataFrame(rows); clean = df.iloc[0]
    df["delta_coco"] = clean.auc_coco_real - df.auc_coco_real

    # full-set clean AUC (the headline number), from the full val_clean cache
    Xf, yf, _ = load(cfg, "val_clean"); mf = metrics(yf, score(Xf))

    out_dir = os.path.join(cfg["output_dir"], "eval"); os.makedirs(out_dir, exist_ok=True)
    if a.save_head:
        ckpt = os.path.join(cfg["output_dir"], f"head_ablation_{name}.pt")
        torch.save({"head": head.state_dict(), "cfg": cfg, "ablation": name, "keep": a.keep}, ckpt)
        print(f"[ablation] saved head -> {ckpt}")
    csv = os.path.join(out_dir, f"auc_grid_ablation_{name}.csv"); df.to_csv(csv, index=False)
    deg = df.iloc[1:]; worst = deg.loc[deg.auc_coco_real.idxmin()]
    summary = {"ablation": name, "keep": a.keep, "n_train_features": int(len(Xtr)),
               "clean_auc_full_set": mf["auc"], "n_full_set": mf["n"],
               "clean_auc_subset": float(clean.auc_coco_real), "clean_auc_alt_real": float(clean.auc_alt_real),
               "shortcut_gap_clean": float(clean.shortcut_gap),
               "worst_transform": f"{worst['transform']}@{worst['level']}", "worst_auc": float(worst.auc_coco_real),
               "max_delta": float(deg.delta_coco.max()), "mean_degraded_auc": float(deg.auc_coco_real.mean()),
               "best_robust_val_auc": float(best)}
    json.dump(summary, open(os.path.join(out_dir, f"summary_ablation_{name}.json"), "w"), indent=1)
    print(df[["transform", "level", "auc_coco_real", "auc_alt_real", "delta_coco"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nclean AUC FULL val set : {mf['auc']:.4f} (n={mf['n']})")
    print(f"worst single transform : {summary['worst_transform']} AUC {summary['worst_auc']:.4f}   max delta {summary['max_delta']:.4f}")
    print(f"-> {csv}")


if __name__ == "__main__":
    main()
