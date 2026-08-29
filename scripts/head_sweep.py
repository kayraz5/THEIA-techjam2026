"""Issues #4/#5/#6: sweep linear-head training variants on cached frozen features.

Everything runs from data/features/ — no extraction, no GPU-heavy work, seconds per configuration.

  python scripts/head_sweep.py --config <cfg> --sweep reg     # weight_decay x epochs x lr   (#4)
  python scripts/head_sweep.py --config <cfg> --sweep norm    # none / l2 / standardize      (#5)
  python scripts/head_sweep.py --config <cfg> --sweep loss    # ce vs focal (+ gamma)        (#6)

Each row reports the headline clean AUC on the FULL designated validation set, plus the worst
per-transform cell and max degradation delta from the cached eval grid.
"""
from __future__ import annotations
import argparse, itertools, json, math, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import load_config, set_seed, get_device, metrics
from src.degradation import EVAL_CONDITIONS
from src.features import cache_path
from src.models import LinearHead
from src.train import FocalLoss, cosine_warmup, sampler_for
import torch.nn as nn

BB, SZ = "google/siglip2-giant-opt-patch16-384", 384


def load(cfg, tag):
    return (lambda z: (z["feats"], z["labels"]))(np.load(cache_path(cfg["features"]["cache_dir"], BB, SZ, tag), allow_pickle=True))


def normalizer(kind, Xtr):
    if kind == "none":  return lambda X: X
    if kind == "l2":    return lambda X: X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
    if kind == "standardize":
        mu, sd = Xtr.mean(0), np.clip(Xtr.std(0), 1e-6, None)   # train stats only — fitting on val would leak
        return lambda X: (X - mu) / sd
    raise KeyError(kind)


def train_head(Xtr, ytr, Xvd, yvd, device, epochs, lr, wd, loss_name, gamma, alpha, batch, warmup, sampler_on, seed):
    set_seed(seed)
    head = LinearHead(Xtr.shape[1]).to(device)
    Xt = torch.from_numpy(Xtr).to(device); yt = torch.from_numpy(ytr).to(device)
    spe = math.ceil(len(Xt) / batch); total = spe * epochs
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = cosine_warmup(opt, total, spe * min(warmup, max(epochs - 1, 0)) if epochs > 1 else 1)
    loss_fn = FocalLoss(gamma, alpha) if loss_name == "focal" else nn.CrossEntropyLoss()
    sampler = sampler_for(ytr, sampler_on)
    def score(X):
        with torch.no_grad():
            return torch.softmax(head(torch.from_numpy(X).to(device)).float(), -1)[:, 1].cpu().numpy()
    best, best_sd = -1, None
    for _ in range(epochs):
        head.train()
        idx = torch.as_tensor(list(sampler)) if sampler else torch.randperm(len(Xt))
        for b in range(spe):
            bi = idx[b * batch:(b + 1) * batch].to(device)
            loss = loss_fn(head(Xt[bi]), yt[bi]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        head.eval()
        r = metrics(yvd, score(Xvd))["auc"]
        if r > best: best, best_sd = r, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_sd)
    return head, best, score


def evaluate(cfg, score, norm):
    aucs = {}
    for cname, lv in EVAL_CONDITIONS:
        t = "clean" if cname == "clean" else f"{cname}_{lv}"
        Xv, yv = load(cfg, f"valsub_{t}")
        aucs[t] = metrics(yv, score(norm(Xv)))["auc"]
    Xf, yf = load(cfg, "val_clean")
    deg = {k: v for k, v in aucs.items() if k != "clean"}
    worst_k = min(deg, key=deg.get)
    return {"clean_auc_full": metrics(yf, score(norm(Xf)))["auc"], "clean_auc_subset": aucs["clean"],
            "worst_cell": worst_k, "worst_auc": deg[worst_k],
            "max_delta": max(aucs["clean"] - v for v in deg.values()), "mean_degraded": float(np.mean(list(deg.values())))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--sweep", choices=["reg", "norm", "loss"], required=True)
    ap.add_argument("--norm", default="none"); ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--wd", type=float, default=None); ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config); device = get_device(); tc = cfg["train"]
    Xs, ys = [], []
    for k in range(tc.get("feature_epochs", 2)):
        X, y = load(cfg, f"train_deg_e{k}"); Xs.append(X); ys.append(y)
    Xtr_raw, ytr = np.concatenate(Xs), np.concatenate(ys)
    Xvd_raw, yvd = load(cfg, "valsub_deg")
    print(f"[sweep] {len(Xtr_raw)} train features from {cfg['name']}; sweep={a.sweep}")

    base = dict(epochs=a.epochs or tc["epochs"], lr=a.lr or tc["lr"], wd=a.wd if a.wd is not None else tc["weight_decay"],
                loss_name=tc["loss"], gamma=tc["focal_gamma"], alpha=tc["focal_alpha"],
                batch=tc["batch_size"], warmup=tc["warmup_epochs"], sampler_on=tc["weighted_sampler"], seed=cfg["seed"])
    if a.sweep == "reg":
        grid = [dict(base, wd=wd, epochs=ep, lr=lr, _norm=a.norm)
                for wd, ep, lr in itertools.product([0.01, 0.1, 0.3, 1.0], [1, 2, 5, 10, 30], [1e-3, 1e-4])]
    elif a.sweep == "norm":
        grid = [dict(base, _norm=n) for n in ["none", "l2", "standardize"]]
    else:
        grid = [dict(base, loss_name="ce", _norm=a.norm)] + \
               [dict(base, loss_name="focal", gamma=g, _norm=a.norm) for g in [1.0, 2.0, 5.0, 10.0, 20.0]]

    rows = []
    for i, g in enumerate(grid):
        norm_kind = g.pop("_norm"); norm = normalizer(norm_kind, Xtr_raw)
        Xtr, Xvd = norm(Xtr_raw), norm(Xvd_raw)
        head, rob, score = train_head(Xtr, ytr, Xvd, yvd, device, **g)
        m = evaluate(cfg, score, norm)
        rows.append({"norm": norm_kind, "loss": g["loss_name"], "gamma": g["gamma"] if g["loss_name"] == "focal" else "",
                     "wd": g["wd"], "epochs": g["epochs"], "lr": g["lr"], "robust_ckpt": rob, **m})
        print(f"  [{i+1}/{len(grid)}] norm={norm_kind:11s} loss={g['loss_name']:5s} wd={g['wd']:<5} ep={g['epochs']:<3} "
              f"lr={g['lr']:g} -> clean {m['clean_auc_full']:.4f}  worst {m['worst_auc']:.4f} ({m['worst_cell']})  maxdelta {m['max_delta']:.4f}", flush=True)
    df = pd.DataFrame(rows).sort_values("clean_auc_full", ascending=False)
    out = os.path.join(cfg["output_dir"], f"sweep_{a.sweep}.csv"); os.makedirs(cfg["output_dir"], exist_ok=True)
    df.to_csv(out, index=False)
    print("\n=== sorted by clean AUC (full val set) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
