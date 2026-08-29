"""Shared head-training + grid-scoring helpers used by the analysis scripts in scripts/.

Mirrors src/train.py::train_frozen exactly (optimiser, cosine schedule, loss, weighted sampler,
best-robust checkpointing) so every experiment differs only in the variable under test."""
from __future__ import annotations
import math
import numpy as np, torch, torch.nn as nn
from src.common import set_seed, metrics
from src.degradation import EVAL_CONDITIONS
from src.features import cache_path
from src.models import LinearHead
from src.train import FocalLoss, cosine_warmup, sampler_for

BB, SZ = "google/siglip2-giant-opt-patch16-384", 384


def load_cached(cfg, tag):
    z = np.load(cache_path(cfg["features"]["cache_dir"], BB, SZ, tag), allow_pickle=True)
    return z["feats"], z["labels"]


def train_head_on(cfg, Xtr, ytr, device, epochs=None, loss_name=None, gamma=None):
    tc = cfg["train"]; set_seed(cfg["seed"])
    epochs = epochs or tc["epochs"]; loss_name = loss_name or tc["loss"]; gamma = gamma or tc["focal_gamma"]
    head = LinearHead(Xtr.shape[1]).to(device)
    Xt = torch.from_numpy(Xtr).to(device); yt = torch.from_numpy(ytr).to(device)
    spe = math.ceil(len(Xt) / tc["batch_size"]); total = spe * epochs
    opt = torch.optim.AdamW(head.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = cosine_warmup(opt, total, spe * tc["warmup_epochs"])
    loss_fn = FocalLoss(gamma, tc["focal_alpha"]) if loss_name == "focal" else nn.CrossEntropyLoss()
    sampler = sampler_for(ytr, tc["weighted_sampler"])
    Xvd, yvd = load_cached(cfg, "valsub_deg")
    def score(X):
        with torch.no_grad():
            return torch.softmax(head(torch.from_numpy(X).to(device)).float(), -1)[:, 1].cpu().numpy()
    best, best_sd = -1, None
    for _ in range(epochs):
        head.train()
        idx = torch.as_tensor(list(sampler)) if sampler else torch.randperm(len(Xt))
        for b in range(spe):
            bi = idx[b * tc["batch_size"]:(b + 1) * tc["batch_size"]].to(device)
            loss = loss_fn(head(Xt[bi]), yt[bi]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        head.eval()
        r = metrics(yvd, score(Xvd))["auc"]
        if r > best: best, best_sd = r, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_sd); head.eval()
    return head, best, score


def eval_grid(cfg, score, prefix="valsub_"):
    aucs = {}
    for cname, lv in EVAL_CONDITIONS:
        t = "clean" if cname == "clean" else f"{cname}_{lv}"
        Xv, yv = load_cached(cfg, f"{prefix}{t}")
        aucs[t] = metrics(yv, score(Xv))["auc"]
    Xf, yf = load_cached(cfg, "val_clean")
    deg = {k: v for k, v in aucs.items() if k != "clean"}
    worst = min(deg, key=deg.get)
    return {"clean_auc_full": metrics(yf, score(Xf))["auc"], "clean_auc_subset": aucs["clean"],
            "worst_cell": worst, "worst_auc": deg[worst],
            "max_delta": max(aucs["clean"] - v for v in deg.values()),
            "mean_degraded": float(np.mean(list(deg.values())))}
