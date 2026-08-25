"""Training.

frozen mode : extract (cached) degraded features for K augmentation epochs + clean/degraded val features,
              then train the linear head on cached features (seconds).
lora mode   : end-to-end with LoRA adapters (rank 32) + head, AMP, on-the-fly degradation.

python -m src.train --config configs/frozen_siglip2_giant.yaml
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from src.common import load_config, set_seed, get_device, build_backbone, metrics
from src.data import eval_subset, build_train, wildfake_validation, wildfake_alt_real, summarize, ExclusionList
from src.degradation import RandomDegradation
from src.features import get_or_extract, ImageDS
from src.models import LinearHead, Detector, apply_lora, total_param_report


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super().__init__(); self.g, self.a = gamma, alpha
    def forward(self, logits, y):
        logp = F.log_softmax(logits.float(), -1); p = logp.exp()
        lp = logp.gather(1, y[:, None])[:, 0]; pt = p.gather(1, y[:, None])[:, 0]
        alpha = torch.where(y == 1, torch.full_like(pt, self.a), torch.full_like(pt, 1 - self.a))
        return (-alpha * (1 - pt) ** self.g * lp).mean()


def make_loss(tc):
    return FocalLoss(tc["focal_gamma"], tc["focal_alpha"]) if tc["loss"] == "focal" else nn.CrossEntropyLoss()


def cosine_warmup(opt, total, warmup):
    def f(step):
        if step < warmup: return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup); return 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def sampler_for(labels, enabled):
    if not enabled: return None
    labels = np.asarray(labels); w = 1.0 / np.bincount(labels)[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(labels), replacement=True)


def robust_auc(head_or_fn, val_sets: dict) -> dict:
    """val_sets: name -> (feats, labels). 'clean' plus degraded conditions; robust = mean over degraded."""
    out = {}
    for k, (X, y) in val_sets.items():
        out[k] = metrics(y, head_or_fn(X))["auc"]
    deg = [v for k, v in out.items() if k != "clean"]
    out["robust"] = float(np.mean(deg)) if deg else out["clean"]
    return out


# ------------------------------------------------------------------ frozen ----------------------------
def train_frozen(cfg, device, backbone, train_s, val_s, alt_s):
    tc, dc = cfg["train"], cfg["degradation"]
    aug = RandomDegradation(dc["prob"], dc["min_transforms"], dc["max_transforms"], dc["hflip"], dc["seed"], backend=dc.get("backend", "table"))
    # K augmentation epochs of training features (each a fresh degradation draw), cached
    K = tc.get("feature_epochs", 3)
    Xs, ys = [], []
    for k in range(K):
        X, y, _, _ = get_or_extract(cfg, backbone, train_s, f"train_deg_e{k}", device, degrade=aug, seed=cfg["seed"] * 100 + k)
        Xs.append(X); ys.append(y)
    Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
    # validation: clean (full set) + a random-degraded copy of the eval subset (for robust checkpointing), + alt real set
    Xv, yv, _, _ = get_or_extract(cfg, backbone, val_s, "val_clean", device)
    sub = eval_subset(val_s, cfg)
    Xvd, yvd, _, _ = get_or_extract(cfg, backbone, sub, "valsub_deg", device, degrade=aug, seed=999)
    Xa, ya, _, _ = get_or_extract(cfg, backbone, alt_s, "altreal_clean", device)
    val_sets = {"clean": (Xv, yv), "degraded": (Xvd, yvd)}

    head = LinearHead(backbone.feature_dim).to(device)
    Xt = torch.from_numpy(Xtr).to(device); yt = torch.from_numpy(ytr).to(device)
    steps_per_epoch = math.ceil(len(Xt) / tc["batch_size"]); total = steps_per_epoch * tc["epochs"]
    opt = torch.optim.AdamW(head.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = cosine_warmup(opt, total, steps_per_epoch * tc["warmup_epochs"]); loss_fn = make_loss(tc)
    sampler = sampler_for(ytr, tc["weighted_sampler"])
    def score(X):
        with torch.no_grad(): return torch.softmax(head(torch.from_numpy(X).to(device)).float(), -1)[:, 1].cpu().numpy()
    best, hist = -1, []
    os.makedirs(cfg["output_dir"], exist_ok=True)
    for ep in range(tc["epochs"]):
        head.train()
        idx = torch.as_tensor(list(sampler)) if sampler else torch.randperm(len(Xt))
        tl = 0
        for b in range(steps_per_epoch):
            bi = idx[b * tc["batch_size"]:(b + 1) * tc["batch_size"]].to(device)
            loss = loss_fn(head(Xt[bi]), yt[bi]); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        head.eval()
        tr_auc = metrics(ytr, score(Xtr))["auc"]; va = robust_auc(score, val_sets)
        rec = {"epoch": ep, "loss": tl / steps_per_epoch, "train_auc": tr_auc, **{f"val_{k}": v for k, v in va.items()}}
        hist.append(rec); print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}))
        if va["robust"] > best:
            best = va["robust"]; torch.save({"head": head.state_dict(), "cfg": cfg, "backbone": backbone.info.__dict__, "epoch": ep},
                                            os.path.join(cfg["output_dir"], "head_best.pt"))
    json.dump(hist, open(os.path.join(cfg["output_dir"], "train_history.json"), "w"), indent=1)
    head.load_state_dict(torch.load(os.path.join(cfg["output_dir"], "head_best.pt"))["head"])
    # checkpoint summary incl. shortcut check: AUC of (COCO reals vs fakes) vs (alt reals vs fakes)
    s_fake = score(Xv[yv == 1]); s_coco = score(Xv[yv == 0]); s_alt = score(Xa)
    summary = {
        "clean_auc_coco_real": metrics(np.r_[np.zeros(len(s_coco)), np.ones(len(s_fake))], np.r_[s_coco, s_fake])["auc"],
        "clean_auc_alt_real": metrics(np.r_[np.zeros(len(s_alt)), np.ones(len(s_fake))], np.r_[s_alt, s_fake])["auc"],
        "best_robust_val_auc": best,
    }
    summary["shortcut_gap"] = summary["clean_auc_coco_real"] - summary["clean_auc_alt_real"]
    print("[checkpoint] " + json.dumps({k: round(v, 4) for k, v in summary.items()}))
    json.dump(summary, open(os.path.join(cfg["output_dir"], "train_summary.json"), "w"), indent=1)


# ------------------------------------------------------------------ lora ------------------------------
def train_lora(cfg, device, backbone, train_s, val_s, alt_s):
    tc, dc, lc = cfg["train"], cfg["degradation"], cfg["lora"]
    backbone = apply_lora(backbone, lc["rank"], lc["alpha"], lc["dropout"]).train()
    head = LinearHead(backbone.feature_dim).to(device); det = Detector(backbone, head).to(device); total_param_report(det)
    i = backbone.info
    aug = RandomDegradation(dc["prob"], dc["min_transforms"], dc["max_transforms"], dc["hflip"], dc["seed"], backend=dc.get("backend", "table"))
    ds = ImageDS(train_s, i.image_size, i.mean, i.std, degrade=aug, seed=cfg["seed"])
    labels = [s.label for s in train_s]; sampler = sampler_for(labels, tc["weighted_sampler"])
    dl = DataLoader(ds, batch_size=tc["batch_size"], sampler=sampler, shuffle=sampler is None, num_workers=2, drop_last=True)
    params = [p for p in det.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = len(dl) * tc["epochs"]; sched = cosine_warmup(opt, total, len(dl) * tc["warmup_epochs"]); loss_fn = make_loss(tc)
    amp = tc["amp"] and device.type in ("cuda", "mps"); adt = torch.bfloat16
    def eval_set(samples, fixed=None, degrade=None, seed=0):
        det.eval(); dsv = ImageDS(samples, i.image_size, i.mean, i.std, degrade=degrade, fixed=fixed, seed=seed)
        out = np.zeros(len(dsv)); ys = np.zeros(len(dsv), int)
        with torch.no_grad():
            for x, y, idx, _ in DataLoader(dsv, batch_size=cfg["eval"]["batch_size"], num_workers=2):
                with torch.autocast(device.type, dtype=adt, enabled=amp):
                    p = det.predict_proba(x.to(device))
                out[idx.numpy()] = p.float().cpu().numpy(); ys[idx.numpy()] = y.numpy()
        det.train(); return out, ys
    best, hist = -1, []; os.makedirs(cfg["output_dir"], exist_ok=True); step = 0
    for ep in range(tc["epochs"]):
        ds.seed = cfg["seed"] * 100 + ep; tl = 0; t0 = time.time()
        for x, y, _, _ in tqdm(dl, desc=f"lora ep{ep}", leave=False):
            with torch.autocast(device.type, dtype=adt, enabled=amp):
                loss = loss_fn(det(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); tl += loss.item(); step += 1
        pc, yc = eval_set(eval_subset(val_s, cfg)); pd_, yd = eval_set(eval_subset(val_s, cfg), degrade=aug, seed=999)
        va = {"clean": metrics(yc, pc)["auc"], "degraded": metrics(yd, pd_)["auc"]}; va["robust"] = va["degraded"]
        rec = {"epoch": ep, "loss": tl / len(dl), "time_s": time.time() - t0, **{f"val_{k}": v for k, v in va.items()}}
        hist.append(rec); print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}))
        if va["robust"] > best:
            best = va["robust"]
            sd = {k: v for k, v in det.state_dict().items() if "lora_" in k or k.startswith("head.")}
            torch.save({"state": sd, "cfg": cfg, "backbone": backbone.info.__dict__, "epoch": ep}, os.path.join(cfg["output_dir"], "lora_best.pt"))
    json.dump(hist, open(os.path.join(cfg["output_dir"], "train_history.json"), "w"), indent=1)
    pa, _ = eval_set(alt_s); pc, yc = eval_set(val_s)
    s_fake, s_coco = pc[yc == 1], pc[yc == 0]
    summary = {"clean_auc_coco_real": metrics(np.r_[np.zeros(len(s_coco)), np.ones(len(s_fake))], np.r_[s_coco, s_fake])["auc"],
               "clean_auc_alt_real": metrics(np.r_[np.zeros(len(pa)), np.ones(len(s_fake))], np.r_[pa, s_fake])["auc"], "best_robust_val_auc": best}
    summary["shortcut_gap"] = summary["clean_auc_coco_real"] - summary["clean_auc_alt_real"]
    print("[checkpoint] " + json.dumps({k: round(v, 4) for k, v in summary.items()}))
    json.dump(summary, open(os.path.join(cfg["output_dir"], "train_summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = load_config(a.config); set_seed(cfg["seed"]); device = get_device(); print(f"[env] device={device}")
    backbone = build_backbone(cfg, device)
    train_s = build_train(cfg); summarize(train_s, "train")
    root = cfg["data"]["wildfake"]["root"]
    val_s = wildfake_validation(root); summarize(val_s, "validation (held-out)")
    alt_s = wildfake_alt_real(root, n=cfg["data"]["alt_real"]["n"], seed=cfg["seed"]); summarize(alt_s, "alt real (held-out)")
    assert len(val_s) > 0 and len(alt_s) > 0, "validation sets empty — download data first"
    # --- mandatory leakage check (spec §3 rule 1): every training run, automatic ---
    ExclusionList(cfg["data"]["validation"]["exclusion_list"]).assert_disjoint([s.path for s in train_s], "train")
    if cfg["mode"] == "frozen":
        for p in backbone.parameters(): p.requires_grad_(False)
        train_frozen(cfg, device, backbone, train_s, val_s, alt_s)
    elif cfg["mode"] == "lora":
        train_lora(cfg, device, backbone, train_s, val_s, alt_s)
    else:
        raise ValueError(cfg["mode"])


if __name__ == "__main__":
    main()
