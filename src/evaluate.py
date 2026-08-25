"""Evaluation harness (spec §6).

python -m src.evaluate --config configs/frozen_siglip2_giant.yaml [--checkpoint results/.../head_best.pt]

Outputs into <output_dir>/eval/:
  auc_grid.csv / auc_grid.png      per-transform x severity AUC, delta (clean - cond), error (1-AUC), on both real sets
  thresholds.csv, roc.png, fpr_vs_threshold.png
  errors_fp.png / errors_fn.png    contact sheets of the 20 most confident FP / FN, grouped by transform
  summary.json                     headline numbers incl. the COCO-vs-alt-real shortcut gap
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_curve
from PIL import Image, ImageDraw
from src.common import load_config, set_seed, get_device, build_backbone, metrics, squish_resize
from src.data import wildfake_validation, wildfake_alt_real, summarize, eval_subset
from src.degradation import EVAL_CONDITIONS, LEVELS, apply_transform
from src.features import get_or_extract
from src.models import LinearHead


def load_scorer(cfg, backbone, device, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = LinearHead(backbone.feature_dim).to(device)
    if "head" in ck: head.load_state_dict(ck["head"])
    else:
        from src.models import apply_lora
        lc = cfg["lora"]; apply_lora(backbone, lc["rank"], lc["alpha"], lc["dropout"])
        sd = ck["state"]; backbone.model.load_state_dict({k.split("backbone.model.", 1)[1]: v for k, v in sd.items() if k.startswith("backbone.")}, strict=False)
        head.load_state_dict({k.split("head.", 1)[1]: v for k, v in sd.items() if k.startswith("head.")})
        backbone.eval()
    head.eval()
    def score(X):
        with torch.no_grad(): return torch.softmax(head(torch.from_numpy(X).to(device)).float(), -1)[:, 1].cpu().numpy()
    return score


def per_condition(cfg, backbone, device, score, val_s, alt_s, seed):
    rows, all_scores = [], {}
    for name, lv in EVAL_CONDITIONS:
        tag = "clean" if name == "clean" else f"{name}_{lv}"
        Xv, yv, _, pv = get_or_extract(cfg, backbone, val_s, f"valsub_{tag}", device, fixed=(name, lv), seed=seed)
        Xa, _, _, pa = get_or_extract(cfg, backbone, alt_s, f"altsub_{tag}", device, fixed=(name, lv), seed=seed)
        sv, sa = score(Xv), score(Xa)
        fake = sv[yv == 1]; coco = sv[yv == 0]
        m_coco = metrics(np.r_[np.zeros(len(coco)), np.ones(len(fake))], np.r_[coco, fake])
        m_alt = metrics(np.r_[np.zeros(len(sa)), np.ones(len(fake))], np.r_[sa, fake])
        rows.append({"transform": name, "level": "" if lv is None else lv, "auc_coco_real": m_coco["auc"], "auc_alt_real": m_alt["auc"],
                     "err_coco_real": m_coco["err"], "err_alt_real": m_alt["err"], "bal_acc@0.5": m_coco["bal_acc@0.5"],
                     "acc@0.5": m_coco["acc@0.5"], "majority_baseline": m_coco["majority_baseline"], "n_real_coco": len(coco), "n_fake": len(fake), "n_real_alt": len(sa)})
        all_scores[tag] = (sv, yv, pv, sa, pa)
    df = pd.DataFrame(rows)
    clean = df[df["transform"] == "clean"].iloc[0]
    df["delta_coco"] = clean.auc_coco_real - df.auc_coco_real
    df["delta_alt"] = clean.auc_alt_real - df.auc_alt_real
    df["shortcut_gap"] = df.auc_coco_real - df.auc_alt_real
    return df, all_scores


def heatmap(df, out_png):
    fams = [f for f in LEVELS]; maxl = max(len(v) for v in LEVELS.values())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, col, title, fmt in zip(axes, ["auc_coco_real", "delta_coco", "err_coco_real"],
                                   ["ROC AUC (COCO real vs DALL-E)", "Degradation delta (clean - cond)", "Error = 1 - AUC"], [".4f", "+.3f", ".4f"]):
        M = np.full((len(fams) + 1, maxl), np.nan); lab = np.full(M.shape, "", dtype=object)
        M[0, 0] = df[df["transform"] == "clean"][col].iloc[0]; lab[0, 0] = "clean"
        for i, f in enumerate(fams):
            for j, lv in enumerate(LEVELS[f]):
                r = df[(df["transform"] == f) & (df.level.astype(str) == str(lv))]
                if len(r): M[i + 1, j] = r[col].iloc[0]; lab[i + 1, j] = str(lv)
        ann = np.where(np.isnan(M), "", np.vectorize(lambda v, l: f"{l}\n{v:{fmt}}" if not np.isnan(v) else "")(M, lab))
        sns.heatmap(M, annot=ann, fmt="", ax=ax, yticklabels=["clean"] + fams, xticklabels=[f"lvl{j+1}" for j in range(maxl)],
                    cmap="RdYlGn" if col == "auc_coco_real" else "RdYlGn_r", vmin=(0.8 if col == "auc_coco_real" else 0), vmax=(1.0 if col == "auc_coco_real" else (0.1 if col == "delta_coco" else 0.2)),
                    cbar=True, linewidths=0.5, mask=np.isnan(M), annot_kws={"size": 8})
        ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()


def threshold_analysis(sv, yv, out_dir):
    fpr, tpr, thr = roc_curve(yv, sv)
    rows = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        pred = sv >= t
        rows.append({"threshold": t, "FPR_real": float(pred[yv == 0].mean()), "TPR_fake": float(pred[yv == 1].mean())})
    for target in [0.001, 0.01, 0.05]:  # operating points at fixed FPR
        i = np.searchsorted(fpr, target, side="right") - 1; i = max(i, 0)
        rows.append({"threshold": float(thr[i]), "FPR_real": float(fpr[i]), "TPR_fake": float(tpr[i]), "note": f"@FPR<={target}"})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "thresholds.csv"), index=False)
    plt.figure(figsize=(4.5, 4.5)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "k--", lw=0.5); plt.xscale("log"); plt.xlim(1e-4, 1)
    plt.xlabel("FPR on real (log)"); plt.ylabel("TPR on fake"); plt.title("ROC (clean)"); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "roc.png"), dpi=140); plt.close()
    ts = np.linspace(0, 1, 201); f = [(sv[yv == 0] >= t).mean() for t in ts]
    plt.figure(figsize=(5, 3.5)); plt.plot(ts, f); plt.yscale("symlog", linthresh=1e-3); plt.xlabel("threshold"); plt.ylabel("FPR on real images")
    plt.title("FPR vs threshold (clean)"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fpr_vs_threshold.png"), dpi=140); plt.close()
    return rows


def contact_sheet(items, out_png, title, thumb=160, cols=5):
    """items: list of (path, score, label, cond). Grouped by cond."""
    items = sorted(items, key=lambda x: (x[3], -abs(x[1] - x[2])))
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 34) + 24), "white"); d = ImageDraw.Draw(sheet); d.text((4, 4), title, fill="black")
    for k, (p, s, y, cond) in enumerate(items):
        try: im = squish_resize(Image.open(p), thumb)
        except Exception: im = Image.new("RGB", (thumb, thumb), "gray")
        x, yy = (k % cols) * thumb, 24 + (k // cols) * (thumb + 34)
        sheet.paste(im, (x, yy)); d.text((x + 3, yy + thumb + 2), f"p(fake)={s:.3f} true={'fake' if y else 'real'}", fill="black")
        d.text((x + 3, yy + thumb + 17), f"cond={cond} {os.path.basename(p)[:18]}", fill="black")
    sheet.save(out_png)


def error_analysis(all_scores, out_dir, k=20):
    fps, fns = [], []
    for tag, (sv, yv, pv, _, _) in all_scores.items():
        for s, y, p in zip(sv, yv, pv):
            (fps if y == 0 else fns).append((p, float(s), int(y), tag))
    fps = sorted(fps, key=lambda x: -x[1])[:k]; fns = sorted(fns, key=lambda x: x[1])[:k]
    contact_sheet(fps, os.path.join(out_dir, "errors_fp.png"), f"Top-{k} most confident FALSE POSITIVES (real scored as fake), grouped by transform")
    contact_sheet(fns, os.path.join(out_dir, "errors_fn.png"), f"Top-{k} most confident FALSE NEGATIVES (fake scored as real), grouped by transform")
    pd.DataFrame(fps + fns, columns=["path", "score", "label", "condition"]).to_csv(os.path.join(out_dir, "errors_top.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n_eval_max", type=int, default=None); a = ap.parse_args()
    cfg = load_config(a.config); set_seed(cfg["seed"]); device = get_device()
    out_dir = os.path.join(cfg["output_dir"], "eval"); os.makedirs(out_dir, exist_ok=True)
    backbone = build_backbone(cfg, device)
    for p in backbone.parameters(): p.requires_grad_(False)
    root = cfg["data"]["wildfake"]["root"]
    val_s = wildfake_validation(root); alt_s = wildfake_alt_real(root, n=cfg["data"]["alt_real"]["n"], seed=cfg["seed"])
    if a.n_eval_max is not None: cfg["eval"]["n_eval_max"] = a.n_eval_max
    full_val = val_s; val_s = eval_subset(val_s, cfg)
    if len(val_s) < len(full_val): print(f"[eval] NOTE: per-condition grid uses a seeded subset of {len(val_s)}/{len(full_val)} validation images")
    summarize(val_s, "eval validation"); summarize(alt_s, "eval alt real")
    ckpt = a.checkpoint or os.path.join(cfg["output_dir"], "head_best.pt" if cfg["mode"] == "frozen" else "lora_best.pt")
    score = load_scorer(cfg, backbone, device, ckpt)
    df, all_scores = per_condition(cfg, backbone, device, score, val_s, alt_s, cfg["degradation"]["seed"])
    df.to_csv(os.path.join(out_dir, "auc_grid.csv"), index=False); heatmap(df, os.path.join(out_dir, "auc_grid.png"))
    sv, yv, *_ = all_scores["clean"]; thr = threshold_analysis(sv, yv, out_dir); error_analysis(all_scores, out_dir)
    Xf, yf, _, pf = get_or_extract(cfg, backbone, full_val, "val_clean", device); sf = score(Xf); mf = metrics(yf, sf)
    # The designated DALL-E Advanced set contains many byte-identical duplicates across session folders; also report deduplicated.
    from src.data import content_hash
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex: hs = list(ex.map(content_hash, pf.tolist()))
    seen, keep = set(), np.zeros(len(pf), bool)
    for i, h in enumerate(hs):
        if h not in seen: seen.add(h); keep[i] = True
    md = metrics(yf[keep], sf[keep])
    clean = df[df["transform"] == "clean"].iloc[0]; deg = df[df["transform"] != "clean"]
    worst = deg.loc[deg.auc_coco_real.idxmin()]
    summary = {
        "config": cfg["name"], "backbone": backbone.info.name, "backbone_params_B": backbone.info.n_params / 1e9, "checkpoint": ckpt,
        "n_val_real_coco": int(clean.n_real_coco), "n_val_fake": int(clean.n_fake), "n_alt_real": int(clean.n_real_alt),
        "class_ratio_real:fake": f"1:{clean.n_fake / clean.n_real_coco:.2f}", "majority_class_baseline_acc": float(clean.majority_baseline),
        "clean_auc_full_set": mf["auc"], "n_full_set": mf["n"], "clean_auc_full_set_dedup": md["auc"], "n_full_set_dedup": md["n"], "n_fake_dedup": md["n_fake"], "n_real_dedup": md["n_real"], "full_set_bal_acc@0.5": mf["bal_acc@0.5"], "full_set_acc@0.5": mf["acc@0.5"],
        "clean_auc": float(clean.auc_coco_real), "clean_auc_alt_real": float(clean.auc_alt_real), "shortcut_gap_clean": float(clean.shortcut_gap),
        "worst_transform": f"{worst["transform"]}@{worst["level"]}", "worst_auc": float(worst.auc_coco_real), "worst_auc_alt_real": float(worst.auc_alt_real),
        "max_delta": float(deg.delta_coco.max()), "mean_degraded_auc": float(deg.auc_coco_real.mean()),
        "clean_bal_acc@0.5": float(clean["bal_acc@0.5"]), "clean_acc@0.5": float(clean["acc@0.5"]), "thresholds": thr,
    }
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print("\n================ EVAL SUMMARY ================")
    print(df[["transform", "level", "auc_coco_real", "auc_alt_real", "shortcut_gap", "delta_coco", "err_coco_real"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nclean AUC FULL val set     : {summary['clean_auc_full_set']:.4f} (n={summary['n_full_set']})   deduplicated: {summary['clean_auc_full_set_dedup']:.4f} (n={summary['n_full_set_dedup']}: {summary['n_real_dedup']} real / {summary['n_fake_dedup']} fake)")
    print(f"clean AUC (subset, COCO)   : {summary['clean_auc']:.4f}   error {1-summary['clean_auc']:.4f}")
    print(f"clean AUC (ALT real)       : {summary['clean_auc_alt_real']:.4f}   <-- shortcut check; gap = {summary['shortcut_gap_clean']:+.4f}")
    print(f"worst single transform     : {summary['worst_transform']} AUC {summary['worst_auc']:.4f}   max delta {summary['max_delta']:.4f}")
    print(f"acc@0.5 {summary['clean_acc@0.5']:.4f} | balanced acc {summary['clean_bal_acc@0.5']:.4f} | majority baseline {summary['majority_class_baseline_acc']:.4f} (ratio {summary['class_ratio_real:fake']})")
    print(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
