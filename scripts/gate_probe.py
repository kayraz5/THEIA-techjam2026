"""Issue #7: can a linear probe on frozen SigLIP2 features identify WHICH degradation was applied?

This is the premise of the gated multi-expert design in issue #2. Tested entirely on cached eval
features (15 conditions x the same underlying images), so it costs seconds and no GPU.

Split is grouped by image identity: the same photo appears in all 15 conditions, so a random row
split would put near-duplicates on both sides and inflate accuracy.

  python scripts/gate_probe.py --config configs/frozen_siglip2_giant_sidonly.yaml
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from src.common import load_config
from src.degradation import EVAL_CONDITIONS
from src.features import cache_path

BB, SZ = "google/siglip2-giant-opt-patch16-384", 384


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = load_config(a.config); cd = cfg["features"]["cache_dir"]
    X, ycond, yfam, groups = [], [], [], []
    for name, lv in EVAL_CONDITIONS:
        tag = "clean" if name == "clean" else f"{name}_{lv}"
        p = cache_path(cd, BB, SZ, f"valsub_{tag}")
        if not os.path.exists(p):
            print(f"  missing {tag}, skipping"); continue
        z = np.load(p, allow_pickle=True)
        X.append(z["feats"]); ycond += [tag] * len(z["feats"]); yfam += [name] * len(z["feats"])
        groups += list(z["paths"])
    X = np.concatenate(X); ycond = np.array(ycond); yfam = np.array(yfam); groups = np.array(groups)
    print(f"[gate] {len(X)} rows, {len(set(ycond))} conditions, {len(set(groups))} unique images")

    # grouped split: an image is entirely in train or entirely in test
    imgs = sorted(set(groups)); rng = np.random.default_rng(cfg["seed"]); rng.shuffle(imgs)
    test_imgs = set(imgs[: len(imgs) // 3])
    te = np.array([g in test_imgs for g in groups]); tr = ~te
    print(f"[gate] grouped split: {tr.sum()} train rows / {te.sum()} test rows "
          f"({len(imgs)-len(test_imgs)} / {len(test_imgs)} images, no image on both sides)")

    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    out = {}
    for label, y in (("15-class (family+severity)", ycond), ("7-class (family only)", yfam),
                     ("2-class (clean vs degraded)", np.where(yfam == "clean", "clean", "degraded"))):
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, y[tr])
        pred = clf.predict(Xte); acc = accuracy_score(y[te], pred)
        chance = pd.Series(y[te]).value_counts(normalize=True).max()
        print(f"\n[gate] {label}: accuracy {acc:.4f}  (majority-class baseline {chance:.4f})")
        out[label] = {"accuracy": float(acc), "majority_baseline": float(chance)}
        if label.startswith("7-class"):
            labs = sorted(set(y)); cm = confusion_matrix(y[te], pred, labels=labs, normalize="true")
            df = pd.DataFrame((cm * 100).round(1), index=labs, columns=labs)
            print("\nrow = true family, col = predicted, % of row:")
            print(df.to_string())
            out["family_confusion_pct"] = df.to_dict()
            out["family_recall"] = {l: float(cm[i, i]) for i, l in enumerate(labs)}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/gate_probe.json", "w"), indent=1)
    print("\n-> results/gate_probe.json")


if __name__ == "__main__":
    main()
