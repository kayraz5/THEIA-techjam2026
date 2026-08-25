"""Baseline comparison table (spec §6f): collects eval/summary.json from several arms into one table."""
import argparse, json, os, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", default="results/comparison.csv"); a = ap.parse_args()
rows = []
for d in a.dirs:
    p = os.path.join(d, "eval", "summary.json")
    if not os.path.exists(p): print(f"skip {d} (no summary)"); continue
    s = json.load(open(p))
    rows.append({"arm": s["config"], "backbone": s["backbone"], "params_B": round(s["backbone_params_B"], 3), "clean_auc": s["clean_auc"],
                 "clean_auc_alt_real": s["clean_auc_alt_real"], "mean_degraded_auc": s["mean_degraded_auc"], "worst": s["worst_transform"], "worst_auc": s["worst_auc"], "max_delta": s["max_delta"]})
df = pd.DataFrame(rows); df.to_csv(a.out, index=False); print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
