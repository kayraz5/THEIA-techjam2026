"""Issue #15: score a trained checkpoint on an EXTERNAL held-out set the model never trained on.

Currently supports Community Forensics-Eval (OwensLab/CommunityForensics-Eval) — fakes from a wide
range of community and commercial generators, with per-generator metadata. Reported alongside the
designated benchmark to expose the generalization gap honestly.

  python scripts/eval_external.py --config configs/frozen_siglip2_giant_sidonly.yaml \
      --checkpoint results/frozen_siglip2_giant_sidonly/head_best.pt
"""
from __future__ import annotations
import argparse, glob, io, json, os, sys
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import load_config, set_seed, get_device, build_backbone, squish_resize, to_tensor, metrics
from src.models import LinearHead


class ParquetImages(Dataset):
    def __init__(self, rows, size, mean, std):
        self.rows, self.size, self.mean, self.std = rows, size, mean, std
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        try:
            img = Image.open(io.BytesIO(r["bytes"])).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.size, self.size))
        return to_tensor(squish_resize(img, self.size), self.mean, self.std), r["label"], i


def load_rows(pattern, img_col, limit=0):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(pattern)):
        t = pq.read_table(f)
        names = t.schema.names
        img = t.column(img_col).to_pylist(); lab = t.column("label").to_pylist()
        meta = {c: t.column(c).to_pylist() for c in ("model_name", "architecture", "real_source") if c in names}
        for j in range(t.num_rows):
            d = img[j]; b = d["bytes"] if isinstance(d, dict) else d
            rows.append({"bytes": b, "label": int(lab[j]),
                         **{k: v[j] for k, v in meta.items()}})
    if limit and len(rows) > limit:
        import random; random.Random(0).shuffle(rows); rows = rows[:limit]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pattern", default="data/commforensics/data/*.parquet")
    ap.add_argument("--img_col", default="image_data"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default="community_forensics")
    a = ap.parse_args()
    cfg = load_config(a.config); set_seed(cfg["seed"]); device = get_device()
    rows = load_rows(a.pattern, a.img_col, a.limit)
    print(f"[external] {len(rows)} images from {a.name}: "
          f"{sum(1 for r in rows if r['label']==0)} real / {sum(1 for r in rows if r['label']==1)} fake")

    backbone = build_backbone(cfg, device)
    for p in backbone.parameters(): p.requires_grad_(False)
    ck = torch.load(a.checkpoint or os.path.join(cfg["output_dir"], "head_best.pt"), map_location="cpu", weights_only=False)
    head = LinearHead(backbone.feature_dim).to(device); head.load_state_dict(ck["head"]); head.eval()

    i = backbone.info; ds = ParquetImages(rows, i.image_size, i.mean, i.std)
    dl = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], num_workers=2)
    scores = np.zeros(len(ds)); labels = np.zeros(len(ds), int)
    dtype = next(backbone.parameters()).dtype
    with torch.no_grad():
        for x, y, idx in tqdm(dl, desc=a.name, leave=False):
            f = backbone(x.to(device, dtype))
            p = torch.softmax(head(f.float()), -1)[:, 1]   # backbone emits bf16; head is fp32 (MPS asserts on mixed dtypes)
            scores[idx.numpy()] = p.cpu().numpy(); labels[idx.numpy()] = y.numpy()

    m = metrics(labels, scores)
    df = pd.DataFrame({"score": scores, "label": labels,
                       "model_name": [r.get("model_name", "") for r in rows],
                       "architecture": [r.get("architecture", "") for r in rows]})
    out_dir = os.path.join(cfg["output_dir"], "eval"); os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"external_{a.name}_scores.csv"), index=False)

    print(f"\n=== {a.name}: overall AUC {m['auc']:.4f}  (n={m['n']}, {m['n_real']} real / {m['n_fake']} fake)")
    print(f"    balanced acc@0.5 {m['bal_acc@0.5']:.4f}  acc@0.5 {m['acc@0.5']:.4f}  majority baseline {m['majority_baseline']:.4f}")
    per = []
    reals = df[df.label == 0]
    for gen, g in df[df.label == 1].groupby("model_name"):
        if len(g) < 20: continue
        y = np.r_[np.zeros(len(reals)), np.ones(len(g))]; s = np.r_[reals.score.values, g.score.values]
        per.append({"generator": gen, "architecture": g.architecture.iloc[0], "n_fake": len(g),
                    "auc_vs_all_reals": metrics(y, s)["auc"], "mean_score": float(g.score.mean())})
    if per:
        pdf = pd.DataFrame(per).sort_values("auc_vs_all_reals")
        pdf.to_csv(os.path.join(out_dir, f"external_{a.name}_per_generator.csv"), index=False)
        print("\nper-generator (vs all reals in this set), worst first:")
        print(pdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    json.dump({"name": a.name, "checkpoint": a.checkpoint, **m},
              open(os.path.join(out_dir, f"external_{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
