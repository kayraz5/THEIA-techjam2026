import glob, numpy as np, torch
from sklearn.metrics import roc_auc_score

def load(tag):
    p = glob.glob(f"data/features/google_siglip2-giant-opt-patch16-384_384px_{tag}_*.npz")
    p = [x for x in p if "_flip_" not in x]
    assert len(p) == 1, (tag, p)
    d = np.load(p[0])
    return d["feats"].astype(np.float32), d["labels"].astype(int)

Xc, y = load("valsub_clean")
Xd, y2 = load("valsub_resize_0.25")
assert (y == y2).all()
real, fake = y == 0, y == 1
print(f"eval subset: {real.sum()} real / {fake.sum()} fake")

import sys as _sys
EXTRA = {}
for _a in _sys.argv[1:]:
    if "=" in _a:
        k, v = _a.split("=", 1); EXTRA[k] = v

heads = {
    "A_ship (sid+laion+mjv5)":      "results/frozen_siglip2_giant_2x2/head_ablation_A_ship.pt",
    "B +COCO reals":                "results/frozen_siglip2_giant_2x2/head_ablation_B_plus_cocoreals.pt",
    "C +GAN":                       "results/frozen_siglip2_giant_2x2/head_ablation_C_plus_gan.pt",
    "D +both":                      "results/frozen_siglip2_giant_2x2/head_ablation_D_plus_both.pt",
    "SHIPPED mjv5 (sid 12k)":       "results/frozen_siglip2_giant_mjv5/head_best.pt",
}
heads.update(EXTRA)

def scores(ck, X):
    sd = torch.load(ck, map_location="cpu")
    sd = sd.get("state_dict", sd.get("head", sd))
    w = [v for k, v in sd.items() if v.ndim == 2][0].float()
    b = [v for k, v in sd.items() if v.ndim == 1][0].float()
    logits = torch.from_numpy(X) @ w.T + b
    if logits.shape[-1] == 2:
        return torch.softmax(logits, dim=-1)[:, 1].numpy()
    return torch.sigmoid(logits).squeeze(-1).numpy()

rows = []
for name, ck in heads.items():
    sc, sd_ = scores(ck, Xc), scores(ck, Xd)
    both_clean = roc_auc_score(y, sc)
    both_down  = roc_auc_score(y, sd_)
    # only FAKES downscaled: real rows clean, fake rows downscaled
    mix_f = np.where(fake, sd_, sc); only_fake = roc_auc_score(y, mix_f)
    # only REALS downscaled  <-- the diagnostic
    mix_r = np.where(real, sd_, sc); only_real = roc_auc_score(y, mix_r)
    fp_clean = (sc[real] > 0.5).mean()
    fp_down  = (sd_[real] > 0.5).mean()
    rows.append((name, both_clean, both_down, only_fake, only_real, fp_clean, fp_down))

print(f"\n{'arm':<26} {'clean':>7} {'both↓':>7} {'fakes↓':>7} {'REALS↓':>7} {'FP clean':>9} {'FP down':>8}")
print("-" * 78)
for r in rows:
    print(f"{r[0]:<26} {r[1]:>7.4f} {r[2]:>7.4f} {r[3]:>7.4f} {r[4]:>7.4f} {r[5]*100:>8.1f}% {r[6]*100:>7.1f}%")
