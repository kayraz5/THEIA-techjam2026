#!/bin/bash
# Chains after queue9. #24 (official NTIRE distortion backend) then #19 (source screening).
cd "$(dirname "$0")/.."
source .venv/bin/activate
FEAT=data/features/google_siglip2-giant-opt-patch16-384_384px

echo "### waiting for queue9 $(date)"
while ! grep -q Q9_DONE logs/queue9.log 2>/dev/null; do sleep 60; done
echo "### queue9 finished, backing up its caches $(date)"
for f in ${FEAT}_train_deg_e[01]_*.npz; do cp -n "$f" "$f.super_bak"; done

# ---------------- #24: official distortion backend ----------------
echo "### 24 official backend $(date)"
python -m src.train    --config configs/frozen_siglip2_giant_official.yaml || exit 1
python -m src.evaluate --config configs/frozen_siglip2_giant_official.yaml
python scripts/eval_external.py --config configs/frozen_siglip2_giant_official.yaml --name community_forensics
python scripts/eval_external.py --config configs/frozen_siglip2_giant_official.yaml \
    --source dir --root data/rrdataset/images --name rrdataset
for f in ${FEAT}_train_deg_e[01]_*.npz; do cp -n "$f" "$f.official_bak"; done
echo "### 24 done $(date)"

# ---------------- #19: per-source screening ----------------
echo "### 19 screening extraction $(date)"
CFG=configs/frozen_siglip2_giant_screen.yaml
OUT=results/frozen_siglip2_giant_screen
python -m src.train --config $CFG || exit 1

SHIP="--keep sid_set --keep laion5b --keep test2017 --keep mj_v5 --keep GAN_based"
echo "### 19 ablations $(date)"
python scripts/data_ablation.py --config $CFG --tag s_ship_base $SHIP --save_head
# one new FAKE family at a time on top of the ship base
for g in VQDM Imagen MAGE MAE VQGAN VQVAE; do
  python scripts/data_ablation.py --config $CFG --tag s_fake_$g $SHIP --keep /$g/ --save_head
done
# one new REAL domain at a time
for r in ffhq celebahq afhq church; do
  python scripts/data_ablation.py --config $CFG --tag s_real_$r $SHIP --keep /$r/ --save_head
done
# GAPL's caution: winners may not compose, so test the full mix as a mix
python scripts/data_ablation.py --config $CFG --tag s_all $SHIP \
  --keep /VQDM/ --keep /Imagen/ --keep /MAGE/ --keep /MAE/ --keep /VQGAN/ --keep /VQVAE/ \
  --keep /ffhq/ --keep /celebahq/ --keep /afhq/ --keep /church/ --save_head

echo "### 19 external evals $(date)"
for t in s_ship_base s_fake_VQDM s_fake_Imagen s_fake_MAGE s_fake_MAE s_fake_VQGAN s_fake_VQVAE \
         s_real_ffhq s_real_celebahq s_real_afhq s_real_church s_all; do
  python scripts/eval_external.py --config $CFG --checkpoint $OUT/head_ablation_${t}.pt --name cf_$t
  python scripts/eval_external.py --config $CFG --checkpoint $OUT/head_ablation_${t}.pt \
    --source dir --root data/rrdataset/images --name rr_$t
done
echo "Q10_DONE $(date)"
