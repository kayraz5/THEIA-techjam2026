#!/bin/bash
# #20 (MJv4 vs MJv5 = vintage) + #22 (pixel-space diffusion) from ONE superset extraction.
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -o pipefail
CFG=configs/frozen_siglip2_giant_super.yaml
OUT=results/frozen_siglip2_giant_super

echo "### superset extraction $(date)"
python -m src.train --config $CFG || exit 1

echo "### ablations $(date)"
# --- #20: vintage, one variable (mj_v5 -> mj_v4), mirrors the 2x2's arm A ---
python scripts/data_ablation.py --config $CFG --tag v20_mjv5 \
  --keep sid_set --keep laion5b --keep mj_v5 --save_head
python scripts/data_ablation.py --config $CFG --tag v20_mjv4 \
  --keep sid_set --keep laion5b --keep mj_v4 --save_head
# both vintages together - does MJv4 still hurt when MJv5 is present?
python scripts/data_ablation.py --config $CFG --tag v20_both \
  --keep sid_set --keep laion5b --keep mj_v5 --keep mj_v4 --save_head

# --- #22: pixel-space diffusion on the ship base, one variable ---
python scripts/data_ablation.py --config $CFG --tag v22_ship \
  --keep sid_set --keep laion5b --keep test2017 --keep mj_v5 --keep GAN_based --save_head
python scripts/data_ablation.py --config $CFG --tag v22_pixdiff \
  --keep sid_set --keep laion5b --keep test2017 --keep mj_v5 --keep GAN_based \
  --keep /ADM/ --keep /DDIM/ --keep /DDPM/ --save_head

echo "### external evals $(date)"
for t in v20_mjv5 v20_mjv4 v20_both v22_ship v22_pixdiff; do
  echo "--- CF: $t"
  python scripts/eval_external.py --config $CFG --checkpoint $OUT/head_ablation_${t}.pt --name cf_$t
  echo "--- RR: $t"
  python scripts/eval_external.py --config $CFG --checkpoint $OUT/head_ablation_${t}.pt \
    --source dir --root data/rrdataset/images --name rr_$t
done
echo "Q9_DONE $(date)"
