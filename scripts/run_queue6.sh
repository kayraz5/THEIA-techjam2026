#!/usr/bin/env bash
# Queue 6: external validation of the #10 winner (sid+laion+mjv5, 0.9994 on the designated benchmark).
# The designated benchmark is saturated and cannot tell genuine generalization from benchmark-fit;
# Community Forensics-Eval can. Requires restoring the scaled feature caches that queue5 overwrote.
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
while pgrep -f "run_queue5.sh" > /dev/null; do sleep 20; done
echo "### restore_scaled_caches $(date)"
for f in data/features/*_train_deg_e[01]_*.npz; do cp -n "$f" "$f.gan_bak" 2>/dev/null || true; done   # keep GAN arm features
for b in data/features/*_train_deg_e[01]_*.npz.scaled_bak; do cp -f "$b" "${b%.scaled_bak}"; done       # restore scaled
echo "### refit_mjv5_head $(date)"
python scripts/data_ablation.py --config configs/frozen_siglip2_giant_scaled.yaml \
    --keep sid_set --keep laion5b --keep mj_v5 --tag sid_laion_mjv5 --save_head || { echo "Q6_FAILED refit"; exit 1; }
echo "### external_eval_mjv5 $(date)"
caffeinate -i python scripts/eval_external.py --config configs/frozen_siglip2_giant_scaled.yaml \
    --checkpoint results/frozen_siglip2_giant_scaled/head_ablation_sid_laion_mjv5.pt || echo "Q6_FAILED external"
echo "Q6_DONE $(date)"
