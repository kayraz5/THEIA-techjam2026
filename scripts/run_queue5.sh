#!/usr/bin/env bash
# Queue 5 (#15 follow-up): GAN-augmented arm — train, score the designated benchmark, then score the
# external Community Forensics set. Eval feature caches are backbone-keyed, so the designated grid is free.
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
while pgrep -f "run_queue3.sh" > /dev/null; do sleep 20; done
for f in data/features/*_train_deg_e[01]_*.npz; do cp -n "$f" "$f.scaled_bak" 2>/dev/null || true; done
echo "### gan_train $(date)"
caffeinate -i python -m src.train --config configs/frozen_siglip2_giant_gan.yaml || { echo "Q5_FAILED train"; exit 1; }
echo "### gan_eval_designated $(date)"
caffeinate -i python -m src.evaluate --config configs/frozen_siglip2_giant_gan.yaml || echo "Q5_FAILED eval"
echo "### gan_eval_external $(date)"
caffeinate -i python scripts/eval_external.py --config configs/frozen_siglip2_giant_gan.yaml \
    --checkpoint results/frozen_siglip2_giant_gan/head_best.pt || echo "Q5_FAILED external"
echo "### results $(date)"
python scripts/make_results.py || echo "Q5_FAILED make_results"
echo "Q5_DONE $(date)"
