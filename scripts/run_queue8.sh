#!/bin/bash
# #25: second external validation set (RRDataset). Scores BOTH the shipped arm and the
# benchmark-optimal arm, so the generalization claim is bounded by two datasets, not one.
cd "$(dirname "$0")/.."
source .venv/bin/activate
echo "### rrdataset $(date)"
for arm in ship mjv5; do
  echo "--- RRDataset: $arm"
  python scripts/eval_external.py \
    --config configs/frozen_siglip2_giant_${arm}.yaml \
    --checkpoint results/frozen_siglip2_giant_${arm}/head_best.pt \
    --source dir --root data/rrdataset/images --name rrdataset
done
echo "Q8_DONE $(date)"
