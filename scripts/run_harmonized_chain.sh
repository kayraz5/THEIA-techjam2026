#!/usr/bin/env bash
# Issue #3: after the harmonized frozen-arm retrain finishes, run the full benchmark grid,
# the SID-only data ablation (same head procedure, cached features), and refresh RESULTS.md.
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
CFG=configs/frozen_siglip2_giant_harmonized.yaml
while pgrep -f "src.train --config $CFG" > /dev/null; do sleep 20; done
grep -aq "\[checkpoint\]" logs/train_harmonized.log || { echo "CHAIN_FAILED train did not reach checkpoint"; exit 1; }
echo "### evaluate $(date)"
caffeinate -i python -m src.evaluate --config $CFG   || { echo "CHAIN_FAILED evaluate"; exit 1; }
echo "### ablation sid_set $(date)"
python scripts/data_ablation.py --config $CFG --keep sid_set  || { echo "CHAIN_FAILED ablation"; exit 1; }
echo "### results $(date)"
python scripts/make_results.py || { echo "CHAIN_FAILED make_results"; exit 1; }
echo "CHAIN_DONE $(date)"
