#!/usr/bin/env bash
# Autonomous queue: waits for the clean-draw test, then TTA (#11), then the CLIP baseline (#12).
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
while pgrep -f "clean_draw_test.py" > /dev/null; do sleep 15; done
echo "### tta $(date)"
caffeinate -i python scripts/tta_test.py --config configs/frozen_siglip2_giant_sidonly.yaml || echo "QUEUE_FAILED tta"
echo "### clip_train $(date)"
caffeinate -i python -m src.train --config configs/baseline_clip_vitl14_sidonly.yaml || echo "QUEUE_FAILED clip_train"
echo "### clip_eval $(date)"
caffeinate -i python -m src.evaluate --config configs/baseline_clip_vitl14_sidonly.yaml || echo "QUEUE_FAILED clip_eval"
echo "### results $(date)"
python scripts/make_results.py || echo "QUEUE_FAILED make_results"
echo "QUEUE_DONE $(date)"
