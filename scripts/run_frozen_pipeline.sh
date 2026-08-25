#!/usr/bin/env bash
# Step 2-6 of the spec for the frozen arm. Waits for downloads, then build_val -> train -> evaluate.
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
CFG=${1:-configs/frozen_siglip2_giant.yaml}
until grep -q wrote logs/dl_laion.log; do sleep 15; done
echo "### build_val $(date)"; python -m src.data.build_val --config "$CFG" || { echo PIPELINE_FAILED build_val; exit 1; }
echo "### train $(date)";     python -m src.train --config "$CFG"          || { echo PIPELINE_FAILED train; exit 1; }
echo "### evaluate $(date)";  python -m src.evaluate --config "$CFG"       || { echo PIPELINE_FAILED evaluate; exit 1; }
echo "PIPELINE_DONE $(date)"
