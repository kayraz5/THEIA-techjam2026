#!/usr/bin/env bash
# Issue #17 step 1: full SID-only arm (WildFake dropped from TRAINING only) + benchmark grid.
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
CFG=configs/frozen_siglip2_giant_sidonly.yaml
echo "### train $(date)"
caffeinate -i python -m src.train --config $CFG    || { echo "CHAIN_FAILED train"; exit 1; }
echo "### evaluate $(date)"
caffeinate -i python -m src.evaluate --config $CFG || { echo "CHAIN_FAILED evaluate"; exit 1; }
echo "### results $(date)"
python scripts/make_results.py                     || { echo "CHAIN_FAILED make_results"; exit 1; }
echo "CHAIN_DONE $(date)"
