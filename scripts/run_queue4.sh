#!/usr/bin/env bash
# Fourth queue: issue #13 — 512px so400m arm (resolution vs parameter-count trade).
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
while pgrep -f "run_queue3.sh" > /dev/null; do sleep 20; done
echo "### so400m512_train $(date)"
caffeinate -i python -m src.train --config configs/frozen_siglip2_so400m_512.yaml || { echo "Q4_FAILED train"; exit 1; }
echo "### so400m512_eval $(date)"
caffeinate -i python -m src.evaluate --config configs/frozen_siglip2_so400m_512.yaml || { echo "Q4_FAILED eval"; exit 1; }
echo "### results $(date)"
python scripts/make_results.py || echo "Q4_FAILED make_results"
echo "Q4_DONE $(date)"
