#!/usr/bin/env bash
# Second autonomous queue: external harder eval (#15) then the LoRA throughput benchmark (#1).
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
while pgrep -f "run_queue.sh" > /dev/null; do sleep 20; done
echo "### external_eval $(date)"
caffeinate -i python scripts/eval_external.py --config configs/frozen_siglip2_giant_sidonly.yaml \
   --checkpoint results/frozen_siglip2_giant_sidonly/head_best.pt || echo "Q2_FAILED external"
echo "### lora_bench $(date)"
caffeinate -i python scripts/lora_bench.py --config configs/lora_siglip2_giant_sidonly.yaml --steps 12 || echo "Q2_FAILED lora_bench"
echo "Q2_DONE $(date)"
