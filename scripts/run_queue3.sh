#!/usr/bin/env bash
# Third queue: scaled-data extraction, then the #9 size curve and #10 generator-diversity ablations,
# all from a single GPU pass. Backs up the SID-only 4k caches first (the shipped head depends on them).
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
while pgrep -f "run_queue2.sh" > /dev/null; do sleep 20; done
for f in data/features/*_train_deg_e[01]_*.npz; do cp -n "$f" "$f.sidonly4k_bak"; done
echo "### scaled_train $(date)"
caffeinate -i python -m src.train --config configs/frozen_siglip2_giant_scaled.yaml || { echo "Q3_FAILED train"; exit 1; }
echo "### size_curve (#9) $(date)"
for n in 4000 8000 12000; do
  python scripts/data_ablation.py --config configs/frozen_siglip2_giant_scaled.yaml --keep sid_set --subsample $n --tag "sid_${n}" || echo "Q3_FAILED curve_$n"
done
echo "### generator_diversity (#10) $(date)"
python scripts/data_ablation.py --config configs/frozen_siglip2_giant_scaled.yaml --keep sid_set --keep laion5b --tag "sid_laion" || echo "Q3_FAILED d1"
python scripts/data_ablation.py --config configs/frozen_siglip2_giant_scaled.yaml --keep sid_set --keep laion5b --keep SDXL --tag "sid_laion_sdxl" || echo "Q3_FAILED d2"
python scripts/data_ablation.py --config configs/frozen_siglip2_giant_scaled.yaml --keep sid_set --keep laion5b --keep mj_v5 --tag "sid_laion_mjv5" || echo "Q3_FAILED d3"
python scripts/data_ablation.py --config configs/frozen_siglip2_giant_scaled.yaml --keep sid_set --keep laion5b --keep SDXL --keep mj_v5 --tag "sid_laion_sdxl_mjv5" || echo "Q3_FAILED d4"
echo "### results $(date)"
python scripts/make_results.py || echo "Q3_FAILED make_results"
echo "Q3_DONE $(date)"
