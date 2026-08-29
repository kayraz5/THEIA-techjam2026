#!/usr/bin/env bash
# 2x2 design: #21 (real-source screening) x #23 (GAN resolution confound), one extraction.
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
CFG=configs/frozen_siglip2_giant_2x2.yaml
echo "### train_2x2 $(date)"
caffeinate -i python -m src.train --config $CFG || { echo "Q7_FAILED train"; exit 1; }
echo "### ablations $(date)"
# 2x2: {ship} x {+coco test2017 low-res reals} x {+GAN low-res fakes}
python scripts/data_ablation.py --config $CFG --keep sid_set --keep laion5b --keep mj_v5 --tag A_ship --save_head
python scripts/data_ablation.py --config $CFG --keep sid_set --keep laion5b --keep mj_v5 --keep test2017 --tag B_plus_cocoreals --save_head
python scripts/data_ablation.py --config $CFG --keep sid_set --keep laion5b --keep mj_v5 --keep GAN_based --tag C_plus_gan --save_head
python scripts/data_ablation.py --config $CFG --keep sid_set --keep laion5b --keep mj_v5 --keep test2017 --keep GAN_based --tag D_plus_both --save_head
echo "### external_evals $(date)"
for t in A_ship B_plus_cocoreals C_plus_gan D_plus_both; do
  echo "--- external: $t"
  caffeinate -i python scripts/eval_external.py --config $CFG --name "cf_$t" \
    --checkpoint results/frozen_siglip2_giant_2x2/head_ablation_$t.pt || echo "Q7_FAILED external_$t"
done
echo "Q7_DONE $(date)"
