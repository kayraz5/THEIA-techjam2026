#!/usr/bin/env bash
# Reproduces the exact data slices used in this repo (laptop-scale subsets of SID_Set and WildFake).
# Full SID_Set is 140 GB and WildFake 1.29 TB; we pull ~5 GB total. All selections are seeded (seed 0).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
MS="https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo?Revision=master&FilePath="
mkdir -p data/wildfake/{csv,zips,images/Real,images/Diffusion_based} data/sid_set logs

# 1) WildFake label CSVs
for f in dalle2 dalle3 real_coco real_imagenet real_laion5b; do
  [ -f data/wildfake/csv/$f.csv ] || curl -sL -o data/wildfake/csv/$f.csv "${MS}label_csv_files/$f.csv"
done

# 2) Held-out VALIDATION set (organisers' designated subset). Never used for training.
#    4,998 COCO val2017 reals (inside coco.zip) + 8,843 DALL-E Advanced (= dalle3.csv) fakes.
[ -f data/wildfake/zips/coco.zip ] || curl -L -C - -o data/wildfake/zips/coco.zip "${MS}Images/Real/coco.zip"
unzip -qn data/wildfake/zips/coco.zip 'coco/coco2017/val2017/*' -d data/wildfake/images/Real
python scripts/remote_zip.py fetch "${MS}Images/Diffusion_based/DALLE.zip" --grep "DALLE/Advanced/DALLE3/" --out data/wildfake/images/Diffusion_based

# 3) TRAINING slices (seeded random subsets, extracted via HTTP range requests):
python scripts/remote_zip.py fetch "${MS}Images/Diffusion_based/DALLE.zip" --grep "DALLE/Typical/DALLE2/" --limit 2500 --seed 0 --out data/wildfake/images/Diffusion_based
python scripts/remote_zip.py fetch "${MS}Images/Real/laion5b.zip" --grep ".jpg" --limit 3000 --seed 0 --out data/wildfake/images/Real

# 4) ALT-REAL held-out set for the COCO-shortcut check (ImageNet reals; never trained on)
python scripts/remote_zip.py fetch "${MS}Images/Real/imagenet.zip" --grep ".jpg" --limit 2500 --seed 0 --out data/wildfake/images/Real

# 5) SID_Set: first 8 of 249 train shards (~850 imgs each, shuffled & balanced within shard)
python - <<'PY'
from huggingface_hub import hf_hub_download
for i in range(8):
    hf_hub_download('saberzl/SID_Set', f'data/train-{i:05d}-of-00249.parquet', repo_type='dataset', local_dir='data/sid_set')
PY

# 6) Build validation manifest + exclusion hash list (training asserts against this)
python -m src.data.build_val --config configs/frozen_siglip2_giant.yaml
