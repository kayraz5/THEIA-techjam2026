#!/usr/bin/env bash
# Pulls everything the SHIPPED arm (configs/frozen_siglip2_giant_ship.yaml) needs ON TOP of
# scripts/prepare_data.sh, plus the two external evaluation sets behind the headline numbers.
#
#   scripts/prepare_data.sh          # base: SID shards 0-7, LAION reals, DALL-E 2, ImageNet alt-reals,
#                                    #       COCO val2017 + DALL-E 3 validation, exclusion list
#   scripts/prepare_data_ship.sh     # this file
#
# Every selection is seeded (seed 0) and was verified to reproduce the exact file set used for the
# shipped head: the test2017 pick matches file-for-file, GAN_based yields the same 3,000 files, MJv5
# comes from Advanced/part_1.zip. Slices are pulled by HTTP-Range (scripts/remote_zip.py), so no
# 50 GB archive is ever downloaded whole.  Total added: ~13 GB download, ~15 GB on disk.
#
# Requires: bash, curl, python (the venv), network access to modelscope.cn, huggingface.co, zenodo.org.
# Windows: run under WSL or Git Bash.  Re-runnable: every step skips files that already exist.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
MS="https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo?Revision=master&FilePath="
enc(){ python -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$1"; }
IMG=data/wildfake/images
mkdir -p logs data/commforensics data/rrdataset

# ---------------------------------------------------------------- training mix (ship config) --------
# 1) Label CSVs for the extra subsets (registry resolves images against these)
for f in mjv5 DF-GAN styleGAN GALIP GigaGAN starGAN BigGAN; do
  [ -f data/wildfake/csv/$f.csv ] || curl -sL -o data/wildfake/csv/$f.csv "${MS}label_csv_files/$f.csv"
done

# 2) SID_Set: shards 8-23 (prepare_data.sh pulled 0-7). Ship config uses max_train: 8000 drawn from all
#    24 shards present, so the SHARD SET matters for exact reproduction.
python - <<'PY'
from huggingface_hub import hf_hub_download
for i in range(8, 24):
    hf_hub_download('saberzl/SID_Set', f'data/train-{i:05d}-of-00249.parquet', repo_type='dataset', local_dir='data/sid_set')
    print('SHARD_DONE', i, flush=True)
PY

# 3) COCO test2017 reals: 3,000 seeded from the coco.zip already downloaded by prepare_data.sh
#    (~200 px median; puts low resolution on the REAL side of the label — see config header / REPORT 7.8).
#    Same seeded shuffle over central-directory order that remote_zip.py uses -> verified identical 3,000 files.
python - <<'PY'
import zipfile, random, os
z = zipfile.ZipFile('data/wildfake/zips/coco.zip')
ents = [i for i in z.infolist() if not i.filename.endswith('/') and '/test2017/' in i.filename]
random.Random(0).shuffle(ents)
for e in ents[:3000]:
    dst = os.path.join('data/wildfake/images/Real', e.filename)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True); open(dst, 'wb').write(z.read(e))
print('test2017: 3000 files ready')
PY

# 4) Midjourney v5 fakes: 2,500 seeded from Advanced/part_1.zip (7 parts x 50 GB; members have FLAT
#    filenames, so they are written straight into the directory mjv5.csv expects).
python scripts/remote_zip.py fetch "${MS}$(enc Images/Diffusion_based/Midjourney/Advanced/part_1.zip)" \
  --limit 2500 --seed 0 --out "$IMG/Diffusion_based/Midjourney/Advanced/mj_v5"

# 5) Six GAN families: 3,000 seeded across GAN_based.zip (DF-GAN, GALIP, GigaGAN, starGAN, styleGAN, BigGAN).
#    Member paths already carry the GAN_based/... prefix the CSVs use.
python scripts/remote_zip.py fetch "${MS}$(enc Images/GAN_based.zip)" --limit 3000 --seed 0 --out "$IMG"

# ---------------------------------------------------------------- external evaluation sets ----------
# 6) Community Forensics-Eval (OwensLab, CVPR 2025): 30 of 413 parquet shards, spread for generator
#    diversity — shards 0-11 plus every 22nd from 20. Scored by scripts/eval_external.py (default paths).
python - <<'PY'
from huggingface_hub import hf_hub_download
shards = list(range(12)) + list(range(20, 413, 22))
for i in shards:
    hf_hub_download('OwensLab/CommunityForensics-Eval', f'data/CompEval-{i:05d}-of-00413.parquet',
                    repo_type='dataset', local_dir='data/commforensics')
    print('CF_SHARD', i, flush=True)
PY

# 7) RRDataset (arXiv:2509.09172, CC-BY-4.0): the 2.16 GB train_val archive from Zenodo, all 3,000 images.
[ -f data/rrdataset/train_val.tar.gz ] || curl -L -C - -o data/rrdataset/train_val.tar.gz \
  "https://zenodo.org/api/records/14963880/files/RRDataset_original_train_val.tar.gz/content"
[ -d data/rrdataset/images/RRDataset_original_train_val ] || { mkdir -p data/rrdataset/images; tar -xzf data/rrdataset/train_val.tar.gz -C data/rrdataset/images; }

# 8) Rebuild the exclusion list against everything now present (idempotent), then verify.
python -m src.data.build_val --config configs/frozen_siglip2_giant_ship.yaml
python - <<'PY'
import os, glob
chk = {
 "SID shards (24)": len(glob.glob('data/sid_set/data/train-*.parquet')),
 "COCO test2017 (3000)": len(os.listdir('data/wildfake/images/Real/coco/coco2017/test2017')),
 "MJv5 (2500)": len(os.listdir('data/wildfake/images/Diffusion_based/Midjourney/Advanced/mj_v5')),
 "GAN files (3000)": sum(len(f) for _,_,f in os.walk('data/wildfake/images/GAN_based')),
 "CF shards (30)": len(glob.glob('data/commforensics/data/*.parquet')),
 "RRDataset imgs (3000)": sum(len(f) for _,_,f in os.walk('data/rrdataset/images')),
}
for k,v in chk.items(): print(f"[ship-data] {k}: {v}")
PY
echo "SHIP_DATA_DONE"
