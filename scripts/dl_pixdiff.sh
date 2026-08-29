#!/bin/bash
# #22: pixel-space diffusion training data (no VAE -> no VAE fingerprint; our worst external cell).
set -x
U="https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo?Revision=master&FilePath="
enc(){ python -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$1"; }
OUT=data/wildfake/images/Diffusion_based
for z in ADM DDIM DDPM; do
  echo "### $z $(date)"
  python scripts/remote_zip.py fetch "${U}$(enc "Images/Diffusion_based/${z}.zip")" \
    --out "$OUT" --limit 1000 --seed 0 --workers 8
done
echo "PIXDIFF_DONE $(date)"
