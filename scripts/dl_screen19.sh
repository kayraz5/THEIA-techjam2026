#!/bin/bash
# #19: per-source screening panel. Pull 1500 each from sources never used in training,
# spanning generator FAMILIES (VQ diffusion, masked-generative, VQ autoencoder, commercial
# diffusion, personalized/adapter SD) and REAL domains (faces, animals, scenes) — the
# untested half of #21. Small slices; HTTP-Range so no archive is downloaded whole.
cd "$(dirname "$0")/.."
source .venv/bin/activate
U="https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo?Revision=master&FilePath="
enc(){ python -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$1"; }
N=1500

pull(){ # archive  outdir  [grep]
  local zip="$1" out="$2" grep="$3"
  echo "### $zip $(date)"
  if [ -n "$grep" ]; then
    python scripts/remote_zip.py fetch "${U}$(enc "$zip")" --out "$out" --limit $N --seed 0 --workers 8 --grep "$grep"
  else
    python scripts/remote_zip.py fetch "${U}$(enc "$zip")" --out "$out" --limit $N --seed 0 --workers 8
  fi
}

# fakes: different generator families
pull "Images/Diffusion_based/VQDM.zip"   data/wildfake/images/Diffusion_based
pull "Images/Diffusion_based/Imagen.zip" data/wildfake/images/Diffusion_based
pull "Images/Other_based.zip"            data/wildfake/images

# reals: different domains (faces / animals / scenes)
for r in ffhq celebahq afhq church; do
  pull "Images/Real/${r}.zip" data/wildfake/images/Real
done
echo "SCREEN19_DL_DONE $(date)"
