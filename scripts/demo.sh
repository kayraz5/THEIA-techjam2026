#!/bin/bash
# End-to-end demo for the video: build a small mixed folder (real + AI, clean + degraded), run the
# deliverable CLI on it, and print a readable scoreboard. Uses only data already on disk.
#
#   bash scripts/demo.sh            # ~1 min after the model is cached
#
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -e
DEMO=demo_images; OUT=demo_preds.json
rm -rf "$DEMO" "$OUT"; mkdir -p "$DEMO"

python - <<'PY'
# 6 real COCO photos + 6 DALL-E 3 images from the held-out validation set (never trained on),
# each saved clean AND under one random harsh degradation, so the video shows both.
import random, os, glob, shutil
from PIL import Image
from src.degradation import apply_transform
rng = random.Random(7)
# Take BOTH classes from the organisers' designated validation subset via its label CSVs, exactly as
# src/data/registry.py defines it (COCO val2017 reals; DALL-E 3 "Advanced" fakes). Never globbed from
# disk: an earlier version of this script fell through to a DALL-E 2 training folder by accident.
import pandas as pd
def csv_paths(name, must_contain):
    df = pd.read_csv(f"data/wildfake/csv/{name}.csv")
    df = df[df.Image_path.str.contains(must_contain, regex=False)]
    ps = ["data/wildfake/images/" + q.lstrip("./") for q in df.Image_path]
    return sorted(q for q in ps if os.path.exists(q))
reals = csv_paths("real_coco", "/val2017/")
fakes = csv_paths("dalle3", "/Advanced/")
assert reals and fakes, (len(reals), len(fakes))
# every demo image must be on the exclusion list, i.e. provably never trained on
from src.data.exclusion import content_hash
excl = set(l.split()[0] for l in open("data/exclusion/wildfake_val_hashes.txt") if l.strip())
harsh = [("jpeg", 30), ("blur", 2.0), ("resize", 0.25), ("noise", 0.1), ("crop", 0.8), ("color", 0.2)]
for tag, pool in (("real", rng.sample(reals, 6)), ("ai", rng.sample(fakes, 6))):
    for i, p in enumerate(pool):
        img = Image.open(p).convert("RGB")
        assert content_hash(Image.open(p)) in excl, f"{p} is not on the held-out exclusion list"
        img.save(f"demo_images/{tag}_{i}_clean.jpg", quality=95)
        t, lv = harsh[i]
        apply_transform(img, t, lv, rng_seed=i).save(f"demo_images/{tag}_{i}_{t}{lv}.jpg", quality=95)
print("built demo_images/ :", len(os.listdir("demo_images")), "files - all 12 sources verified on the held-out exclusion list; label is only in the filename")
PY

echo; echo "=============== predict.py ==============="
python predict.py --image_dir "$DEMO" --output "$OUT"

echo; echo "=============== scoreboard ==============="
python - <<'PY'
import json, os
rows = json.load(open("demo_preds.json"))
print(f"{'image':<32} {'p(AI)':>7}  verdict   truth")
print("-"*64)
ok = 0
for r in sorted(rows, key=lambda r: r["image_path"]):
    name = os.path.basename(r["image_path"]); p = r["pred"]
    truth = "AI" if name.startswith("ai_") else "real"
    verdict = "AI" if p >= 0.5 else "real"
    ok += verdict == truth
    print(f"{name:<32} {p:>7.3f}  {verdict:<8}  {truth}{'' if verdict==truth else '   <-- miss'}")
print("-"*64); print(f"{ok}/{len(rows)} correct at threshold 0.5")
PY
