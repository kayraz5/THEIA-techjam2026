"""Build the held-out validation manifest + exclusion list. Run once after data download.
python -m src.data.build_val --config configs/frozen_siglip2_giant.yaml"""
import argparse, json, yaml
from .registry import wildfake_validation, wildfake_alt_real, summarize
from .exclusion import build_exclusion_list

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
root = cfg["data"]["wildfake"]["root"]
val = wildfake_validation(root)
summarize(val, "validation (organisers' WildFake subset)")
alt = wildfake_alt_real(root, n=cfg["data"]["alt_real"]["n"], seed=cfg["seed"])
summarize(alt, "alt real set (shortcut check)")
n = build_exclusion_list([s.path for s in val + alt], cfg["data"]["validation"]["exclusion_list"])
print(f"[exclusion] wrote {n} content hashes -> {cfg['data']['validation']['exclusion_list']}")
json.dump({"val": [(s.path, s.label, s.source) for s in val], "alt_real": [(s.path, s.label, s.source) for s in alt]},
          open("data/exclusion/val_manifest.json", "w"))
