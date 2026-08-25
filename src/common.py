"""Shared helpers: config, device, preprocessing (squish resize), model construction, metrics."""
from __future__ import annotations
import os, random
import numpy as np, torch, yaml
from PIL import Image
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

def load_config(path: str) -> dict:
    return yaml.safe_load(open(path))

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

def squish_resize(img: Image.Image, size: int) -> Image.Image:
    """Resize to size x size ignoring aspect ratio. NOT a crop (spec §2)."""
    return img.convert("RGB").resize((size, size), Image.BICUBIC)

def to_tensor(img: Image.Image, mean, std) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    m = torch.tensor(mean).view(3, 1, 1); s = torch.tensor(std).view(3, 1, 1)
    return (x - m) / s

def build_backbone(cfg: dict, device):
    from src.models import Backbone
    b = cfg["backbone"]; dtype = DTYPES[b.get("dtype", "float32")]
    if device.type == "cpu": dtype = torch.float32
    try:
        bb = Backbone(b["name"], b.get("image_size"), dtype=dtype)
    except (AssertionError, RuntimeError, MemoryError) as e:
        if not b.get("fallback"): raise
        print(f"\n[backbone] !!! {b['name']} failed ({type(e).__name__}: {str(e)[:120]}). "
              f"FALLING BACK to {b['fallback']} — flag this in any reported numbers !!!\n", flush=True)
        bb = Backbone(b["fallback"], b.get("image_size"), dtype=dtype)
    return bb.to(device).eval()

def metrics(y_true, scores) -> dict:
    y_true = np.asarray(y_true); scores = np.asarray(scores)
    out = {"n": int(len(y_true)), "n_fake": int(y_true.sum()), "n_real": int((1 - y_true).sum())}
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, scores))
        out["err"] = 1 - out["auc"]  # spec §6c
        pred = (scores >= 0.5).astype(int)
        out["acc@0.5"] = float(accuracy_score(y_true, pred))
        out["bal_acc@0.5"] = float(balanced_accuracy_score(y_true, pred))
        out["majority_baseline"] = float(max(y_true.mean(), 1 - y_true.mean()))
    return out
