"""Thin wrapper around the NTIRE 2026 official `distort_images` pipeline (third_party/aug_utils_train).
Official pool: gaussian/lens blur, colour shift/saturation, JPEG, white/impulse noise, brighten/darken,
jitter, quantisation, linear contrast. Note it does NOT include resize or crop."""
from __future__ import annotations
import random, sys, os
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "third_party"))
from aug_utils_train.utils_data import distort_images, distortion_functions  # noqa: E402

_NAME = {v: k for k, v in distortion_functions.items()}

def official_distort(img: Image.Image, max_n: int, rng: random.Random):
    random.seed(rng.getrandbits(32)); np.random.seed(rng.getrandbits(32))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    y, fns, vals = distort_images(x, max_distortions=max_n)
    out = Image.fromarray((y.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    return out, [(_NAME[f], float(v)) for f, v in zip(fns, vals)]
