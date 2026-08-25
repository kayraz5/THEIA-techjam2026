"""Degradation pipeline shared by training augmentation and the evaluation harness.

Spec §4 table. Pure PIL/numpy, deterministic under a caller-supplied seed.

    from src.degradation import RandomDegradation, apply_transform, EVAL_CONDITIONS
    aug = RandomDegradation(prob=1.0, min_n=1, max_n=3, seed=0)
    img, applied = aug(img)                      # training
    img = apply_transform(img, "jpeg", 30)       # eval condition
    for cond in EVAL_CONDITIONS: ...             # (name, level) pairs for the AUC grid

An optional `backend="official"` in RandomDegradation routes training augmentation through the
NTIRE 2026 official `distort_images` (third_party/aug_utils_train) for comparability; the
official pool has no resize/crop, so the eval grid always uses the table implementation here.
"""
from __future__ import annotations

import io
import random
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# name -> ordered list of severity levels (mild -> harsh). Used for both sampling and the eval grid.
LEVELS: dict[str, list] = {
    "jpeg": [90, 70, 50, 30],          # quality
    "blur": [0.5, 1.0, 2.0],           # gaussian sigma
    "resize": [0.5, 0.25],             # downscale factor, then upscale back to original size
    "noise": [0.02, 0.05, 0.10],       # gaussian sigma in [0,1] pixel units
    "color": [0.2],                    # brightness/contrast/saturation each in [1-a, 1+a]
    "crop": [0.8],                     # centre crop fraction
}
FAMILIES = list(LEVELS.keys())
EVAL_CONDITIONS: list[tuple[str, object]] = [("clean", None)] + [(f, lv) for f in FAMILIES for lv in LEVELS[f]]


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize_down_up(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator | None = None) -> Image.Image:
    rng = rng or np.random.default_rng()
    x = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    x = x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return Image.fromarray((np.clip(x, 0, 1) * 255).round().astype(np.uint8))


def color_jitter(img: Image.Image, amount: float, rng: random.Random | None = None) -> Image.Image:
    """Brightness, contrast, saturation each scaled by an independent factor in [1-amount, 1+amount]."""
    rng = rng or random.Random()
    img = img.convert("RGB")
    for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enh(img).enhance(rng.uniform(1 - amount, 1 + amount))
    return img


def center_crop(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    cw, ch = max(1, round(w * frac)), max(1, round(h * frac))
    l, t = (w - cw) // 2, (h - ch) // 2
    return img.crop((l, t, l + cw, t + ch))


def apply_transform(img: Image.Image, name: str, level, rng_seed: int | None = None) -> Image.Image:
    """Apply one named transform at a level from LEVELS[name]. Stochastic ones (noise, color) take rng_seed."""
    if name == "clean":
        return img
    if name == "jpeg":
        return jpeg_compress(img, level)
    if name == "blur":
        return gaussian_blur(img, level)
    if name == "resize":
        return resize_down_up(img, level)
    if name == "noise":
        return gaussian_noise(img, level, np.random.default_rng(rng_seed))
    if name == "color":
        return color_jitter(img, level, random.Random(rng_seed))
    if name == "crop":
        return center_crop(img, level)
    raise KeyError(name)


class RandomDegradation:
    """Sample 1..3 distinct families, each at an independently sampled level, apply in sequence,
    then horizontal flip with p=hflip. `prob` is the probability of degrading at all (spec: 1.0)."""

    def __init__(self, prob: float = 1.0, min_n: int = 1, max_n: int = 3, hflip: float = 0.5,
                 seed: int | None = 0, families: list[str] | None = None, backend: str = "table"):
        self.prob, self.min_n, self.max_n, self.hflip = prob, min_n, max_n, hflip
        self.families = families or FAMILIES
        self.backend = backend
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        if backend == "official":
            from .official import official_distort  # lazy: needs torch + kornia
            self._official: Callable = official_distort

    def reseed(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def sample_plan(self) -> list[tuple[str, object]]:
        if self.rng.random() >= self.prob:
            return []
        n = self.rng.randint(self.min_n, self.max_n)
        fams = self.rng.sample(self.families, n)
        return [(f, self.rng.choice(LEVELS[f])) for f in fams]

    def __call__(self, img: Image.Image) -> tuple[Image.Image, list[tuple[str, object]]]:
        img = img.convert("RGB")
        if self.backend == "official":
            if self.rng.random() < self.prob:
                img, applied = self._official(img, self.max_n, self.rng)
            else:
                applied = []
        else:
            applied = self.sample_plan()
            for name, level in applied:
                seed = self.rng.getrandbits(32)
                img = apply_transform(img, name, level, rng_seed=seed)
        if self.rng.random() < self.hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img, applied
