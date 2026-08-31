"""Model lifecycle for the live feed demo: one backbone, N heads, single-frame scoring.

This deliberately reuses the shipped inference path rather than reimplementing it —
`squish_resize` / `to_tensor` from src.common and `load_scorer` from src.evaluate are the
exact functions `predict.py` calls. If this file and `predict.py` ever disagree on a score,
this file is wrong.

Standalone check (loads, warms, self-tests, prints throughput):
    python project_demo/detector.py
"""
from __future__ import annotations
import hashlib, io, json, os, sys, threading, time

import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.common import load_config, get_device, build_backbone, squish_resize, to_tensor  # noqa: E402
from src.evaluate import load_scorer  # noqa: E402

DEFAULT_CONFIG = "configs/frozen_siglip2_giant_ship.yaml"

# name -> checkpoint. Only `ship` and `mjv5` are git-tracked; the rest are skipped if absent.
# Every loaded head scores every frame: feature extraction is the ~85 ms, and each additional
# head is one 1536x2 matmul, so the side-by-side comparison is effectively free.
DEFAULT_HEADS = [
    ("ship",     "results/frozen_siglip2_giant_ship/head_best.pt"),          # active default
    ("mjv5",     "results/frozen_siglip2_giant_mjv5/head_best.pt"),          # benchmark-optimal arm
    ("gan_only", "results/frozen_siglip2_giant_2x2/head_ablation_C_plus_gan.pt"),  # REPORT 7.8 arm C
    ("sidonly",  "results/frozen_siglip2_giant_sidonly/head_best.pt"),       # REPORT 7.7 collapse arm
]

# Startup credibility anchor: a file whose score is committed in demo_preds.json.
SELFTEST_IMAGE = "demo_images/ai_0_clean.jpg"
SELFTEST_EXPECTED = 0.99720299243927
SELFTEST_TOL = 1e-3


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Detector:
    def __init__(self, config_path: str = DEFAULT_CONFIG, head_specs=None, self_test: bool = True):
        head_specs = head_specs or DEFAULT_HEADS
        self.config_path = config_path
        self.cfg = load_config(os.path.join(REPO, config_path))
        self.device = get_device()
        t0 = time.time()
        self.backbone = build_backbone(self.cfg, self.device)   # prints the [param-check] line
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.load_s = time.time() - t0
        self.info = self.backbone.info
        self.dtype = next(self.backbone.parameters()).dtype     # bfloat16 on mps

        self.heads = {}   # name -> dict(score=fn, ckpt=str, sha256=str, bytes=int)
        for name, rel in head_specs:
            path = os.path.join(REPO, rel)
            if not os.path.exists(path):
                print(f"[heads] SKIP {name}: {rel} not found", flush=True)
                continue
            self.heads[name] = {
                "score": load_scorer(self.cfg, self.backbone, self.device, path),
                "ckpt": rel,
                "sha256": _sha256(path),
                "bytes": os.path.getsize(path),
            }
            print(f"[heads] {name:9s} {rel}", flush=True)
        assert self.heads, "no head checkpoints found — nothing to score with"
        self.active = "ship" if "ship" in self.heads else next(iter(self.heads))

        self.lock = threading.Lock()   # serialises MPS; the backbone is not re-entrant
        self.warm_ms = self._warmup()
        self.selftest = self._selftest() if self_test else {"status": "disabled"}
        self.ready = True

    # ---- inference -------------------------------------------------------

    def _forward(self, img: Image.Image) -> dict:
        """One PIL image -> {head_name: p(AI)}. The single source of truth for scoring."""
        size = self.info.image_size
        # squish_resize is a full-frame resize to 384x384, NOT a centre crop, and mean=std=0.5
        # rather than ImageNet stats. Features are fed to the head raw — no normalisation
        # (results/frozen_siglip2_giant_sidonly/sweep_norm.csv found `none` beat standardize and l2).
        x = to_tensor(squish_resize(img, size), self.info.mean, self.info.std)[None]
        with self.lock, torch.inference_mode():
            # .float() is REQUIRED: the backbone emits bf16, heads are fp32, and MPS
            # hard-asserts on a mixed-dtype matmul (REPORT engineering notes).
            feats = self.backbone(x.to(self.device, self.dtype)).float().cpu().numpy()
            return {n: float(min(max(h["score"](feats)[0], 0.0), 1.0)) for n, h in self.heads.items()}

    def score_jpeg_bytes(self, data: bytes) -> dict:
        img = Image.open(io.BytesIO(data))
        img.load()
        return self._forward(img)

    def score_path(self, path: str) -> dict:
        img = Image.open(path)
        img.load()
        return self._forward(img)

    def set_active(self, name: str) -> str:
        if name not in self.heads:
            raise KeyError(name)
        self.active = name
        return self.active

    # ---- startup ---------------------------------------------------------

    def _warmup(self) -> float:
        """Three forwards on zeros; report the steady-state warm latency, not the first."""
        size = self.info.image_size
        z = torch.zeros(1, 3, size, size)
        ms = []
        for _ in range(3):
            t = time.time()
            with self.lock, torch.inference_mode():
                self.backbone(z.to(self.device, self.dtype)).float().cpu()
            ms.append((time.time() - t) * 1000)
        print(f"[warmup] first {ms[0]:.0f} ms, warm {ms[-1]:.0f} ms", flush=True)
        return round(ms[-1], 1)

    def _selftest(self) -> dict:
        """Reproduce a value committed in demo_preds.json, proving this is the shipped path.

        demo_images/ is gitignored, so a fresh clone must skip rather than crash.
        """
        path = os.path.join(REPO, SELFTEST_IMAGE)
        if not os.path.exists(path):
            return {"status": "skipped", "file": SELFTEST_IMAGE,
                    "hint": "run `bash scripts/demo.sh` to build demo_images/"}
        got = self.score_path(path)[self.active]
        ok = abs(got - SELFTEST_EXPECTED) <= SELFTEST_TOL
        print(f"[self-test] {SELFTEST_IMAGE} -> {got:.6f} (expected {SELFTEST_EXPECTED:.6f}) "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        return {"status": "pass" if ok else "fail", "file": SELFTEST_IMAGE,
                "got": round(got, 6), "expected": SELFTEST_EXPECTED, "tol": SELFTEST_TOL,
                "head": self.active}

    # ---- reporting -------------------------------------------------------

    def health(self) -> dict:
        return {
            "ready": True,
            "device": self.device.type,
            "dtype": str(self.dtype).replace("torch.", ""),
            "backbone": self.info.name,
            "n_params": int(self.info.n_params),
            "feature_dim": int(self.info.hidden_size),
            "image_size": int(self.info.image_size),
            "mean": list(self.info.mean),
            "std": list(self.info.std),
            "n_prefix_tokens": int(self.info.n_prefix_tokens),
            "config": self.config_path,
            "active_head": self.active,
            "heads": {n: {"ckpt": h["ckpt"], "sha256": h["sha256"], "bytes": h["bytes"]}
                      for n, h in self.heads.items()},
            "selftest": self.selftest,
            "warm_ms": self.warm_ms,
            "load_s": round(self.load_s, 1),
        }


if __name__ == "__main__":
    d = Detector()
    print(json.dumps(d.health(), indent=2))
    target = os.path.join(REPO, SELFTEST_IMAGE)
    if os.path.exists(target):
        t0 = time.time()
        n = 10
        for _ in range(n):
            out = d.score_path(target)
        dt = time.time() - t0
        print(f"\n[bench] {n} single-image scores: {1000*dt/n:.0f} ms each, {n/dt:.1f} img/s")
        print(f"[bench] all heads: " + "  ".join(f"{k}={v:.4f}" for k, v in out.items()))
