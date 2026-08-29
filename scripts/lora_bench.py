"""Issue #1 step 1: measure actual LoRA training throughput on this hardware before scheduling a long run.

Builds the real LoRA model (rank from config, adapters on attention + MLP projections) and times
forward+backward on real batches from the configured training set. Reports measured img/s and
extrapolates epoch and full-run wall-clock, replacing the issue's rule-of-thumb estimate.

  python scripts/lora_bench.py --config configs/lora_siglip2_giant_sidonly.yaml --steps 12
"""
from __future__ import annotations
import argparse, os, sys, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import load_config, set_seed, get_device, build_backbone
from src.data import build_train
from src.degradation import RandomDegradation
from src.features import ImageDS
from src.models import LinearHead, Detector, apply_lora, count_params, total_param_report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=12); a = ap.parse_args()
    cfg = load_config(a.config); set_seed(cfg["seed"]); device = get_device()
    tc, dc, lc = cfg["train"], cfg["degradation"], cfg["lora"]

    backbone = build_backbone(cfg, device)
    n_frozen = count_params(backbone.model)
    backbone = apply_lora(backbone, lc["rank"], lc["alpha"], lc["dropout"]).train()
    head = LinearHead(backbone.feature_dim).to(device)
    det = Detector(backbone, head).to(device); total_param_report(det)
    trainable = count_params(det, trainable_only=True)
    print(f"[lora-bench] backbone {n_frozen/1e9:.3f}B frozen params, {trainable/1e6:.1f}M trainable "
          f"({100*trainable/n_frozen:.2f}% of backbone)")

    train_s = build_train(cfg)
    i = backbone.info
    aug = RandomDegradation(dc["prob"], dc["min_transforms"], dc["max_transforms"], dc["hflip"], dc["seed"])
    ds = ImageDS(train_s, i.image_size, i.mean, i.std, degrade=aug, seed=cfg["seed"])
    dl = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, num_workers=2, drop_last=True)
    params = [p for p in det.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    amp = tc["amp"] and device.type in ("cuda", "mps")

    times, n_img = [], 0
    for step, (x, y, _, _) in enumerate(dl):
        if step >= a.steps + 2: break
        t0 = time.time()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss = loss_fn(det(x.to(device)), y.to(device))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if device.type == "mps": torch.mps.synchronize()
        dt = time.time() - t0
        if step >= 2:                      # discard warm-up steps
            times.append(dt); n_img += len(y)
        print(f"  step {step:3d}  {dt:6.2f}s  loss {loss.item():.4f}", flush=True)

    per_step = sum(times) / len(times); ips = tc["batch_size"] / per_step
    n_train = len(train_s)
    epoch_s = n_train / ips
    print(f"\n[lora-bench] batch_size={tc['batch_size']} rank={lc['rank']} amp={amp}")
    print(f"[lora-bench] MEASURED {ips:.2f} img/s  ({per_step:.2f} s/step over {len(times)} steps)")
    print(f"[lora-bench] {n_train} training images -> {epoch_s/60:.1f} min/epoch")
    for ep in (5, 30):
        print(f"[lora-bench]   {ep:2d} epochs = {ep*epoch_s/3600:.1f} h training "
              f"(+ per-epoch validation and the eval grid on top)")


if __name__ == "__main__":
    main()
