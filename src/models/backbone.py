"""Backbone wrapper: any HF vision tower -> mean-pooled final-layer patch features.

Design (spec §2): squish-resize input, final hidden layer, global average pool over
ALL patch tokens (no CLS, no attention pooling), single linear head.

Nothing here is SigLIP-specific except the family detection table below, so the same
train/eval code runs with DINOv3 or CLIP by changing `backbone.name` in the config.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel

log = logging.getLogger(__name__)

PARAM_LIMIT = 2_000_000_000  # hard constraint from spec §1


@dataclass
class BackboneInfo:
    name: str
    family: str
    n_params: int
    hidden_size: int
    image_size: int
    mean: tuple
    std: tuple
    n_prefix_tokens: int  # non-patch tokens at the start of the sequence (CLS/registers)


def _family(name: str) -> str:
    n = name.lower()
    if "siglip" in n:
        return "siglip"
    if "dinov3" in n or "dinov2" in n:
        return "dino"
    if "clip" in n:
        return "clip"
    raise ValueError(f"Unknown backbone family for {name}; add it to _family()")


def count_params(m: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))


def assert_under_limit(n_params: int, what: str) -> None:
    msg = f"[param-check] {what}: {n_params/1e9:.3f}B parameters (limit {PARAM_LIMIT/1e9:.0f}B)"
    print(msg, flush=True)
    log.info(msg)
    assert n_params < PARAM_LIMIT, f"{what} has {n_params} params, exceeds {PARAM_LIMIT}"


class Backbone(nn.Module):
    def __init__(self, name: str, image_size: int | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.name = name
        self.family = _family(name)
        cfg = AutoConfig.from_pretrained(name)
        vcfg = getattr(cfg, "vision_config", cfg)

        if self.family == "siglip":
            from transformers import SiglipVisionModel
            self.model = SiglipVisionModel.from_pretrained(name, dtype=dtype)
            n_prefix = 0  # SigLIP has no CLS token; every token is a patch
        elif self.family == "clip":
            from transformers import CLIPVisionModel
            self.model = CLIPVisionModel.from_pretrained(name, dtype=dtype)
            n_prefix = 1  # CLS
        else:  # dino
            self.model = AutoModel.from_pretrained(name, dtype=dtype)
            n_prefix = 1 + int(getattr(vcfg, "num_register_tokens", 0))

        proc = AutoImageProcessor.from_pretrained(name)
        mean = tuple(float(x) for x in proc.image_mean)
        std = tuple(float(x) for x in proc.image_std)
        default_size = getattr(vcfg, "image_size", None) or 384
        self.info = BackboneInfo(
            name=name,
            family=self.family,
            n_params=count_params(self.model),
            hidden_size=vcfg.hidden_size,
            image_size=int(image_size or default_size),
            mean=mean,
            std=std,
            n_prefix_tokens=n_prefix,
        )
        assert_under_limit(self.info.n_params, f"backbone vision tower {name}")
        if self.info.image_size != default_size:
            print(f"[backbone] NOTE: running {name} at {self.info.image_size}px (native {default_size}px)")

    @property
    def feature_dim(self) -> int:
        return self.info.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B,3,H,W) normalised. Returns (B, hidden) mean-pooled patch tokens."""
        kwargs = {}
        if self.family == "dino" or (self.family == "clip" and pixel_values.shape[-1] != 224):
            kwargs["interpolate_pos_encoding"] = True
        out = self.model(pixel_values=pixel_values, **kwargs)
        tokens = out.last_hidden_state  # (B, N, C) — final hidden layer, post-norm
        patches = tokens[:, self.info.n_prefix_tokens :, :]
        return patches.mean(dim=1)


class LinearHead(nn.Module):
    """Single linear layer -> 2 logits (real=0 / fake=1). Not an MLP (spec §2)."""

    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats)


class Detector(nn.Module):
    def __init__(self, backbone: Backbone, head: LinearHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(pixel_values))

    @torch.no_grad()
    def predict_proba(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(pixel_values).float(), dim=-1)[:, 1]


def apply_lora(backbone: Backbone, rank: int = 32, alpha: int | None = None, dropout: float = 0.05) -> Backbone:
    """LoRA on attention (q,k,v,out) and MLP (fc1,fc2) projections. Head is trained separately."""
    from peft import LoraConfig, get_peft_model

    fam = backbone.family
    if fam == "siglip" or fam == "clip":
        targets = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    else:  # dino
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "query", "key", "value", "dense", "fc1", "fc2"]
    cfg = LoraConfig(r=rank, lora_alpha=alpha or 2 * rank, lora_dropout=dropout, target_modules=targets, bias="none")
    backbone.model = get_peft_model(backbone.model, cfg)
    n_train = count_params(backbone.model, trainable_only=True)
    print(f"[lora] rank={rank} trainable params in backbone: {n_train/1e6:.1f}M")
    return backbone


def total_param_report(det: Detector) -> None:
    n = count_params(det)
    assert_under_limit(n, "full detector (backbone + adapters + head)")
