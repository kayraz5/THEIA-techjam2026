# CLAUDE.md — AIGC Image Detection Benchmark (Hackathon Final Round Prep)

## Project Context
We're building a prototype to detect AI-generated images with robustness to real-world post-processing transformations, for a hackathon final round. This phase is specifically about **benchmarking candidate models** on the provided datasets before committing to an architecture.

## Problem Statement (from brief)
Build a system that distinguishes AI-generated images from authentic images, robust to: JPEG compression, Gaussian blur, resize (down/upscale), Gaussian noise, color jitter, and center crop. Must maintain accuracy on both clean and transformed images, not just clean data.

## Hard Constraints
- **Model size: <2B parameters.** This is a strict rule — any candidate model must be verified under this cap before benchmarking.
- Hackathon-scale compute. No production infra, no internal systems.
- Public/properly licensed datasets only.

## Datasets
Training/dev candidates (do NOT use for the validation benchmark below):
- SID_Set — https://huggingface.co/datasets/saberzl/SID_Set
- CIFAKE — https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- WildFake — https://modelscope.cn/datasets/hy2628982280/WildFake/summary (note: requires translation via UI button before use, may need manual handling)

**Validation/demo benchmark set** (reference only, not for training, not for final scoring):
- Non-AIGC: COCO val2017 — 4,998 images
- AIGC: DALL·E Advanced — 8,843 images

This mismatch (train sources vs. validation sources) is intentional in the brief — treat it as a generalization stress test, not noise to eliminate.

## Current Goal: Model Benchmarking
Before building a custom architecture, we want to benchmark existing/candidate models (open-source AIGC detectors, general-purpose classifiers, or CNN/ViT backbones under 2B params) on:
1. **Clean accuracy** — baseline performance on unmodified images from the validation set.
2. **Robustness under transformation** — same images run through each transform type at each parameter level:

| Transform | Parameters |
|---|---|
| JPEG Compression | quality = 90, 70, 50, 30 |
| Gaussian Blur | σ = 0.5, 1.0, 2.0 |
| Resize | 0.5×, 0.25× then upscale back |
| Gaussian Noise | σ = 0.02, 0.05, 0.10 |
| Color Jitter | brightness/contrast/saturation ±20% |
| Center Crop | crop 80% |

Report per-model: AUC/ROC (preferred over raw accuracy), plus accuracy, at each clean/transform condition. We need this in a table (clean vs. each transform/severity) — this table will become part of the Robustness Evaluation Summary deliverable later.

## What "Benchmarking" Means Here (Scope for This Session)
- Do NOT fine-tune yet. This pass is about **zero-shot / off-the-shelf performance** of candidate models to decide which to build on.
- Build a reusable eval harness now (this will be reused for every future iteration, so make it modular):
  - Load validation set (COCO val2017 subset + DALL·E Advanced subset).
  - Apply transform matrix, generate transformed copies (cache them, don't regenerate every run).
  - Run each candidate model, collect predictions (confidence score per image).
  - Score: AUC, accuracy, FPR/FNR at a fixed threshold (and ideally an ROC curve).
  - Output: a results table/CSV + simple visualization (heatmap of AUC by transform × severity, per model).
- Track parameter count for each candidate model explicitly — hard filter out anything ≥2B before it goes into the leaderboard.

## Candidate Models to Consider Benchmarking
(Confirm current availability/license before use — do not assume from memory, check HF model cards)
- CNNDetection-style detector (Wang et al., "CNN-generated images are surprisingly easy to spot")
- Small ViT/DeiT-based classifiers fine-tuned for AIGC detection (search HF hub for "ai-generated-image-detection")
- Frequency-domain baselines (simple DCT/FFT + lightweight classifier) as a robustness reference point
- General-purpose small vision backbones (EfficientNet, ResNet50) as fine-tune starting points, benchmarked zero-shot first if pretrained AIGC weights exist

## Output We Want From This Phase
1. A leaderboard: model × (clean AUC, avg AUC under transform, worst-case AUC under transform, param count).
2. A robustness heatmap per top 2-3 candidates: transform type × severity → AUC drop from clean.
3. A short written recommendation: which model(s) to carry forward into fine-tuning, with reasoning tied to the robustness numbers (not just clean accuracy).
4. Reusable code (harness + transform pipeline + scoring) that we will extend, not throw away.

## Non-Goals for This Phase
- No final architecture decisions yet.
- No training/fine-tuning yet.
- No demo video, no write-up polishing — this is internal benchmarking only.
- Don't over-invest in the interactive Gradio demo yet; that comes after we pick a model.

## Style/Working Notes
- Prefer clarity and reproducibility over cleverness — this harness needs to be reused many times as we iterate.
- Cache transformed images and predictions where possible; re-running the full matrix from scratch every time will be slow.
- Log everything (model name, transform config, scores) to a single results file/table rather than scattered outputs.