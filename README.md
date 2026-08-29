# Robust AI-Generated Image Detector

Prototype for a hackathon track on detecting AI-generated images **under real-world transformations**
(JPEG, blur, resize, noise, colour shift, crop). Given an image it outputs `p(AI-generated) ∈ [0,1]`.

The engine reproduces the 5th-place NTIRE 2026 *Robust AI-Generated Image Detection* recipe exactly:

```
image -> squish-resize 384x384 (no crop, aspect ignored)
      -> SigLIP2 vision tower (google/siglip2-giant-opt-patch16-384, 1.164 B params, text tower discarded)
      -> final-layer patch tokens -> global average pool (no CLS, no attention pooling)
      -> one linear layer -> 2 logits
```

Every model is parameter-counted at startup and asserted `< 2 B` ([backbone.py](src/models/backbone.py)).
Backbone is swappable by config (SigLIP2 / DINOv3 / CLIP share one code path).

**Headline metric is ROC AUC**, reported for clean images and per transform/severity. Accuracy is never
reported without balanced accuracy and the majority-class baseline, because the validation split is 1 : 1.77.

> **RESULTS: see [results/RESULTS.md](results/RESULTS.md)** (auto-filled from `results/<arm>/eval/summary.json`).

### Ship candidate

`configs/frozen_siglip2_giant_ship.yaml` -> `results/frozen_siglip2_giant_ship/head_best.pt` (16 KB).

| | |
|---|---|
| Designated benchmark, clean AUC (13,841 imgs) | **0.9991** (dedup 0.9991) |
| Worst single transform | noise@0.1 **0.9951** (max delta 0.0039) |
| Accuracy @ 0.5 / balanced acc / majority baseline | 0.9850 / 0.9856 / 0.6545 |
| Shortcut gap (COCO reals vs ImageNet reals) | +0.0011 |
| **External generalization** (Community Forensics-Eval, 3,759 imgs, unseen) | **0.9917** |

Selected on **external** generalization, not the designated benchmark - the benchmark is saturated
(0.9994 is reachable, leaving ~0.0006 of headroom, below the noise floor for 4,998 real images) and can
no longer rank arms. The previous candidate (`frozen_siglip2_giant_mjv5`, 0.9994 designated) scores
**0.9715** on the same external set under a like-for-like comparison. Full reasoning:
[docs/REPORT.md](docs/REPORT.md), section 7.4.

---

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
scripts/prepare_data.sh          # ~5 GB: seeded slices of SID_Set + WildFake, builds the exclusion list
```

Hardware used: Apple M5 Pro, 64 GB unified memory (MPS, bf16). SigLIP2-giant runs at ~5.4 img/s here;
this throughput dictated every subsampling decision below. On an 80 GB GPU nothing needs subsampling —
set `max_train: 0`, `n_eval_max: 0`.

## Reproduce every number

| Step | Command | Output |
|---|---|---|
| Param check + backbone load | `python -c "from src.common import *; build_backbone(load_config('configs/frozen_siglip2_giant.yaml'), get_device())"` | prints `[param-check] ... 1.164B` |
| Degradation unit tests | `pytest tests/test_degradation.py` | 8 tests |
| Pipeline smoke test (stub backbone) | `pytest tests/test_pipeline_smoke.py` | train→eval→predict in ~2 min |
| Validation set + exclusion list | `python -m src.data.build_val --config configs/frozen_siglip2_giant.yaml` | `data/exclusion/wildfake_val_hashes.txt` |
| Frozen baseline (arm 1) | `python -m src.train --config configs/frozen_siglip2_giant.yaml` | `results/frozen_siglip2_giant/head_best.pt`, `train_summary.json` |
| Full eval grid | `python -m src.evaluate --config configs/frozen_siglip2_giant.yaml` | `results/frozen_siglip2_giant/eval/{auc_grid.csv,auc_grid.png,thresholds.csv,roc.png,fpr_vs_threshold.png,errors_fp.png,errors_fn.png,summary.json}` |
| LoRA arm | `python -m src.train --config configs/lora_siglip2_giant.yaml && python -m src.evaluate --config configs/lora_siglip2_giant.yaml` | `results/lora_siglip2_giant/` |
| CLIP ViT-L/14 linear-probe baseline | same two commands with `configs/baseline_clip_vitl14.yaml` | `results/baseline_clip_vitl14/` |
| DINOv3 comparison | `configs/frozen_dinov3_large.yaml` (gated HF repo: `export HF_TOKEN=...` after requesting access) | `results/frozen_dinov3_large/` |
| Comparison table | `python -m src.compare results/baseline_clip_vitl14 results/frozen_siglip2_giant results/lora_siglip2_giant` | `results/comparison.csv` |
| Deliverable | `python predict.py --image_dir <dir> --output preds.json` | JSON `[{"image_path", "pred"}]` |

All randomness is seeded (`seed: 0` in configs; degradation draws are seeded per (epoch, index)).

## Data

| Source | Role | What we use | Why |
|---|---|---|---|
| [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) | train | 4,000 imgs from the first 8/249 train shards (label 0 real / 1 full-synthetic; label 2 *tampered* excluded) | full set is 140 GB |
| [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake) | train | 3,000 LAION-5B reals + 2,500 DALL-E 2 fakes, seeded random slices pulled by HTTP-range from the zips ([remote_zip.py](scripts/remote_zip.py)) | full set is 1.29 TB; SDXL/MJ-v5 sit in 50 GB archives and were not pulled |
| WildFake **validation** | held-out | **4,998 COCO val2017 reals + 8,843 DALL-E Advanced (dalle3) fakes** — the organisers' designated subset | never trained on |
| WildFake ImageNet reals | held-out **alt-real** | 1,000 seeded ImageNet reals | COCO-shortcut check (below) |
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | **off** | `data.cifake.enabled: false` | 32×32 px: forensic artefacts don't survive, nothing transfers to 384 px inputs. Smoke-test only. |

### Leakage guard
`src/data/exclusion.py` hashes the *decoded pixels* (64×64 RGB + original size) of every validation and
alt-real image. `src/train.py` asserts on **every run** that zero training samples hit that list before any
feature is extracted; the pipeline smoke test verifies the assertion fires. Structural guards too: `dalle3`
is refused as a training subset and `real_coco` is filtered to `test2017` only.

### The COCO shortcut
All designated-validation reals are COCO photos. A model can learn "looks like COCO" instead of "is real".
We therefore report AUC twice for every cell — fakes vs COCO reals **and** fakes vs ImageNet reals — and
print the gap. A large gap = shortcut, not success.

## Degradation pipeline ([src/degradation](src/degradation/))

`distortion_prob = 1.0`: every training image gets 1–3 distinct transform families, each at an independently
sampled severity, then horizontal flip p=0.5. The same module builds the evaluation conditions, seeded.

| Family | Levels |
|---|---|
| JPEG quality | 90 / 70 / 50 / 30 |
| Gaussian blur σ | 0.5 / 1.0 / 2.0 |
| Resize (down then back up) | 0.5× / 0.25× |
| Gaussian noise σ | 0.02 / 0.05 / 0.10 |
| Colour jitter (brightness, contrast, saturation) | ±20 % |
| Centre crop | 80 % |

**Official NTIRE pipeline.** The challenge's `distort_images` code *was* downloadable
(Codabench 12761 → Data → "Transformations Script"); it is vendored in
[third_party/aug_utils_train](third_party/aug_utils_train/) and exposed via `degradation.backend: official`.
It was **not** made the default because (a) its pool has no resize/crop, which are two of our six evaluation
families, and (b) its level values differ from the table above. Our eval grid therefore uses the table
implementation; the official pipeline is available for a training-augmentation ablation.

## Training
Cross-entropy (focal γ=2, α=0.5 as a config switch), AdamW wd 0.01, cosine LR with 1-epoch warmup,
weighted sampler for class imbalance, mixed precision, checkpoint on **best robust validation AUC**
(AUC on a randomly-degraded copy of the validation subset), not clean AUC.
Frozen mode caches features to `data/features/*.npz` so head retraining takes seconds.

## Limitations
- **Laptop-scale data.** ~9.5 k training images vs the ~277 k of the challenge. Numbers here are for the
  *pipeline*, not a claim about the recipe's ceiling.
- **Grid on a subset.** Per-condition AUCs use a seeded 2,000-image subset of the 13,841-image validation set
  (+1,000 alt reals). Clean AUC is additionally reported on the full set. At ~5 img/s the full grid would take
  ~11 h per arm.
- **Single-transform evaluation.** The NTIRE test set chains 1–5 of 36 transforms incl. neural compression and
  adversarial watermarks; our six families applied singly are far easier. Expect inflated numbers.
- **Validation fakes are one generator (DALL-E 3).** Generalisation to other generators is not measured here.
- **Training generators partly overlap the validation family** (DALL-E 2 in train, DALL-E 3 in val).
- **DINOv3 arm** requires gated-repo access; **CLIP baseline** runs at 224 px (its native resolution).
- Content-hash exclusion catches re-encodes and renames, not edits/crops of validation images.
- **The designated validation set is heavily duplicated.** The 8,843 DALL-E Advanced files contain only 3,719
  unique images (1,808 images repeated across 3+ session folders; found by our pixel-content hashing). We report
  clean AUC both on the set as given and deduplicated (`clean_auc_full_set_dedup` in `eval/summary.json`).
- **Dataloader workers.** On this 64 GB unified-memory machine more than 2 spawned workers starves the GPU
  (6 workers → ~1 img/s, 2 → ~6 img/s). Defaults are set to 2; raise them on a discrete-GPU box.

## Deviations from the spec (all logged in the run summary too)
1. Training data subsampled (see table) and SDXL / Midjourney-v5 WildFake subsets not pulled.
2. Per-condition grid on a 2,000-image seeded subset; clean AUC also on full set.
3. Severity levels: the spec table gives 4/3/2/3/1/1 levels per family (not 5 each); we sample from the table.
4. Colour jitter has one level (±20 %); each of brightness/contrast/saturation draws its own factor in [0.8, 1.2].
5. Official NTIRE distortion code vendored but not default (rationale above).
6. Frozen mode uses 2 cached augmentation draws of the training set (`feature_epochs`), then 30 head epochs.

## Repo layout
```
configs/        one YAML per arm: frozen_siglip2_giant, lora_siglip2_giant, frozen_dinov3_large, baseline_clip_vitl14
src/data/       registry.py (sources), exclusion.py (leak guard), build_val.py
src/degradation transforms.py (table impl, shared train/eval), official.py (NTIRE wrapper)
src/models/     backbone.py (wrapper, param assert, linear head, LoRA)
src/features.py cached frozen-feature extraction
src/train.py    frozen + lora
src/evaluate.py grid, deltas, error rates, thresholds/ROC, contact sheets
src/compare.py  baseline comparison table
predict.py      deliverable CLI
scripts/        prepare_data.sh, remote_zip.py, run_frozen_pipeline.sh
third_party/    official NTIRE 2026 distortion pipeline
tests/          degradation unit tests, end-to-end smoke test
results/        CSVs, heatmaps, contact sheets, RESULTS.md
```

## Contributions
- Chin Mun Yau — everything so far (spec, data plumbing, model, eval harness). Update this section per team member.
