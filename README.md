# Robust AI-Generated Image Detector

Detect AI-generated images even after common real-world transformations such as JPEG compression,
blur, resizing, noise, colour shifts, and cropping. Given an image, the detector returns
`p(AI-generated) ∈ [0,1]`.

This repository contains the trained detector, evaluation pipeline, reproducible experiment results,
and a live vertical-video demo.

## At a glance

| | |
|---|---|
| Best external ROC AUC | **0.9917** on Community Forensics-Eval |
| Designated benchmark ROC AUC | **0.9991** clean; **0.9951** under the worst tested transform |
| Model | SigLIP2-giant vision tower + one linear classification layer |
| Model size | 1.164B parameters; asserted below the 2B limit at startup |
| Input | An image or a directory of images |
| Output | A probability from 0 (real) to 1 (AI-generated) |
| Main metric | ROC AUC, reported clean and by transform/severity |

**Start here:** [run the detector](#quick-start) · [review the results](results/RESULTS.md) ·
[open the technical report](docs/REPORT.md) · [run the web demo](project_demo/README.md)

## How it works

The detector reproduces the 5th-place NTIRE 2026 *Robust AI-Generated Image Detection* recipe:

```
image
  → squish-resize to 384×384 (no crop; aspect ratio ignored)
  → SigLIP2-giant vision tower (text tower discarded)
  → mean-pool the final-layer patch tokens
  → one linear layer
  → real / AI-generated logits
```

The vision tower is `google/siglip2-giant-opt-patch16-384` (1.164B parameters). The model does not use
a CLS token or attention pooling. Every backbone is parameter-counted at startup and must remain below
2B parameters ([backbone.py](src/models/backbone.py)). SigLIP2, DINOv3, and CLIP all share the same
configurable code path.

## Results

The shipped model is `configs/frozen_siglip2_giant_ship.yaml`, paired with the 16KB linear-head
checkpoint at `results/frozen_siglip2_giant_ship/head_best.pt`.

| Evaluation | Result |
|---|---:|
| Community Forensics-Eval (3,759 unseen images) | **0.9917 ROC AUC** |
| Designated benchmark, clean (13,841 images) | **0.9991 ROC AUC** (0.9991 deduplicated) |
| Worst single transform: Gaussian noise at σ=0.1 | **0.9951 ROC AUC** (maximum drop: 0.0039) |
| Accuracy at 0.5 | 0.9850 |
| Balanced accuracy at 0.5 | 0.9856 |
| Majority-class baseline | 0.6545 |
| COCO-real vs ImageNet-real shortcut gap | +0.0011 |

The model was selected on **external generalization**, not the designated benchmark. The benchmark is
saturated: 0.9994 is reachable, leaving about 0.0006 of headroom—below the noise floor for 4,998 real
images. The previous candidate, `frozen_siglip2_giant_mjv5`, reaches 0.9994 on the designated benchmark
but only **0.9715** on the same external set in a like-for-like comparison.

The headline metric throughout this project is ROC AUC, reported for clean images and for every
transform/severity. Accuracy is always accompanied by balanced accuracy and the majority-class baseline
because the validation split is imbalanced at 1:1.77. See [the complete results](results/RESULTS.md) or
[the model-selection reasoning](docs/REPORT.md) (section 7.8).

## Quick start

### Requirements

- Python 3.12
- About 2.3GB for the model download
- macOS with Apple Silicon, Linux with an NVIDIA GPU, or CPU-only execution
- WSL on Windows if you plan to run the data-preparation scripts

The code automatically selects CUDA, then MPS, then CPU. CPU execution uses fp32.

### Install and verify

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements-lock.txt

# Offline smoke tests; no model or dataset download required
pytest tests/test_degradation.py tests/test_pipeline_smoke.py
```

`requirements-lock.txt` contains the exact versions used for the reported results. Use
`requirements.txt` instead if you prefer the latest compatible versions. On NVIDIA systems, install
the matching CUDA build of PyTorch first by following the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Run the detector

```bash
python predict.py --image_dir path/to/images --output preds.json
```

The first run downloads about 2.3GB of SigLIP2-giant weights from Hugging Face; no token is required.
The command uses the shipped config and checkpoint by default and writes:

```json
[
  {"image_path": "path/to/image.jpg", "pred": 0.9972}
]
```

Use `--config` and `--checkpoint` to select a different experiment arm.

## Project guide

| Deliverable | Location |
|---|---|
| Inference script (image dir → JSON of `image_path`, `pred`) | [`predict.py`](predict.py) |
| Robustness Evaluation Summary (clean vs each transform) | [`docs/ROBUSTNESS_SUMMARY.md`](docs/ROBUSTNESS_SUMMARY.md) |
| Error Analysis Note (FPs, FNs, trade-offs) | [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) |
| Full technical report (methodology, every parameter, literature) | [`docs/REPORT.md`](docs/REPORT.md) |
| Dataset and model license audit | [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) |
| Experiment log (25 issues, each with a results comment) | GitHub Issues |
| Live web demo (vertical feed, real-time detection on video frames) | [`project_demo/`](project_demo/) |
| Demo video | *link in Devpost submission* |

## Reproduce the results

### Prepare the data

Data is needed only for retraining and evaluation:

```bash
# About 5GB: base SID_Set and WildFake slices, validation set, exclusion list
scripts/prepare_data.sh

# About 13GB: shipped training mix and two external evaluation sets
scripts/prepare_data_ship.sh
```

Allow about 20GB for `data/`, plus roughly 1GB per experiment arm for feature caches under
`data/features/`. Model weights require about 2.3GB, plus another 2GB if the so400m fallback is used.
Nothing under `data/` or `logs/` is tracked by Git.

### Run an experiment

| Step | Command | Output |
|---|---|---|
| Param check + backbone load | `python -c "from src.common import *; build_backbone(load_config('configs/frozen_siglip2_giant.yaml'), get_device())"` | prints `[param-check] ... 1.164B` |
| Degradation unit tests | `pytest tests/test_degradation.py` | 8 tests |
| Pipeline smoke test (stub backbone) | `pytest tests/test_pipeline_smoke.py` | train→eval→predict in ~2 min |
| Validation set + exclusion list | `python -m src.data.build_val --config configs/frozen_siglip2_giant.yaml` | `data/exclusion/wildfake_val_hashes.txt` |
| Frozen baseline (arm 1) | `python -m src.train --config configs/frozen_siglip2_giant.yaml` | `results/frozen_siglip2_giant/head_best.pt`, `train_summary.json` |
| **Ship arm** (needs `prepare_data_ship.sh`) | `python -m src.train --config configs/frozen_siglip2_giant_ship.yaml && python -m src.evaluate --config configs/frozen_siglip2_giant_ship.yaml` | `results/frozen_siglip2_giant_ship/` |
| External generalization (headline 0.9917 / 0.9732) | `python scripts/eval_external.py --config configs/frozen_siglip2_giant_ship.yaml` then `... --source dir --root data/rrdataset/images --name rrdataset` | `results/frozen_siglip2_giant_ship/eval/external_*.json` |
| One-class downscale control | `python scripts/resolution_control.py` | `results/frozen_siglip2_giant_2x2/resolution_control.txt` |
| Full eval grid | `python -m src.evaluate --config configs/frozen_siglip2_giant.yaml` | `results/frozen_siglip2_giant/eval/{auc_grid.csv,auc_grid.png,thresholds.csv,roc.png,fpr_vs_threshold.png,errors_fp.png,errors_fn.png,summary.json}` |
| LoRA arm | `python -m src.train --config configs/lora_siglip2_giant.yaml && python -m src.evaluate --config configs/lora_siglip2_giant.yaml` | `results/lora_siglip2_giant/` |
| CLIP ViT-L/14 linear-probe baseline | same two commands with `configs/baseline_clip_vitl14.yaml` | `results/baseline_clip_vitl14/` |
| DINOv3 comparison | `configs/frozen_dinov3_large.yaml` (gated HF repo: `export HF_TOKEN=...` after requesting access) | `results/frozen_dinov3_large/` |
| Comparison table | `python -m src.compare results/baseline_clip_vitl14 results/frozen_siglip2_giant results/lora_siglip2_giant` | `results/comparison.csv` |
| Deliverable | `python predict.py --image_dir <dir> --output preds.json` (defaults to the ship config + committed head; `--config/--checkpoint` to override) | JSON `[{"image_path", "pred"}]` |

All randomness is seeded (`seed: 0` in configs; degradation draws are seeded per (epoch, index)).

### Hardware and runtime notes

The reported experiments ran on an Apple M5 Pro with 64GB unified memory using MPS and bf16.
SigLIP2-giant processes about 5.4 images per second on that machine. Expect roughly 50–100 images per
second on an A100 and 0.3–0.5 images per second on CPU. An 80GB GPU can use the complete datasets by
setting `max_train: 0` and `n_eval_max: 0`.

- Prevent the machine from sleeping during long runs. On macOS, use `caffeinate -i <command>`.
- Keep dataloader workers at 2 on unified-memory Macs; higher counts can starve the GPU.
- Progress bars use carriage returns. For readable logs, run `tr '\r' '\n' < logs/x.log`.

## Datasets and leakage protection

### Dataset overview

| Source | Role | What we use | Why |
|---|---|---|---|
| [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) | train | baseline arm: 4,000 imgs from shards 0–7 of 249; **ship arm: 8,000 seeded from shards 0–23** (label 0 real / 1 full-synthetic; label 2 *tampered* excluded) | full set is 140 GB |
| [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake) | train | baseline arm: 3,000 LAION-5B reals + 2,500 DALL-E 2 fakes. **Ship arm: 3,000 LAION reals + 3,000 COCO test2017 reals + 2,500 MJ-v5 fakes + 3,000 across six GAN families** (DF-GAN, GALIP, GigaGAN, starGAN, styleGAN, BigGAN). All seeded slices pulled by HTTP-range from the zips ([remote_zip.py](scripts/remote_zip.py), [prepare_data_ship.sh](scripts/prepare_data_ship.sh)) | full set is 1.29 TB; never downloaded whole |
| [Community Forensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval) | held-out **external** | 3,759 imgs from 30 of 413 shards | generalization to unseen generators (headline 0.9917) |
| [RRDataset](https://zenodo.org/records/14963880) | held-out **external** | all 3,000 train+val imgs (1,500/1,500) | second, independent generalization check (0.9732) |
| WildFake **validation** | held-out | **4,998 COCO val2017 reals + 8,843 DALL-E Advanced (dalle3) fakes** — the organisers' designated subset | never trained on |
| WildFake ImageNet reals | held-out **alt-real** | 1,000 seeded ImageNet reals | COCO-shortcut check (below) |
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | **off** | `data.cifake.enabled: false` | 32×32 px: forensic artefacts don't survive, nothing transfers to 384 px inputs. Smoke-test only. |

### Leakage guard
`src/data/exclusion.py` hashes the *decoded pixels* (64×64 RGB + original size) of every validation and
alt-real image. `src/train.py` asserts on **every run** that zero training samples hit that list before any
feature is extracted; the pipeline smoke test verifies the assertion fires. Structural guards too: `dalle3`
is refused as a training subset and `real_coco` is filtered to `test2017` only.

### COCO shortcut check
All designated-validation reals are COCO photos. A model can learn "looks like COCO" instead of "is real".
We therefore report AUC twice for every cell — fakes vs COCO reals **and** fakes vs ImageNet reals — and
print the gap. A large gap = shortcut, not success.

## Robustness and training

### Degradation pipeline

Implementation: [`src/degradation/`](src/degradation/)

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

### Training
Cross-entropy (focal γ=2, α=0.5 as a config switch), AdamW wd 0.01, cosine LR with 1-epoch warmup,
weighted sampler for class imbalance, mixed precision, checkpoint on **best robust validation AUC**
(AUC on a randomly-degraded copy of the validation subset), not clean AUC.
Frozen mode caches features to `data/features/*.npz` so head retraining takes seconds.

## Limitations

### Shipped detector

- **Out-of-distribution is the real frontier, not robustness.** All 14 degraded conditions sit above 0.995,
  but the external sets sit at 0.9917 (Community Forensics) and 0.9732 (RRDataset). The remaining errors are
  content failures — photorealistic renders of mundane scenes, graphics-like real photos — that are the same
  clean and degraded (`docs/ERROR_ANALYSIS.md`).
- **Pixel-space diffusion is the weakest architecture family** (Hourglass 0.932). Training on ADM/DDIM/DDPM
  lifts it to 0.979 but raises resolution sensitivity; shipped as an option, not the default (issue #22).
- **A residual resolution cue remains.** When only real images are downscaled, 5.1% get flagged vs 1.2%
  clean. Reduced ~13× from the unfixed GAN arm; not zero. Expect a raised false-positive rate on heavily
  thumbnailed real photos (`scripts/resolution_control.py`).
- **The 0.5 threshold is not the right operating point.** Calibration shifted when we changed the training
  mix (acc@0.5 fell 0.6 pt while AUC held). For FPR ≤ 1% use 0.64; the table is in `eval/thresholds.csv`.
- **Frozen probe.** A linear head can only re-weight features the backbone already computes. Published work
  (B-Free) shows end-to-end fine-tuning can beat frozen probes by large margins; we measured LoRA at
  ~0.06 img/s on this laptop (~22 days per run) and could not test it.

### Evaluation

- **The designated benchmark is saturated.** 0.9994 is reachable, leaving ~0.0006 of headroom — below the
  noise floor for 4,998 real images. It can no longer rank arms; every decision after the first week was
  made on the external sets. Issue #20 further shows the benchmark ranks training generators *backwards*
  relative to generalization, because its fakes (DALL-E 3, 2023) reward vintage-matched training data.
- **The designated validation fakes are one generator and heavily duplicated.** 8,843 DALL-E files contain
  only 3,719 unique images. We report clean AUC on the set as given and deduplicated.
- **Two external sets, both partial.** Community Forensics is 30 of 413 shards; RRDataset is the `original`
  split only — its re-digitization axis (photographed screens, where published detectors lose 88–90 points)
  is unmeasured and is the single most valuable test we did not run.
- **The alt-real shortcut check is blind to resolution shortcuts** (both real sets are ~200 px). Caught by
  adding the one-class-downscale control after it nearly fooled us.
- **Grid on a 2,000-image subset**; clean AUC also on the full set. Single transforms only — the NTIRE test
  chains up to 5 of 36 transforms, so our conditions are easier than the competition's.
- Content-hash exclusion catches re-encodes and renames, not edits or crops of validation images.

## Next steps

1. **Score RRDataset's re-digitization split.** It is the honest stress test for social-media deployment and
   nothing in our training resembles a photographed screen.
2. **Retrain with a third low-resolution real source** to drive the 5.1% residual toward the 0.7% the
   benchmark-optimal arm shows, then re-add pixel-space diffusion data on top.
3. **Re-run the DALL-E 2 experiment with external evaluation.** Dropping it raised the designated score by
   0.06, but #20 shows that benchmark rewards vintage-matching; we have not established it improved the
   detector. This is the one result in our own story we can no longer fully defend.
4. **A better frozen encoder, judged on the external sets.** Three 2026 papers rank PE-Core and DINOv3 above
   SigLIP2 as frozen feature sources; DINOv3 needs gated-repo access, PE-Core needs a param-count check.
5. **SSAFE-style MMD source screening** on the per-source feature caches we already have — a principled,
   CPU-only replacement for our brute-force `--keep` sweeps.
6. **LoRA on real GPU hardware**, judged externally — the only way to test whether "benchmark-limited" is
   really "frozen-probe-limited".

## Deviations from the specification

These are also recorded in the run summary.

1. Training data subsampled (see table). The shipped mix differs from the spec's `[laion, dalle2]` WildFake slice — DALL-E 2 was dropped and COCO test2017 reals, MJv5 and six GAN families added; every change is measured in `docs/REPORT.md` §6–7.
2. Per-condition grid on a 2,000-image seeded subset; clean AUC also on full set.
3. Severity levels: the spec table gives 4/3/2/3/1/1 levels per family (not 5 each); we sample from the table.
4. Colour jitter has one level (±20 %); each of brightness/contrast/saturation draws its own factor in [0.8, 1.2].
5. Official NTIRE distortion code vendored but not default (rationale above).
6. Frozen mode uses 2 cached augmentation draws of the training set (`feature_epochs: 2` in every config; the code default if unset is 3), then 30 head epochs.

## Repository layout
```
configs/        one YAML per arm (18). Shipped: frozen_siglip2_giant_ship. Baseline: frozen_siglip2_giant.
                Others are the ablation arms referenced in docs/REPORT.md (2x2, gan, mjv5, sidonly, official, ...)
src/data/       registry.py (sources), exclusion.py (leak guard), build_val.py
src/degradation transforms.py (table impl, shared train/eval), official.py (NTIRE wrapper)
src/models/     backbone.py (wrapper, param assert, linear head, LoRA)
src/features.py cached frozen-feature extraction
src/train.py    frozen + lora
src/evaluate.py grid, deltas, error rates, thresholds/ROC, contact sheets
src/compare.py  baseline comparison table
predict.py      deliverable CLI
project_demo/   live web demo: vertical video feed, frames scored on MPS as you scroll.
                detector.py (model lifecycle + startup self-test), server.py (FastAPI),
                fetch_videos.py (manifest -> ffmpeg normalise -> validate), preflight.py,
                videos.json (manifest + pre-registered clip criteria), static/ (no build step).
                Media is gitignored and fetched from the original hosts; see docs/DATA_LICENSES.md.
scripts/        prepare_data.sh + prepare_data_ship.sh (all data pulls), remote_zip.py (HTTP-range zip slicing),
                eval_external.py, data_ablation.py, resolution_control.py, head_sweep.py, make_results.py.
                run_queue*.sh / run_*_chain.sh / dl_*.sh are the historical launch scripts for issues #1-#25;
                they are kept for provenance and are not meant to be re-run.
third_party/    official NTIRE 2026 distortion pipeline
tests/          degradation unit tests, end-to-end smoke test (both offline, CPU, no data needed)
results/        CSVs, heatmaps, contact sheets, RESULTS.md; the two committed heads (*_ship, *_mjv5) are 16 KB each
requirements-lock.txt   exact versions used for every reported number
```

## Team

- Chin Mun Yau
- Kayra
- Adam
- Boey
- Verlin
