# Robust AI-Generated Image Detector

Prototype for a hackathon track on detecting AI-generated images **under real-world transformations**
(JPEG, blur, resize, noise, colour shift, crop). Given an image it outputs `p(AI-generated) ∈ [0,1]`.

The engine reproduces the 5th-place NTIRE 2026 _Robust AI-Generated Image Detection_ recipe exactly:

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

|                                                                            |                                         |
| -------------------------------------------------------------------------- | --------------------------------------- |
| Designated benchmark, clean AUC (13,841 imgs)                              | **0.9991** (dedup 0.9991)               |
| Worst single transform                                                     | noise@0.1 **0.9951** (max delta 0.0039) |
| Accuracy @ 0.5 / balanced acc / majority baseline                          | 0.9850 / 0.9856 / 0.6545                |
| Shortcut gap (COCO reals vs ImageNet reals)                                | +0.0011                                 |
| **External generalization** (Community Forensics-Eval, 3,759 imgs, unseen) | **0.9917**                              |

Selected on **external** generalization, not the designated benchmark - the benchmark is saturated
(0.9994 is reachable, leaving ~0.0006 of headroom, below the noise floor for 4,998 real images) and can
no longer rank arms. The previous candidate (`frozen_siglip2_giant_mjv5`, 0.9994 designated) scores
**0.9715** on the same external set under a like-for-like comparison. Full reasoning:
[docs/REPORT.md](docs/REPORT.md), section 7.8.

### Hackathon deliverables — where each one lives

| Deliverable                                                      | Location                                                   |
| ---------------------------------------------------------------- | ---------------------------------------------------------- |
| Inference script (image dir → JSON of `image_path`, `pred`)      | [`predict.py`](predict.py)                                 |
| Robustness Evaluation Summary (clean vs each transform)          | [`docs/ROBUSTNESS_SUMMARY.md`](docs/ROBUSTNESS_SUMMARY.md) |
| Error Analysis Note (FPs, FNs, trade-offs)                       | [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md)         |
| Full technical report (methodology, every parameter, literature) | [`docs/REPORT.md`](docs/REPORT.md)                         |
| Dataset and model license audit                                  | [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md)           |
| Experiment log (25 issues, each with a results comment)          | GitHub Issues                                              |
| Demo video                                                       | _link in Devpost submission_                               |

---

## Setup

Tested on macOS (Apple Silicon, MPS). Linux with an NVIDIA GPU and CPU-only machines work through the same
code path (`src/common.py:get_device()` picks `cuda` → `mps` → `cpu`; CPU forces fp32). **Windows: use WSL**
(the data scripts need `bash`, `curl`, `unzip`, `tar`); the Python side itself is platform-neutral.

```bash
# 1. Environment (Python 3.12). requirements.txt has ranges; requirements-lock.txt pins the exact versions
#    every number in this repo was produced with (torch 2.13, transformers 5.15, peft 0.20, numpy 2.5).
uv venv --python 3.12 .venv && source .venv/bin/activate      # Windows/WSL: same; native PowerShell: .venv\Scripts\activate
uv pip install -r requirements-lock.txt                        # or -r requirements.txt for latest compatible
#    NVIDIA GPU: install the matching CUDA build of torch FIRST from https://pytorch.org/get-started/locally/
#    (plain `pip install torch` on Linux/Windows may give a CPU-only wheel), then run the line above.

# 2. Quick verify — any OS, no data, no downloads, ~3 min on CPU
pytest tests/test_degradation.py tests/test_pipeline_smoke.py  # stub backbone; exercises train -> eval -> predict

# 3. Run the shipped detector on your own images (first run downloads the ~2.3 GB SigLIP2-giant weights
#    from Hugging Face into ~/.cache/huggingface; no token needed)
python predict.py --image_dir <dir> --output preds.json        # uses configs/frozen_siglip2_giant_ship.yaml
                                                               # + results/frozen_siglip2_giant_ship/head_best.pt (16 KB, in git)

# 4. Data — only needed to retrain / re-evaluate
scripts/prepare_data.sh           # ~5 GB: base slices of SID_Set + WildFake, validation set, exclusion list
scripts/prepare_data_ship.sh      # ~13 GB more: the ship training mix (SID 8-23, COCO test2017, MJv5, 6 GAN
                                  #   families) + the two external eval sets (Community Forensics, RRDataset)
```

Disk: ~2.3 GB model weights (+2 GB if the so400m fallback triggers), ~20 GB `data/` after both scripts, plus
feature caches in `data/features/` (~1 GB per arm). Nothing under `data/` or `logs/` is in git.

Hardware used: Apple M5 Pro, 64 GB unified memory (MPS, bf16). SigLIP2-giant runs at ~5.4 img/s here;
this throughput dictated every subsampling decision below. On an 80 GB GPU nothing needs subsampling —
set `max_train: 0`, `n_eval_max: 0`. Expect ~50–100 img/s on an A100, and ~0.3–0.5 img/s on CPU (fine for
`predict.py` on a few hundred images; not for training/eval — the ship extraction is ~35 k images).

**Runtime pitfalls (all platforms)**: the machine must not sleep during long runs (macOS: `caffeinate -i <cmd>`);
keep `--workers`/dataloader workers at 2 on unified-memory Macs (more starves the GPU); progress bars use
`\r`, so read logs with `tr '\r' '\n' < logs/x.log`.

## Reproduce every number

| Step                                               | Command                                                                                                                                                 | Output                                                                                                                                               |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Param check + backbone load                        | `python -c "from src.common import *; build_backbone(load_config('configs/frozen_siglip2_giant.yaml'), get_device())"`                                  | prints `[param-check] ... 1.164B`                                                                                                                    |
| Degradation unit tests                             | `pytest tests/test_degradation.py`                                                                                                                      | 8 tests                                                                                                                                              |
| Pipeline smoke test (stub backbone)                | `pytest tests/test_pipeline_smoke.py`                                                                                                                   | train→eval→predict in ~2 min                                                                                                                         |
| Validation set + exclusion list                    | `python -m src.data.build_val --config configs/frozen_siglip2_giant.yaml`                                                                               | `data/exclusion/wildfake_val_hashes.txt`                                                                                                             |
| Frozen baseline (arm 1)                            | `python -m src.train --config configs/frozen_siglip2_giant.yaml`                                                                                        | `results/frozen_siglip2_giant/head_best.pt`, `train_summary.json`                                                                                    |
| **Ship arm** (needs `prepare_data_ship.sh`)        | `python -m src.train --config configs/frozen_siglip2_giant_ship.yaml && python -m src.evaluate --config configs/frozen_siglip2_giant_ship.yaml`         | `results/frozen_siglip2_giant_ship/`                                                                                                                 |
| External generalization (headline 0.9917 / 0.9732) | `python scripts/eval_external.py --config configs/frozen_siglip2_giant_ship.yaml` then `... --source dir --root data/rrdataset/images --name rrdataset` | `results/frozen_siglip2_giant_ship/eval/external_*.json`                                                                                             |
| One-class downscale control                        | `python scripts/resolution_control.py`                                                                                                                  | `results/frozen_siglip2_giant_2x2/resolution_control.txt`                                                                                            |
| Full eval grid                                     | `python -m src.evaluate --config configs/frozen_siglip2_giant.yaml`                                                                                     | `results/frozen_siglip2_giant/eval/{auc_grid.csv,auc_grid.png,thresholds.csv,roc.png,fpr_vs_threshold.png,errors_fp.png,errors_fn.png,summary.json}` |
| LoRA arm                                           | `python -m src.train --config configs/lora_siglip2_giant.yaml && python -m src.evaluate --config configs/lora_siglip2_giant.yaml`                       | `results/lora_siglip2_giant/`                                                                                                                        |
| CLIP ViT-L/14 linear-probe baseline                | same two commands with `configs/baseline_clip_vitl14.yaml`                                                                                              | `results/baseline_clip_vitl14/`                                                                                                                      |
| DINOv3 comparison                                  | `configs/frozen_dinov3_large.yaml` (gated HF repo: `export HF_TOKEN=...` after requesting access)                                                       | `results/frozen_dinov3_large/`                                                                                                                       |
| Comparison table                                   | `python -m src.compare results/baseline_clip_vitl14 results/frozen_siglip2_giant results/lora_siglip2_giant`                                            | `results/comparison.csv`                                                                                                                             |
| Deliverable                                        | `python predict.py --image_dir <dir> --output preds.json` (defaults to the ship config + committed head; `--config/--checkpoint` to override)           | JSON `[{"image_path", "pred"}]`                                                                                                                      |

All randomness is seeded (`seed: 0` in configs; degradation draws are seeded per (epoch, index)).

## Data

| Source                                                                                           | Role                  | What we use                                                                                                                                                                                                                                                                                                                                                                         | Why                                                                                              |
| ------------------------------------------------------------------------------------------------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)                                       | train                 | baseline arm: 4,000 imgs from shards 0–7 of 249; **ship arm: 8,000 seeded from shards 0–23** (label 0 real / 1 full-synthetic; label 2 _tampered_ excluded)                                                                                                                                                                                                                         | full set is 140 GB                                                                               |
| [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake)                                 | train                 | baseline arm: 3,000 LAION-5B reals + 2,500 DALL-E 2 fakes. **Ship arm: 3,000 LAION reals + 3,000 COCO test2017 reals + 2,500 MJ-v5 fakes + 3,000 across six GAN families** (DF-GAN, GALIP, GigaGAN, starGAN, styleGAN, BigGAN). All seeded slices pulled by HTTP-range from the zips ([remote_zip.py](scripts/remote_zip.py), [prepare_data_ship.sh](scripts/prepare_data_ship.sh)) | full set is 1.29 TB; never downloaded whole                                                      |
| [Community Forensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval)     | held-out **external** | 3,759 imgs from 30 of 413 shards                                                                                                                                                                                                                                                                                                                                                    | generalization to unseen generators (headline 0.9917)                                            |
| [RRDataset](https://zenodo.org/records/14963880)                                                 | held-out **external** | all 3,000 train+val imgs (1,500/1,500)                                                                                                                                                                                                                                                                                                                                              | second, independent generalization check (0.9732)                                                |
| WildFake **validation**                                                                          | held-out              | **4,998 COCO val2017 reals + 8,843 DALL-E Advanced (dalle3) fakes** — the organisers' designated subset                                                                                                                                                                                                                                                                             | never trained on                                                                                 |
| WildFake ImageNet reals                                                                          | held-out **alt-real** | 1,000 seeded ImageNet reals                                                                                                                                                                                                                                                                                                                                                         | COCO-shortcut check (below)                                                                      |
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | **off**               | `data.cifake.enabled: false`                                                                                                                                                                                                                                                                                                                                                        | 32×32 px: forensic artefacts don't survive, nothing transfers to 384 px inputs. Smoke-test only. |

### Leakage guard

`src/data/exclusion.py` hashes the _decoded pixels_ (64×64 RGB + original size) of every validation and
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

| Family                                           | Levels             |
| ------------------------------------------------ | ------------------ |
| JPEG quality                                     | 90 / 70 / 50 / 30  |
| Gaussian blur σ                                  | 0.5 / 1.0 / 2.0    |
| Resize (down then back up)                       | 0.5× / 0.25×       |
| Gaussian noise σ                                 | 0.02 / 0.05 / 0.10 |
| Colour jitter (brightness, contrast, saturation) | ±20 %              |
| Centre crop                                      | 80 %               |

**Official NTIRE pipeline.** The challenge's `distort_images` code _was_ downloadable
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

## Limitations — and what we would do with more time

**Measured limitations of the shipped detector**

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

**Limitations of the evaluation**

- **The designated benchmark is saturated.** 0.9994 is reachable, leaving ~0.0006 of headroom — below the
  noise floor for 4,998 real images. It can no longer rank arms; every decision after the first week was
  made on the external sets. Issue #20 further shows the benchmark ranks training generators _backwards_
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

**What we would do given more time, in order**

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

## Deviations from the spec (all logged in the run summary too)

1. Training data subsampled (see table). The shipped mix differs from the spec's `[laion, dalle2]` WildFake slice — DALL-E 2 was dropped and COCO test2017 reals, MJv5 and six GAN families added; every change is measured in `docs/REPORT.md` §6–7.
2. Per-condition grid on a 2,000-image seeded subset; clean AUC also on full set.
3. Severity levels: the spec table gives 4/3/2/3/1/1 levels per family (not 5 each); we sample from the table.
4. Colour jitter has one level (±20 %); each of brightness/contrast/saturation draws its own factor in [0.8, 1.2].
5. Official NTIRE distortion code vendored but not default (rationale above).
6. Frozen mode uses 2 cached augmentation draws of the training set (`feature_epochs: 2` in every config; the code default if unset is 3), then 30 head epochs.

## Repo layout

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
scripts/        prepare_data.sh + prepare_data_ship.sh (all data pulls), remote_zip.py (HTTP-range zip slicing),
                eval_external.py, data_ablation.py, resolution_control.py, head_sweep.py, make_results.py.
                run_queue*.sh / run_*_chain.sh / dl_*.sh are the historical launch scripts for issues #1-#25;
                they are kept for provenance and are not meant to be re-run.
third_party/    official NTIRE 2026 distortion pipeline
tests/          degradation unit tests, end-to-end smoke test (both offline, CPU, no data needed)
results/        CSVs, heatmaps, contact sheets, RESULTS.md; the two committed heads (*_ship, *_mjv5) are 16 KB each
requirements-lock.txt   exact versions used for every reported number
```

## Team and contributions

<!-- TODO before submission: one line per team member -->

- Chin Mun Yau — spec, data plumbing, model, eval harness, experiment programme (issues #1–#25).
- _(other members: to be added)_
