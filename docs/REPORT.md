# Robust AI-Generated Image Detection — Full Technical Report

**Project:** techjam2026 · **Date:** 2026-08-29 · **Hardware:** single Apple M5 Pro laptop (64 GB, MPS)

This document is the complete record: what we built, what we changed, what we deliberately did not
change, what we measured, what we read, and what we still don't know. It is written to be readable
by someone who is not a machine-learning researcher — every term is defined the first time it appears —
while carrying enough detail to defend each number.

---

## 0. Executive summary

We built a detector that takes an image and returns `p(AI-generated)` — a number between 0 and 1.

**The headline result: we moved the score from 0.9309 to 0.9994 without changing the model, the
architecture, or a single training hyperparameter. Every point came from fixing the training data.**

| | Starting point (spec as written) | Shipped detector |
|---|---|---|
| Clean ROC-AUC (designated benchmark, 13,841 images) | 0.9309 | **0.9994** |
| Worst single-transform cell | resize@0.25 → 0.8097 | noise@0.02 → **0.9991** |
| Max degradation drop | 0.1118 | **0.0003** |
| Accuracy @ 0.5 (with baseline 0.6545) | — | **0.9910** (balanced 0.9915) |
| Shortcut gap (COCO reals vs ImageNet reals) | −0.0270 | **−0.0002** |
| External generalization (unseen dataset, 3,759 images) | not measured | **0.9644** |

Along the way we ran 25 tracked experiments, falsified two of our own hypotheses, caught two
confounds (spurious shortcuts the model was exploiting instead of actually detecting fakes), and
found that the single external weak point is **not** robustness to image damage but **generalization
to generator architectures we never trained on**.

**The one-line story for the pitch:** *architecture was a solved problem; data curation was the entire
game, and we can prove it because we measured both.*

---

## 1. Glossary — read this first if you are not a data scientist

| Term | Plain meaning |
|---|---|
| **ROC-AUC** (or just AUC) | Take one real photo and one AI image at random. AUC is the probability the detector scores the AI one higher. 0.5 = coin flip, 1.0 = perfect. It is *threshold-free*: it measures ranking quality, not a yes/no decision. |
| **Clean AUC** | AUC on undamaged images. |
| **Degraded / transform / distortion** | We deliberately damage the test images (compress, blur, shrink, add noise…) to simulate real-world sharing. A detector that only works on pristine images is useless. |
| **Worst cell** | Of the 15 damage conditions we test, the one where the detector scores lowest. This is the honest number — an average hides the failure. |
| **Max delta** | Clean AUC minus worst cell. How much performance the worst damage costs. Small = robust. |
| **Backbone / encoder** | A large pretrained vision model that converts an image into a list of numbers (a "feature vector") describing it. We do not train it. |
| **Frozen** | We leave the backbone's weights untouched and only train a tiny classifier on top of its output. Fast, cheap, reproducible. |
| **Linear head / probe** | The tiny classifier: one matrix multiply mapping 1,536 numbers to one score. ~1,537 trainable parameters vs 1.16 billion frozen ones. |
| **LoRA** | A method for cheaply fine-tuning the big backbone itself. We benchmarked it and rejected it on cost (see §6.2). |
| **Patch tokens** | The backbone chops the image into a 24×24 grid of squares and describes each one. 576 descriptions, each 1,536 numbers. |
| **Mean pooling** | Average those 576 descriptions into one. The alternative (CLS token, attention pooling) is discussed in §6.2. |
| **Shortcut / confound** | The model learning a cue that *correlates* with the answer in our data but isn't actually evidence of AI generation — e.g. "JPEG files are real, PNG files are fake." Works on the benchmark, fails in the wild. |
| **Held-out / validation set** | Images the model is never trained on, used to score it. If training data leaks into it, every number is fiction. |
| **Ablation** | Remove or change exactly one thing, re-measure, and attribute the difference to that thing. |
| **In-distribution vs out-of-distribution (OOD)** | Test images that look like training data vs test images from a genuinely different source. OOD is the only honest test of generalization. |
| **TPR / FPR** | True positive rate (fraction of AI images caught) / false positive rate (fraction of real photos wrongly flagged). |
| **Balanced accuracy** | Average of "% of reals correct" and "% of fakes correct". Immune to class imbalance. |
| **Majority baseline** | The accuracy you'd get by always guessing the more common class. Ours is **0.6545** — so raw accuracy below ~65% is worse than a constant. This is why we never report accuracy alone. |

---

## 2. The task and how it is scored

Input: an image. Output: `p(AI-generated) ∈ [0,1]`.

The evaluation is a **grid**: 6 families of image damage × several severity levels, plus a clean row.
15 conditions total, each scored by AUC.

| Family | What it simulates | Levels tested |
|---|---|---|
| **jpeg** | Upload/re-upload compression | quality 90, 70, 50, 30 |
| **blur** | Out-of-focus, denoising filters | Gaussian σ = 0.5, 1.0, 2.0 |
| **resize** | Thumbnailing, low-bandwidth delivery | downscale 0.5×, 0.25× (then back up) |
| **noise** | Sensor noise, transmission artifacts | Gaussian σ = 0.02, 0.05, 0.10 |
| **color** | Filters, auto-enhance | brightness/contrast/saturation ±0.2 |
| **crop** | Reframing, watermark removal | centre crop to 80% |

Defined once in `src/degradation/transforms.py` and used **verbatim** for both training augmentation
and evaluation. This is deliberate: one table, no chance of train/eval drift.

**Three numbers gate everything** (from the original brief):
1. Clean AUC
2. Worst-case single-transform AUC
3. Gap between the two real-image sources (the shortcut check)

---

## 3. System architecture

```
image
  └─ squish-resize to 384×384        (stretch to square, NO crop — preserves all content,
                                      distorts aspect ratio; the NTIRE recipe's choice)
  └─ SigLIP2-giant vision tower      (1.164 B params, FROZEN — asserted < 2 B at every startup)
  └─ 576 patch tokens × 1,536 dims
  └─ mean-pool over patch tokens     (no CLS token, no attention pooling)
  └─ 1,536-dim feature vector        (cached to disk as .npz)
  └─ linear head → 1 logit → sigmoid → p(AI-generated)
```

**Why frozen features are the whole reason this project was possible.** Extracting features is the
expensive step (~10–12 images/second on this laptop). Once cached, training a new head takes
*seconds*. That is what let us run ~30 ablations on a laptop: most of our experiments cost **zero GPU
time** because they reuse the same cached features and only vary the head or the label mix.

Key implementation details worth defending:
- **Parameter cap enforced in code.** `assert_under_limit` in `src/models/backbone.py` hard-fails at
  load if the backbone exceeds 2 B. Not a comment — an assertion. Measured: 1.163659776 B.
- **Leakage check is automatic.** `src/train.py` calls `ExclusionList.assert_disjoint()` before a
  single feature is extracted, on every run. The exclusion list is built from **pixel-content hashes**
  (SHA-1 of the image downscaled to 64×64 RGB, plus original dimensions) — so a re-saved, renamed, or
  re-compressed copy of a validation image is still caught. It cannot be forgotten.
- **Backbone is swappable by config.** `src/models/backbone.py` is family-generic (SigLIP2 / DINOv3 /
  CLIP); only patch-token slicing is model-specific, so all arms share train/eval unchanged. This is
  what made the CLIP baseline a config file rather than a rewrite.

---

## 4. Methodology — how we made the measurements trustworthy

This section matters more than any individual result. Our biggest finding was a confound, and we only
found it because of these controls.

### 4.1 The exclusion list (no leakage, enforced)
4,998 COCO val2017 reals + 8,843 DALL-E 3 fakes are designated validation. They are hashed by pixel
content and structurally refused by `src/data/registry.py`. Asserted every run.

### 4.2 Two independent real-image sources (the original shortcut check)
Benchmarks pair AI images against one real dataset (here: COCO). A detector can score well by learning
"looks like COCO → real" instead of "shows no generation artifacts → real". So we score every arm
against a **second real set from a different source** (1,000 WildFake ImageNet reals) and report the
gap. **Large gap = shortcut, not success.**

Our shipped arm's gap is −0.0002 (i.e. it does *marginally better* on the non-benchmark reals). Good.

### 4.3 The one-class-downscale control (added after we got burned)
§4.2 has a blind spot we discovered the hard way: **both** our real sets are ~200 px median. So a model
using *resolution* as a cue would pass the shortcut check. We added a stronger test:

> Degrade **only one class** and re-score. If the model is genuinely detecting generation artifacts,
> downscaling only the reals shouldn't help it; if it's secretly using "low resolution → real", the
> AUC will collapse.

This caught a confound the standard control missed (§7.4). We later found the published cousin of this
method — **BIAS-ID (arXiv:2605.31153)** — and we cite it rather than claiming novelty (§10.3).

### 4.4 Full-set *and* subset reporting, plus deduplication
The per-condition grid runs on a seeded 2,000-image subset (compute). Clean AUC is **also** computed on
the full 13,841 images. We additionally discovered the designated validation set is internally
duplicated — only **3,719 of 8,843** DALL-E files are unique images — so `summary.json` carries both
`clean_auc_full_set` and `clean_auc_full_set_dedup`.

### 4.5 One variable at a time
`src/head_utils.py` mirrors the production training path exactly, so experiment scripts differ from the
shipped pipeline *only* in the variable under test. Ablations that vary the data mix reuse identical
cached features — the encoder output is bit-identical across arms, so any AUC difference is attributable
to the labels/mix alone.

### 4.6 Robust checkpointing
Checkpoints select on **degraded** validation AUC (`checkpoint_metric: robust_auc`), not clean AUC.
The task is robustness; the selection metric should be too.

---

## 5. Data

| Role | Source | Size | Notes |
|---|---|---|---|
| Train — real | SID_Set (24 shards), WildFake LAION reals, WildFake ImageNet | 12k + 3k | |
| Train — fake | SID_Set synthetic, WildFake MJv5 | 12k + 3k | label 2 "tampered" excluded |
| **Validation (never trained on)** | 4,998 COCO val2017 reals + 8,843 DALL-E 3 fakes | 13,841 | organisers' designated subset |
| Alt-real control | 1,000 WildFake ImageNet reals | 1,000 | non-COCO real source |
| External eval | Community Forensics-Eval (CVPR 2025), 30 shards | 3,759 | 10 generators, never seen |
| External eval #2 | RRDataset | in progress | second generalization bound |

WildFake is 1.29 TB. We never downloaded it. `scripts/remote_zip.py` pulls **seeded slices via HTTP
Range requests**, coalescing byte windows — extracting a 3,000-image sample from a 50 GB archive
without fetching the archive. This is the only reason generator-diversity experiments were feasible.

**Median shortest side by source** (this table is the whole of §7.4):

| Source | Median shortest side |
|---|---|
| WildFake GAN fakes | **256** (91% under 384 px) |
| Validation COCO reals | **200** (100% under 384 px) |
| Alt-real ImageNet | **200** |
| COCO test2017 reals | **200** |
| LAION reals | 713 |
| Validation DALL-E 3 fakes | **1024** |
| MJv5 fakes | **1024** |

---

## 6. Parameter ledger

### 6.1 Parameters we changed — and what it bought

| # | Parameter | From → To | Effect on clean AUC | Verdict |
|---|---|---|---|---|
| 1 | `data.wildfake.train_subsets` — **drop DALL-E 2** | `[laion, dalle2]` → `[laion]` | 0.9309 → **0.9931** | **The single biggest win in the project.** |
| 2 | `data.wildfake.train_subsets` — **add MJv5** | `[laion]` → `[laion, mjv5]` | 0.9949 → **0.9994** | Second biggest. One generator, +0.0045. |
| 3 | `data.sid_set.max_train` | 4,000 → 12,000 | 0.9945 → 0.9949 | Flat. Tripling data bought +0.0004. |
| 4 | Training-image format harmonization | off → on (all train images re-encoded JPEG q90) | +0.0007 | **Null result. Our hypothesis was wrong** (§7.2). Kept anyway — it removes a real risk at zero cost. |
| 5 | `train.feature_epochs` (independent degradation draws) | 2 → 3 | 0.9932 → 0.9938 | Small real gain. |
| 6 | `train.loss` | ce → focal (γ=5) | 0.9932 → 0.9939 | Small gain; collapses at γ=20 (0.9888). |

Everything below the line — every architectural and optimization knob — moved things by ≤0.001.
Everything above it is data. **That ratio is the finding.**

### 6.2 Parameters we deliberately did NOT change — and why

This is the more important half of the ledger. Each of these was considered, most were measured, and
each was left alone for a stated reason rather than by omission.

| Parameter | Left at | Why not changed |
|---|---|---|
| **Backbone** | `siglip2-giant-opt-patch16-384` (1.164 B) | Spec-mandated; reproduces the NTIRE 2026 5th-place recipe. 2026 literature ranks PE-Core and DINOv3 above it (§10.1) — but by the time we could test, the benchmark was at 0.9994 with 0.0006 headroom, **below the noise floor for 4,998 real images**. A better backbone is *unmeasurable here*. DINOv3 is additionally blocked on gated repo access. |
| **Input resolution** | 384 px | Inherited from the checkpoint, not tuned. Giant is published *only* at 384, so testing 512 means simultaneously dropping to a 0.429 B backbone (`so400m-patch16-512`) — the resolution question cannot be asked in isolation. Config written (`frozen_siglip2_so400m_512.yaml`), deferred on measurability. |
| **Pooling** | mean over patch tokens | Spec-mandated (no CLS, no attention pooling). **TAP (arXiv:2604.26772, 2026) disputes this ruling** and makes tunable attention pooling its central contribution. Blocked practically: attention pooling needs the full 576×1,536 token sequence per image (33 GB+) instead of one 1,536-dim vector — it breaks the feature cache that makes everything else cheap. **Parked as a documented decision, not an oversight.** |
| **`weight_decay`** | 0.01 | 40-config sweep. 0.01 → 1.0 is a 100× change and moves clean AUC by 0.0001. Irrelevant. |
| **`epochs`** | 30 | Sweep says 30 > 10 > 5 > 2 > 1 — the *opposite* of the issue's prediction (§7.5). |
| **`lr`** | 1e-3 | Beats 1e-4 at every epoch count. |
| **Feature normalization** | none | Measured: none 0.9932 > standardize 0.9917 > **L2 0.9856**. L2 is actively harmful — it discards embedding magnitude, which for a frozen encoder carries real signal (degraded and synthetic images differ in activation magnitude, not just direction). Bonus: avoiding standardization means no train/serve mismatch risk from persisting scaler stats. |
| **`distortion_prob`** | 1.0 | Tested directly (§7.6): adding clean images to the training draw *hurt* (0.9924 vs 0.9932 at equal volume). The spec's aggressive setting is correct. |
| **Test-time augmentation** | off | hflip view-averaging bought **+0.0002 for 2× inference cost**. The two views correlate r = 0.996 — the model is already flip-invariant. Rejected. |
| **LoRA fine-tuning** | not run | Benchmarked, not guessed: **~260 s/step at batch 16 = 0.06 img/s → ~18 h/epoch → ~22 days for 30 epochs.** On MPS the backward pass is ~170× the forward (activation-memory thrashing), not the usual ~2×. Infeasible on this hardware; documented with the measurement. |
| **Gated multi-expert ensemble** | not built | Killed by a **cheap pre-check** before any GPU spend: we asked whether a probe can even tell *which* degradation was applied. 7-class family accuracy 0.6247; clean-vs-degraded 0.8817 against a **0.9333 majority baseline** — i.e. worse than a constant predictor. blur↔resize confuse badly (both are low-pass). **You cannot route on a gate that can't tell clean from damaged.** |
| **CIFAKE dataset** | disabled | 32×32 px. Behind a config flag, defaults off. |
| **Ensembles / model fusion** | none | Brief: "no extra models, ensembles, or fusion until the single-backbone baseline is measured and reported." Honoured. |
| **Dataloader workers** | 2 | >2 starves the GPU on this unified-memory Mac (6 workers → ~1 img/s; 2 → ~11 img/s). Measured, not assumed. |

**The pattern:** eight of these were rejected *on measurement*, not on intuition, and three of them
(LoRA, the gate, TTA) were killed by a cheap pre-check before any expensive run. That's the
methodology we'd defend hardest.

---

## 7. Findings

### 7.1 The baseline was broken, and it was the data

The spec'd configuration scored **0.9309 clean, worst cell resize@0.25 = 0.8097**. Before assuming a
pipeline bug, we ran a free ablation: retrain the head on the *same cached features*, varying only
which sources are included.

| Training sources | Clean AUC |
|---|---|
| SID_Set only | 0.9932 |
| + WildFake LAION reals | 0.9891 |
| + WildFake DALL-E 2 fakes | 0.9691 |
| full original mix | 0.9316 |
| **WildFake training slice alone** | **0.6234** |

The WildFake training slice, scored on its own, is barely above chance. **Adding data made the
detector worse.** Since the features were identical across all rows, this could only be the label mix.

### 7.2 We falsified our own first hypothesis

Our first explanation was a **format confound**: in the WildFake training slice, DALL-E 2 fakes are all
512² PNG and LAION reals are JPEG — so "JPEG → real" is perfectly learnable and completely useless.

We fixed it: every training image re-encoded through one JPEG q90 pass, so file format stops correlating
with label (validation untouched). Result: **+0.0007. A null result.**

We report this because it's the honest shape of the work — the plausible hypothesis was wrong, and
finding that out is what forced the per-source screening in §7.1, which found the actual cause.

### 7.3 The cause: generator vintage, not file format

Dropping DALL-E 2 alone recovered nearly the whole gap. The likely mechanism is **vintage mismatch**:
DALL-E 2 is 2022; the validation fakes are DALL-E 3 (2023). Training on an obsolete generator teaches
artifacts that modern generators no longer produce — and actively costs performance.

The converse also holds and is the second-biggest win: **adding one modern, high-tier generator (MJv5)
beat tripling the training volume.** 0.9949 → 0.9994 from 3,000 MJv5 images, versus +0.0004 from
8,000 extra SID_Set images.

> **Slide-ready framing:** *which* generators you train on dominates *how many* images you have.

### 7.4 The resolution confound — caught by a control we built after being burned

Trying to close a known GAN weakness (§7.7), we added 3,000 WildFake GAN images to training. The
result collapsed: **clean 0.9197, worst cell resize@0.25 = 0.6531, max delta 0.2607.**

The one-class-downscale control (§4.3) gave the smoking gun:

| Manipulation | SID-only | **Ship (MJv5)** | GAN arm |
|---|---|---|---|
| both clean | 0.9913 | 0.9994 | 0.9137 |
| both resized 0.25× | 0.9943 | 0.9996 | 0.6531 |
| only **fakes** downscaled | 0.9918 | 0.9994 | 0.9328 |
| only **reals** downscaled | 0.9938 | 0.9996 | **0.6183** |
| % of reals flagged at t=0.5 (clean → downscaled) | — | 0.7% → **0.7%** | 23% → **87%** |

Downscaling only the *real* photos made the GAN arm flag **87% of them as AI-generated**. The model had
learned an **inverted resolution cue** — WildFake GAN fakes are 256 px, so "low resolution → fake" —
which then fires on every low-resolution real photo. The ship candidate is completely unmoved (0.7% →
0.7%).

**And the standard control could not have caught this.** Both our real sets (COCO and alt-real ImageNet)
sit at 200 px median — the alt-real check is *structurally blind* to a resolution shortcut. That is a
genuine methodological lesson and it's in the report because it nearly fooled us.

### 7.5 Every architectural and optimization knob was a dead end

| Experiment | Result |
|---|---|
| Regularization sweep (40 configs: wd × epochs × lr) | Configured defaults sit at the optimum. wd irrelevant (100× change → 0.0001 AUC). |
| Feature normalization | none 0.9932 > standardize 0.9917 > L2 0.9856 |
| Focal loss | γ=5 → 0.9939 (vs CE 0.9932); collapses at γ=20 |
| TTA (hflip) | +0.0002 for 2× cost; views correlate r=0.996 |
| Clean training draw | 3 degraded 0.9938 > 2 degraded 0.9932 > 2 degraded + 1 clean 0.9924 |
| Degradation-type gate probe | 7-class 0.6247; clean-vs-degraded 0.8817 vs 0.9333 baseline |

**A note worth putting on a slide:** the regularization issue was opened because the head appeared to
overfit — validation peaked at epoch 1 and decayed by epoch 29, and a plain sklearn logistic regression
beat our trained head. On the *cleaned* data that reverses completely: training longer keeps helping,
and the trained head is at the sweep optimum. **The "overfitting" was the head fitting the data
confound.** Regularization was a symptom-level fix for a data-level problem — we nearly shipped the
wrong fix.

### 7.6 `distortion_prob = 1.0` is validated, not inherited

The spec mandates that *every* training image gets 1–3 random distortions — the head never sees an
undamaged image. That seemed risky, so we tested it at equal volume:

| Training draw | Clean AUC |
|---|---|
| 3 degraded draws | **0.9938** |
| 2 degraded draws (config default) | 0.9932 |
| 2 degraded + 1 clean draw | 0.9924 |

Clean images actively hurt. The aggressive setting is correct, and now we know rather than assume.

### 7.7 The real weakness: generator architectures we never trained on

The designated benchmark is saturated. So we pulled an **external** set the model has never seen —
Community Forensics-Eval (CVPR 2025), 3,759 images across 10 generators.

| Generator | Architecture | SID-only | **Ship (+MJv5)** | GAN arm |
|---|---|---|---|---|
| Hourglass | **Pixel-space diffusion** | 0.5247 *(chance)* | **0.7320** | 0.9683 |
| DFGAN | **GAN** | 0.6880 | **0.9513** | 0.9999 |
| Imagen3 | Commercial | 0.9765 | 0.9850 | 0.9513 |
| Midjourney v6 (kvikontent) | Latent diffusion | 0.9644 | 0.9884 | 0.9966 |
| Firefly Image2 | Commercial | 0.9615 | 0.9908 | 0.9643 |
| Midjourney V6.1 | Commercial | 0.9688 | 0.9947 | 0.9562 |
| Firefly Image3 | Commercial | 0.9830 | 0.9948 | 0.9610 |
| Kandinsky 2.2 | Latent diffusion | 0.9921 | 0.9982 | 0.9963 |
| LCM-LoRA SSD-1B | Latent diffusion | 0.9859 | 0.9983 | 0.9952 |
| Stable Cascade | Other | 0.9994 | 0.9998 | 0.9991 |
| **Overall** | | **0.8276** | **0.9644** | 0.9866 *(confounded)* |

Three things to read off this table:

1. **Modern latent-diffusion and commercial generators transfer for free** (0.96–0.9998), even though
   we never trained on any of them.
2. **The failures are architectural, not stylistic.** Hourglass is *pixel-space* diffusion (no VAE) and
   DFGAN is a *GAN*. Our training data is entirely diffusion-era latent models — the detector has never
   seen a GAN artifact or a non-VAE diffusion artifact in its life. The dominant modern detection cue is
   believed to be the **VAE decoder fingerprint**; generators without a VAE simply don't have one.
3. **One generator generalized across architecture families.** Adding MJv5 — a latent-diffusion model —
   raised DFGAN (a GAN) from 0.6880 to 0.9513 and Hourglass from chance to 0.7320. We did not expect
   that, and as far as we can establish from the literature, nobody has published this specific measurement (§10.2).

**This is the honest weakness, and we are going to state it in the pitch rather than hide it:** the
detector is excellent on the benchmark and on modern generators, and measurably weaker on architecture
families absent from its training data. We know exactly which ones and by how much.

---

## 8. What we shipped

`configs/frozen_siglip2_giant_mjv5.yaml` → `results/frozen_siglip2_giant_mjv5/head_best.pt` (**16 KB**)

| Metric | Value |
|---|---|
| Clean AUC (full 13,841) | 0.99936 |
| Clean AUC (deduplicated, 8,717) | 0.99941 |
| Worst cell | noise@0.02 — 0.99912 |
| Max degradation delta | 0.00026 |
| Mean degraded AUC | 0.99951 |
| Accuracy @ 0.5 | 0.99104 |
| Balanced accuracy @ 0.5 | 0.99147 |
| **Majority-class baseline** | **0.6545** |
| Alt-real AUC | 0.99957 |
| **Shortcut gap** | **−0.00018** |
| External (Community Forensics) | 0.9644 |
| Backbone params | 1.163659776 B (cap: 2 B) |

The entire trained artifact is **16 KB** on top of an off-the-shelf frozen encoder. Retraining the head
from cached features takes seconds.

---

## 9. Papers we read — what we took, what we left out

We used the literature two ways: to **avoid burning GPU time** on questions already answered, and to
**check our own claims** before putting them on a slide. Both directions produced results.

### 9.1 Papers that supported us — and what we took

| Paper | What it says | What we took |
|---|---|---|
| **Grommelt et al., "Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets"** — arXiv:2403.17608, ECCV 2024 W | Names exactly two canonical dataset biases: **JPEG compression and image size**. Audits GenImage: ImageNet reals ~450², Midjourney 1024², SD 512², GLIDE/ADM/VQDM 256², BigGAN 128². Their fix (matching real-image sizes to the generators') gains **+11.06 pp mean cross-generator accuracy, max +41.29 pp**. | **Took the whole diagnosis.** Our §7.4 resolution confound is their documented failure mode with the sign flipped. Their intervention (match size distributions across classes) is the fix we're running. Our "87% of reals flipped" magnitude is consistent with swings of their size. |
| **Bernabeu-Perez et al., "Present and Future Generalization of Synthetic Image Detectors"** — arXiv:2409.14128 | *"SIG model age emerges as the dominant factor affecting generalization performance."* Cross-generator recall: MJ 1/2 (2022) **54.55%**, DALL-E 3 (2023) 68.64%, MJ 5/6 (2023) **68.92%**. | **Took the vintage explanation** for §7.3. Their worst training source is early-vintage Midjourney; their best are MJ 5/6 and DALL-E 3. Our DALL-E 2 regression and MJv5 gain sit on their trend line — independent confirmation on a different axis. |
| **Cozzolino et al., "Zero-Shot Detection of AI-Generated Images"** — arXiv:2409.15875, ECCV 2024 | The only true real-source ablation we found (Fig. 6). COCO gives the best and most uniform results; real-source choice **interacts with the evaluation real distribution**. | **Took the caveat**: our LAION real-source delta (−0.0012) is eval-set dependent and must not be read as a general property. |
| **Cozzolino et al.** — arXiv:2312.00195 | Table 2: COCO + Latent Diffusion = 92.4 AUC vs LSUN + Latent Diffusion = 88.7. Cause: LSUN images *"have all the same size, and most of them are compressed with the same quality factor."* | **Our WildFake confound, published two years earlier.** Cited as prior art for §7.2/§7.4. |
| **Community Forensics** — arXiv:2411.04125, CVPR 2025 | 2.7 M images, 4,803 generators. §4.3 scales **generator count** 3→3333 at fixed 100K images. Separate volume curve: *"performance improves with more training images, [but] begins to plateau at approximately 27K."* | **Took the dataset** (as our external eval) **and the corrected claim**. See §9.3 — we had been citing this paper for the wrong thing. |
| **SSAFE** — arXiv:2606.08634 | Frozen encoder + single linear head — *our exact architecture*. Table 4 on OpenFake: 5K→100K training images gives 99.4→100.0 AUC, **flat**, and 30K beats an end-to-end 4M-image model. *"Massive data accumulation is unnecessary."* Also: MMD-based generator clustering beats random source selection (89.4% vs 86.3%). | **Took the validation** of our flat 4k/8k/12k curve — same shape, same scale, independent replication. **Took their §3.4 MMD screening method** as the principled successor to our brute-force source sweep (it's CPU-only for us — we already cache per-source features). |
| **Composite Data Augmentations for Synthetic Image Detection** — arXiv:2506.11490, EUSIPCO 2025 | **+22.53% mAP** from augmentation vs none. Best family triple by greedy search: JPEG + Gaussian blur + colour invert. | **Took it as the sourceable replacement** for a figure we could not source (§9.3). |
| **Ren et al.** — arXiv:2602.07814, Feb 2026 | *"Training data explains 20–60% performance variance within architectural families, often exceeding variance between different architectures"*; *"training data alignment with target generators critically outweighs architectural innovations."* Also: out-of-the-box detectors fall from ~79% on 2020-21 generators to **~38% on 2024+**. | **Took the headline thesis of our report.** Every architectural knob we tried moved ≤0.001; data decisions moved 0.07. This paper says exactly that, measured independently. |
| **Rajan & Ojha** — arXiv:2410.11835 | The dominant modern detection signal for SD-family images is the **VAE decoder fingerprint**. They explicitly scope out *"pixel-space diffusion models where the VAE is not available"* as future work. Also warn that augmentation applied at different effective scales per class **creates** confounds. | **Took the mechanism** that explains our Hourglass failure (§7.7): no VAE → no VAE fingerprint → chance-level detection. |
| **Corvi et al.** — arXiv:2211.00680, ICASSP 2023 | Spectral analysis of averaged noise residuals: strong periodic artifacts for Latent/Stable Diffusion, **markedly weaker for ADM and DALL-E 2**. | Supports the latent/pixel asymmetry without framing it as such. Closest empirical evidence for §7.7. |
| **BIAS-ID** — arXiv:2605.31153 | Formalizes per-class **Score Shift** aggregated into a Transform Sensitivity score, across resize/JPEG/WebP/rotation/grayscale over six detectors. | **Took the citation and dropped our novelty claim** (§9.3). Our one-class-downscale test is the resize row of their framework with an AUC readout. |
| **B-Free** — arXiv:2412.17671, CVPR 2025 | Same DINOv2 backbone: **linear probe 80.8 AUC vs end-to-end 99.0 AUC**. Random augmentation *reduces but does not eliminate* format bias. | **Took the limitation** (§9.4) and the confirmation that augmentation is not a substitute for fixing the training mix — which is what §7.2 found empirically. |
| **GAPL** — arXiv:2512.12982, CVPR 2026 | *"Benefit then Conflict"*: at fixed image count, raising generator count 1→8 eventually *decreases* accuracy as source heterogeneity grows. | **Took the caution**: per-source winners may not compose. Our screening must be followed by a combined-mix run, not treated as additive. |

### 9.2 Where the literature was silent — our possible contributions

Stated carefully, because "novel" is a claim we've already had to retract once.

| Our result | Literature status |
|---|---|
| **One latent-diffusion generator unlocking cross-architecture detection** (MJv5 lifting DFGAN 0.688→0.951, Hourglass 0.525→0.732) | **No direct precedent found.** Community Forensics reports the direction but attributes it to *thousands* of generators; we see it from one. |
| **A specific older generator being actively harmful** (DALL-E 2 costing 0.06 AUC) | **No precedent found.** Nearest is GAPL's aggregate "Benefit then Conflict", which is not "generator X hurts". |
| **Additive real-source ablation** ("add real source X to a fixed mix, measure the delta") | **No precedent found.** Every study we read swaps the real set wholesale or checks pairing. |
| **Latent-diffusion → pixel-diffusion transfer as a controlled split** | **Literature silent.** The mechanism (VAE fingerprint) is asserted by Rajan & Ojha and never measured; Community Forensics is *"entirely (or almost entirely) latent diffusion."* Our Hourglass numbers are the kind nobody has published. |
| Quantified same-domain-reals benchmark inflation | Universally acknowledged qualitatively (SSAFE, B-Free); never given a number. |

### 9.3 Papers that corrected us — claims we removed

**These are in the report on purpose.** Three claims we were repeating did not survive a proper read.

1. **"Augmentation is worth +5.8 AUC" — UNSOURCED. Removed.**
   This figure came into the project via the build spec and appears in our README. A full read of the
   **NTIRE 2026 challenge report (arXiv:2604.11487)** finds **no ablation table and no AUC delta for
   augmentation**, for the 5th-place method or any other. The only claim in the report is qualitative.
   *Action: remove from README before the pitch; the sourceable substitute is +22.53% mAP from
   arXiv:2506.11490.* (The same read did confirm our reproduction is faithful in every other respect:
   siglip2-giant-opt-patch16-384, squish resize, mean-pool over final-layer patch tokens, linear head,
   `distortion_prob=1.0`, ≤3 ops, 5 levels.)

2. **Our one-class-downscale control is not novel.** BIAS-ID (arXiv:2605.31153) formalizes the same
   idea. We present it as **convergent methodology with a citation**, not as an invention.

3. **We were citing Community Forensics for the wrong claim.** We had used it for "generalization
   scales with training volume." It doesn't say that — its headline curve holds volume fixed at 100K and
   scales **generator count**. Its actual volume curve *plateaus at ~27K*. This correction happens to
   support our flat data-size curve more strongly than the misreading did.

### 9.4 Where the literature contradicts us — unresolved

| Tension | Detail |
|---|---|
| **SDXL** | Bernabeu-Perez put SDXL in their *good* training-generator group. We measured it as negligible (+0.0007) and slightly negative stacked on MJv5. So "2023 vintage" alone does not predict our result — something separates MJv5 from SDXL that vintage doesn't capture (commercial aesthetic tuning? prompt distribution? native resolution?). **This is a live, uncontrolled confound in our own MJv5 result**, and the MJv4-vs-MJv5 experiment designed to resolve it is blocked on data access. |
| **Number of augmentation families** | arXiv:2506.11490: *"using more than three augmentations during training does not improve model performance and may even reduce effectiveness."* But NTIRE 2026 practice runs the other way — 36 transformation types, with top teams using multi-level pipelines. **Unresolved in published work.** Our 6-family table sits between the two. |
| **Frozen probes vs end-to-end** | B-Free: same backbone, linear probe 80.8 AUC vs end-to-end 99.0. Community Forensics Fig. 6a: *"freezing the backbone consistently leads to worse results."* SSAFE is the counterweight (frozen probe at SOTA), and the deciding variable appears to be **data curation quality** — which is exactly what we found. Still: **"we are benchmark-limited" may also be "frozen-probe limited", and we cannot distinguish those on this hardware.** |
| **Backbone choice** | Three independent 2026 papers rank other encoders above SigLIP2 as frozen feature sources (SSAFE: PE-Core-G14-448 first; "Simplicity Prevails": DINOv3 best on GenImage, PE best on Chameleon; TAP: PE-G14 > DINOv3-7B > SigLIP2-SO400M). NTIRE 2026 ranks 1, 2 and 4 all used DINOv3. **We could not test this meaningfully** — see §6.2. |

---

## 10. Limitations — stated plainly

1. **The designated benchmark is saturated.** At 0.9994 there is ~0.0006 of headroom, far below the
   noise floor for ~4,998 real images. It can no longer rank anything. Every remaining decision has to be
   made on the external eval.
2. **One external dataset is not a generalization bound.** Community Forensics-Eval alone cannot
   distinguish "our detector generalizes" from "that dataset is easier than advertised". A second set
   (RRDataset, CC-BY-4.0) is downloading; Chameleon is gated (academic-only, email request).
3. **The validation fakes are a single generator** (DALL-E 3, 2023), and its 8,843 files are only
   **3,719 unique images**. We report deduplicated AUC alongside the raw number.
4. **The alt-real control is structurally blind to resolution shortcuts** — both real sets are 200 px.
   Fixed by adding the one-class-downscale control, but any shortcut sharing that blind spot could still
   hide.
5. **Our reals are ImageNet, LAION and COCO — all in SSAFE's "outdated" category.** Their Table 3 shows a
   probe keeping 99.5–99.7% *fake* accuracy while collapsing on *real* accuracy against modern
   photography (SocialRF 41.8%, CommunityAI 66.0%, Chameleon 66.9%). We have not tested this failure mode.
6. **The MJv5 win is not fully attributed.** Vintage, commercial tuning, and 1024 px native resolution
   are all confounded in that one change. The MJv4-vs-MJv5 test that isolates vintage is designed and blocked.
7. **Frozen-probe ceiling.** We cannot rule out that end-to-end fine-tuning would beat us; LoRA is 22 days
   on this laptop.
8. **Data access.** Three planned experiments (#19 per-generator screening, #20 MJv4-vs-MJv5, #22
   pixel-space diffusion) are blocked by ModelScope throttling since 2026-08-29. Configs and methods are
   committed and ready to run when access returns.

---

## 11. What we would do next, in priority order

1. **Finish the resolution fix for GAN data** (#23, running). Add ~200 px COCO test2017 reals so low
   resolution appears on *both* sides of the label — Grommelt's intervention. The GAN arm's 0.9866
   external score is the best we've seen; if the confound is removable, that's the biggest available win.
2. **Close the pixel-space-diffusion gap** (#22) with DDPM/DDIM/ADM training data, applying the same
   resolution lesson. Hourglass at 0.7320 is our worst external cell.
3. **Second external eval** (#25) — RRDataset, then Chameleon if access is granted.
4. **MJv4 vs MJv5** (#20) — isolates vintage from brand/resolution and is the falsifiable test of our
   central data theory.
5. **SSAFE's MMD-based source screening** — CPU-only for us, and a strictly better method than our
   brute-force sweep for deciding which of WildFake's ~25 sources earn their place.
6. **Backbone comparison on the external eval** (not the saturated benchmark) — PE-Core, DINOv3.

Explicit non-goals: ensembles, fusion, and any architectural complexity. We have measured that this
category of change buys ≤0.001 in our setup, and the brief forbids it until the single-backbone
baseline is reported — which it now is.

---

## 12. Appendix

### 12.1 Reproduction
```bash
source .venv/bin/activate
python -m src.data.build_val --config configs/frozen_siglip2_giant_mjv5.yaml   # manifest + exclusion hashes
python -m src.train          --config configs/frozen_siglip2_giant_mjv5.yaml   # asserts no leakage first
python -m src.evaluate       --config configs/frozen_siglip2_giant_mjv5.yaml   # 15-condition grid + plots
python -m src.compare results/frozen_siglip2_giant results/frozen_siglip2_giant_mjv5 ...
python predict.py --image_dir <dir> --output preds.json                        # the deliverable CLI
```

### 12.2 Where the evidence lives
| Artifact | Path |
|---|---|
| Consolidated findings | `docs/FINDINGS.md` |
| Per-arm results | `results/<arm>/RESULTS.md`, `results/<arm>/eval/` |
| Cross-arm table | `results/comparison.csv` |
| Per-condition grids | `results/<arm>/eval/auc_grid*.csv` |
| Gate-probe output | `results/gate_probe.json` |
| Hyperparameter sweeps | `results/frozen_siglip2_giant_sidonly/sweep_{reg,norm,loss}.csv` |
| Full reasoning + literature | GitHub issues #1–#25 (each carries a results comment) |
| Shipped checkpoint | `results/frozen_siglip2_giant_mjv5/head_best.pt` (16 KB) |

### 12.3 Engineering notes worth repeating
- **Mac must stay awake** (`caffeinate -i`) — sleep silently pauses MPS jobs.
- **MPS hard-asserts on mixed-dtype matmul** — backbone emits bf16, heads are fp32; always `.float()`
  features before the head or you get `Abort trap: 6`.
- **Feature cache key** = (backbone, image_size, tag, sample-path list). It does *not* encode the
  training source mix, so different arms silently overwrite each other's caches. It also does not detect
  pixel changes. Back up before switching arms.
- **Archive member paths often do not match label-CSV paths** — SDXL and MJv5 both needed relocating
  after extraction.

### 12.4 Experiment index
25 tracked issues. Closed/resolved: #3 (format harmonization, null), #4 (regularization, defaults
optimal), #5 (normalization, none wins), #6 (focal, γ=5 marginal), #7 (gate probe, kills #2), #8 (clean
draw, hurts), #9 (data size, flat), #10 (generator diversity, **biggest win**), #11 (TTA, rejected), #12
(CLIP baseline, 0.9766), #15 (external eval, the real weakness), #17 (SID-only ship candidate), #18 (GAN
arm, confounded), #21/#23 (real-source screening × resolution fix, running). Deferred on measurability:
#1 (LoRA, 22 days), #13 (resolution), #14 (attention pooling), #2 (ensemble, killed by #7). Blocked on
data access: #19, #20, #22. In progress: #24 (official NTIRE distortion backend), #25 (second external set).
