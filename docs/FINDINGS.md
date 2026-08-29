# Findings — consolidated record

Working notes behind the numbers. Per-arm detail: `results/*/RESULTS.md`. Full reasoning and
literature: GitHub issues #1-#25 (each carries a results comment).

## Arm comparison

| Arm | Designated clean (full 13,841) | Worst cell | Max delta | External (Community Forensics) |
|---|---|---|---|---|
| Original spec mix | 0.9309 | resize@0.25 0.8097 | 0.1118 | — |
| Harmonized mix (#3) | 0.9316 | resize@0.25 0.8106 | 0.1113 | — |
| CLIP ViT-L/14 baseline (#12) | 0.9766 | jpeg@70 0.9732 | 0.0021 | — |
| SID-only (#17) | 0.9931 | crop@0.8 0.9895 | 0.0018 | 0.8276 |
| GAN-augmented (#18) | 0.9197 | resize@0.25 0.6531 | 0.2607 | 0.9866 (confounded) |
| **SID + LAION + MJv5 (ship)** | **0.9994** | **noise@0.02 0.9991** | **0.0003** | **0.9644** |

Ship candidate: `configs/frozen_siglip2_giant_mjv5.yaml`,
`results/frozen_siglip2_giant_mjv5/head_best.pt` (16 KB, committed).
acc@0.5 0.9900, balanced acc 0.9907, majority baseline 0.6545. Shortcut gap -0.0002.

## Per-source screening (same features, same head procedure, sources vary)

| Training sources | Designated clean |
|---|---|
| SID_Set only | 0.9932 |
| + WildFake LAION reals | 0.9891 |
| + WildFake DALL-E 2 fakes | 0.9691 |
| full original mix | 0.9316 |
| WildFake slice alone | 0.6234 |
| SID 12k only | 0.9949 |
| + LAION + SDXL | 0.9956 |
| **+ LAION + MJv5** | **0.9994** |
| + LAION + SDXL + MJv5 | 0.9991 |

Data-size curve (#9), SID_Set per draw: 4k 0.9945 / 8k 0.9949 / 12k 0.9949 — flat.

## External generalization, per generator (Community Forensics-Eval, 3,759 imgs)

| Generator | Architecture | SID-only | +MJv5 (ship) | GAN arm |
|---|---|---|---|---|
| Hourglass | PixDiff | 0.5247 | 0.7320 | 0.9683 |
| DFGAN | GAN | 0.6880 | 0.9513 | 0.9999 |
| Imagen3 | Commercial | 0.9765 | 0.9850 | 0.9513 |
| kvikontent_midjourney_v6 | LatDiff | 0.9644 | 0.9884 | 0.9966 |
| Firefly_Image2 | Commercial | 0.9615 | 0.9908 | 0.9643 |
| MidjourneyV6_1 | Commercial | 0.9688 | 0.9947 | 0.9562 |
| Firefly_Image3 | Commercial | 0.9830 | 0.9948 | 0.9610 |
| kandinsky_2_2 | LatDiff | 0.9921 | 0.9982 | 0.9963 |
| LCM_lora_ssd1b | LatDiff | 0.9859 | 0.9983 | 0.9952 |
| stable_cascade | Other | 0.9994 | 0.9998 | 0.9991 |
| **overall** | | **0.8276** | **0.9644** | **0.9866** |

## Resolution-shortcut control (one-class downscale)

Degrade only one class and re-score. A head using resolution as a cue collapses when only the
**reals** are downscaled.

| Manipulation | SID-only | **Ship (MJv5)** | GAN arm |
|---|---|---|---|
| both clean | 0.9913 | 0.9994 | 0.9137 |
| both resized 0.25x | 0.9943 | 0.9996 | 0.6531 |
| only FAKES downscaled | 0.9918 | 0.9994 | 0.9328 |
| only REALS downscaled | 0.9938 | 0.9996 | **0.6183** |
| reals flagged at 0.5, clean -> downscaled | — | 0.7% -> **0.7%** | 23% -> **87%** |

Source resolutions (median shortest side): WildFake GAN fakes **256** (91% under 384px);
validation COCO reals **200** (100% under 384); validation DALL-E 3 fakes **1024**; MJv5 **1024**;
LAION reals 713; COCO test2017 reals **200**; **alt-real ImageNet 200** — i.e. the alt-real control
shares the COCO resolution profile and therefore **cannot detect a resolution shortcut**.

## Rejected / negative results

- **Clean training draw (#8)**: 2 degraded 0.9932 / 2 degraded + 1 clean **0.9924** / 3 degraded
  **0.9938**. Clean images hurt at equal volume. Validates `distortion_prob = 1.0`.
- **TTA (#11)**: +0.0002 for 2x cost. Views correlate r = 0.996 — model already hflip-invariant.
- **Normalization (#5)**: none 0.9932 > standardize 0.9917 > L2 0.9856.
- **Regularization (#4)**: 40-config sweep; configured defaults already at the optimum.
- **Focal loss (#6)**: gamma 5 = 0.9939 vs CE 0.9932; collapses at gamma 20 (0.9888).
- **Gate probe (#7)**: 7-class family 0.6247; clean-vs-degraded 0.8817 vs a 0.9333 majority baseline.
- **LoRA (#1)**: measured 0.06 img/s (~260 s/step at batch 16) -> ~18 h/epoch, ~22 days for 30 epochs.

## Literature verdicts (see issue comments for citations)

| Our finding | Verdict |
|---|---|
| Resolution shortcut | **Supported** — Grommelt "Fake or JPEG?" ECCV 2024 W; debiasing worth +11 to +41 pp |
| PNG/JPEG format confound | **Supported** — best-documented confound in the field |
| Newer/higher-tier generators win | **Supported** — Bernabeu-Perez 2409.14128, "model age is the dominant factor" |
| SDXL negligible | **Partly contradicted** — that paper puts SDXL in its *good* group |
| Volume saturates | **Supported** — CF's scaling claim is generator *count* at fixed volume; CF volume plateaus ~27K; SSAFE flat 5K-100K |
| Data > architecture | **Supported** — training data explains 20-60% of variance |
| One generator unlocks cross-architecture detection | **Apparently novel** — CF reports the direction but attributes it to thousands of generators |
| A specific old generator being harmful | **No precedent found** |
| Additive real-source ablation | **No precedent found** |
| Latent -> pixel-diffusion transfer | **Literature silent** — mechanism asserted, never measured |

### Corrections to our own claims
- **"+5.8 AUC from augmentation" is UNSOURCED.** Not in the NTIRE report (no ablation table). Came via
  the build spec. **Remove from README before the pitch.** Sourceable alternative: +22.53% mAP
  (arXiv:2506.11490, EUSIPCO 2025).
- **The one-class-downscale control is not novel** — BIAS-ID (arXiv:2605.31153) formalizes it. Cite it.
- **Frozen probes may cap below end-to-end** — B-Free: same backbone, linear probe 80.8 AUC vs
  end-to-end 99.0. So "benchmark-limited" may also be "frozen-probe limited"; we cannot separate these
  on this hardware.

## Known limitations
- Designated benchmark is saturated (0.9994, ~0.0006 headroom) — it can no longer rank anything.
- One external set only; RRDataset (CC-BY-4.0) pending. Chameleon is gated (email, academic-only).
- Validation fakes are a single generator (DALL-E 3, 2023); its 8,843 files are only 3,719 unique images.
- Alt-real control cannot detect resolution shortcuts (both real sets are 200px).
- #19/#20/#22 blocked: ModelScope throttling all requests since 2026-08-29 afternoon.
