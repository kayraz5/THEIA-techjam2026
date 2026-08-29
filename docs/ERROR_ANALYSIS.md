# Error Analysis Note

Shipped detector: `configs/frozen_siglip2_giant_ship.yaml` → `results/frozen_siglip2_giant_ship/head_best.pt`.
Evaluated on the designated validation subset (2,000-image seeded grid subset: 691 COCO reals, 1,309 DALL-E 3
fakes; clean AUC on the full 13,841 = 0.9991). Contact sheets: `results/frozen_siglip2_giant_ship/eval/errors_fp.png`
and `errors_fn.png`; ranked list: `errors_top.csv`.

## 1. False positives — real photos scored as AI

**The 20 most confident false positives across all 15 conditions come from only 11 distinct photographs**, and
four of them account for 14 of the 20 slots:

| Image | Appears | Conditions | What it is |
|---|---|---|---|
| `img159690` | **7×** | blur 1.0/2.0, crop, jpeg 30, noise 0.02, resize 0.25/0.5 | a flat graffiti-style sticker on a plain white background |
| `img162789` | 3× | blur 2.0, jpeg 30, noise 0.1 | high-contrast black-and-white studio portrait |
| `img163644` | 2× | blur 2.0, resize 0.25 | zebras against a uniform grass background |
| `img159456` | 2× | blur 2.0, crop | a bird on a branch, shallow depth of field |

Three patterns, in decreasing importance:

1. **Graphics-like reals.** The dominant FP is a *photograph of a rendered graphic* — a sticker with flat
   colour fills and a featureless white ground. It has almost none of the sensor noise, lens blur, or
   texture statistics the model associates with a camera. Scored 0.99+ under seven different degradations
   and is essentially a label-noise case: it is a real photo of something that was itself digitally made.
2. **Studio / clinical photography.** Monochrome portraits and product-style shots with controlled lighting,
   shallow depth of field, and clean backgrounds. These share surface statistics with generated images
   (smooth gradients, low sensor noise) more than with typical COCO snapshots.
3. **Degradation pushes borderline reals over.** Every FP in the top 20 is under a *harsh* condition —
   blur σ=2.0, resize 0.25×, noise 0.1, or JPEG q30. No clean-condition real appears. Heavy low-pass damage
   removes exactly the high-frequency camera texture that distinguishes categories 1–2 from generated
   imagery, so images that were already borderline tip over.

**Operating-point consequence.** At threshold 0.5 the false-positive rate on COCO reals is **1.16%**
(8 of 691). The threshold table (`eval/thresholds.csv`) gives the trade:

| Target | Threshold | FPR (reals) | TPR (fakes) |
|---|---|---|---|
| default | 0.500 | 1.16% | 97.78% |
| **FPR ≤ 1%** | **0.639** | 0.87% | 97.63% |
| FPR ≤ 0.1% | 0.945 | 0.00% | 94.88% |
| FPR ≤ 5% | 0.127 | 4.78% | 99.39% |

For a moderation use-case where flagging a real user's photo is the costlier error, **0.64 is the better
default** — it removes a quarter of false positives for a 0.15-point drop in recall.

## 2. False negatives — AI images scored as real

**The FN sheet is dominated by a single image**: a DALL-E 3 render of a convention-hall cosplay scene
(inflatable unicorn, Wonder Woman, Darth Vader, Stormtroopers) appears in **13 of the 20 slots** — scored
0.009–0.031 under *every* condition including clean. The remaining slots are a PS3 game-box product shot
(2×), an art-gallery interior (2×), a graffiti wall, and a toy soldier holding a globe.

What these have in common is not a degradation — it is **content**:

- Every one is a photorealistic rendering of a **mundane, cluttered, real-world scene** — a trade-show
  floor, a retail shelf, a gallery wall. These are the kinds of pictures phones take, not the kinds of
  pictures people usually *prompt for*. DALL-E 3 reproducing "boring phone snapshot" is its hardest mode to
  detect because the model's positive class was learned largely from aesthetic, high-fidelity generations.
- The scores are **flat across conditions**: the unicorn image is 0.016 clean and 0.011–0.031 degraded.
  **This is a content failure, not a robustness failure.** Degradation does not create these misses; it is
  irrelevant to them.
- Several contain **legible text** (FLY SWATTER, HOY, gallery signage), a cue that older detectors used
  against generators. DALL-E 3 renders text well enough that it no longer helps — and may actively hurt, if
  the model has learned "coherent text → real".

**Consequence.** Recall at 0.5 is 97.8%. The residual 2.2% is not evenly spread: it is a small set of
images that the detector is *confidently* wrong about, and no threshold below ~0.03 recovers them without
flagging half the real set. These are the images that would need more training data of that *content
type*, not more augmentation.

## 3. Trade-offs in the approach — stated plainly

| Decision | What we gained | What it cost |
|---|---|---|
| **Shipped arm D over the benchmark-optimal arm** (see `docs/REPORT.md` §7.8) | External generalization 0.9715 → **0.9917** (Community Forensics) and 0.9318 → **0.9732** (RRDataset); DFGAN 0.964 → 0.9996; Hourglass 0.751 → 0.932 | Designated clean AUC 0.9994 → 0.9991; worst-cell 0.9991 → 0.9951; accuracy@0.5 0.991 → 0.985 |
| **Adding GAN training data + low-resolution reals** to fix the resolution confound | GAN arm usable (was flagging 69.5% of downscaled reals; now 5.1%) | **A 5.1% residual remains** — 1.2% → 5.1% of reals flagged when only reals are downscaled. Reduced ~13×, not eliminated. A deployment seeing heavily thumbnailed real photos should expect a raised FP rate |
| **Threshold calibration shift** | — | The 0.5 operating point moved; acc@0.5 fell 0.6 points while AUC held. Re-tune from `thresholds.csv` for any fixed-threshold deployment |
| **Frozen backbone + 16 KB linear head** | Retrains in seconds; ~10 img/s on a laptop; every experiment reproducible from cached features | Cannot learn new *features*, only re-weight existing ones. Literature (B-Free) shows end-to-end can beat frozen probes by wide margins; LoRA measured at ~22 days on this hardware, so untested here |
| **Six-family degradation table over the official 12-family pipeline** (#24) | Better OOD generalization (+0.0046 CF, +0.0151 RRDataset) | Not directly comparable to published NTIRE numbers, which used the official augmentations |
| **Training data all diffusion-era + GAN; no pixel-space diffusion in the shipped arm** | Clean benchmark, minimal resolution residual | Hourglass (pixel-space diffusion, no VAE) remains the worst external cell at 0.932. Adding ADM/DDIM/DDPM lifts it to 0.979 but raises resolution sensitivity to 12% — shipped as an *option*, not the default (#22) |

## 4. What the errors say about the next step

- **FPs are a photography-style problem**: graphics-like and studio-style reals. More real-image *domain
  diversity* (product photography, illustrations, monochrome) is the fix — not more augmentation, since
  augmentation is what pushes them over.
- **FNs are a content problem**: photorealistic renders of mundane scenes. More *fake* data of that
  content type is the fix — and this is precisely the WildFake/SID_Set gap, since both skew toward
  aesthetic generations.
- **Neither is a robustness problem.** Across the 15 conditions the worst cell is 0.9951 and the maximum
  degradation drop is 0.0039. The detector's remaining errors are the same images clean and degraded.
