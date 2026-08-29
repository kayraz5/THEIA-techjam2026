# Devpost submission — written project description (draft)

> Paste-ready. Sections follow the brief's required list. Fill the two `[ ]` placeholders (team, video link).

---

## Robust AI-Generated Image Detection Under Real-World Transformations

### The problem, as we came to understand it

Detectors of AI-generated images are usually benchmarked on pristine files. Real images arrive compressed,
resized, blurred, re-shared and screenshotted — and a detector that collapses under a JPEG re-encode is
useless for moderation. The brief asks for a detector that holds up across six transform families
(JPEG, blur, resize, noise, colour, crop) at multiple severities.

We reproduced a competition-grade recipe — the NTIRE 2026 5th-place configuration — and it scored **0.931
ROC-AUC** on the benchmark, losing **0.11** under 4× downscaling. Three weeks later we ship **0.9991**, with
a worst-case drop of **0.004**. We never changed the model, the architecture, or a single training
hyperparameter. Every point came from fixing the training data — and we can prove that, because we
measured both.

Along the way we caught two shortcuts the model was exploiting instead of detecting fakes, falsified two
of our own hypotheses (including our headline explanation), and found that the real frontier is not
robustness to damage but generalization to generator architectures the model never saw.

### How the solution addresses the problem

**Architecture.** `image → squish-resize 384² → SigLIP2-giant vision tower (1.164 B params, frozen) → mean-pool
over 576 patch tokens → 16 KB linear head → p(AI-generated)`. The backbone is asserted under the 2 B
parameter cap at every startup. The entire trained artifact is a 16 KB linear layer.

**Robustness by training.** Every training image receives 1–3 random distortions from the same table used
for evaluation (`distortion_prob = 1.0`). We tested this: adding clean images to training *hurt*.

**Data curation as the method.** The spec's training mix contained a 2022 generator whose slice alone scored
0.62 on validation — adding it made the detector worse. Per-source ablation on cached features found it in
minutes. Adding one modern generator (MJv5, 3,000 images) beat tripling the training volume. Adding GAN data
collapsed the detector until we discovered it had learned "low resolution → fake" (GAN fakes are 256 px,
benchmark reals are 200 px); adding low-resolution *real* photos alongside fixed it — and neither change
works alone.

**The decision that defines the project.** A benchmark-optimal arm exists at 0.9994. We shipped 0.9991
instead, because the designated benchmark is saturated (~0.0006 of headroom, below the noise floor for
4,998 real images) and can no longer rank arms — while on two independent held-out sets the shipped arm
wins: **0.9917 vs 0.9715** (Community Forensics-Eval, 10 unseen generators) and **0.9732 vs 0.9318**
(RRDataset). We traded 0.0003 of a number we could no longer measure for gains on two we could.

**Honesty controls built into the pipeline.** A pixel-content-hash exclusion list is asserted on every
training run so validation data cannot leak. Every AUC is reported against two real-image sources so a
"looks like COCO → real" shortcut would show as a gap (ours: +0.001). A one-class-downscale control catches
resolution shortcuts the standard check is blind to. Accuracy is never reported without balanced accuracy
and the 0.6545 majority baseline.

### Results

| | Spec as written | **Shipped** |
|---|---|---|
| Clean ROC-AUC (13,841 imgs) | 0.9309 | **0.9991** |
| Worst transform | resize 0.25× → 0.8097 | noise σ=0.1 → **0.9951** |
| Max degradation drop | 0.1118 | **0.0039** |
| External — Community Forensics-Eval | — | **0.9917** |
| External — RRDataset | — | **0.9732** |
| Trained parameters | — | **1,537** (16 KB) |
| Inference throughput (laptop, MPS) | — | ~10 img/s |

25 tracked experiments, each with a results comment on its GitHub issue. Full record: `docs/REPORT.md`.

### Development tools

VS Code · Claude Code · Git/GitHub (Issues as the experiment log) · uv (Python 3.12 venv) · pytest ·
macOS / Apple M5 Pro (64 GB unified memory, MPS backend) — the entire project ran on one laptop.

### Models

- **`google/siglip2-giant-opt-patch16-384`** — vision tower only, frozen, 1.164 B params (Apache 2.0). The
  shipped backbone.
- `google/siglip2-so400m-patch14-384` — declared fallback under the 2 B cap; never triggered.
- `openai/clip-vit-large-patch14` — baseline comparison arm (0.9766 designated).
- One trained component: a 1,536 → 2 linear layer.

### Libraries and frameworks

PyTorch 2.13 · Hugging Face Transformers 5.15 · PEFT 0.20 (LoRA arm, benchmarked and deferred) · scikit-learn
(ROC-AUC, logistic-regression probes) · NumPy 2.5 · pandas · Pillow · Matplotlib / Seaborn · tqdm ·
huggingface_hub · kornia (only for the vendored official NTIRE distortion pipeline, ablation #24) · pytest.
Exact versions in `requirements-lock.txt`.

### Datasets and assets

| Dataset | License | Role |
|---|---|---|
| SID_Set (Hugging Face) | CC BY 4.0 | training reals + full-synthetic fakes |
| WildFake (ModelScope, 1.29 TB — never downloaded whole; seeded slices via HTTP-Range) | Apache 2.0 | training slices (LAION + COCO test2017 reals, MJv5, six GAN families); designated validation subset (COCO val2017 + DALL-E 3); ImageNet alt-real control |
| Community Forensics-Eval (CVPR 2025) | CC BY-NC-SA 4.0 | **evaluation only** — 3,759 imgs, 10 unseen generators |
| RRDataset (Zenodo) | CC BY 4.0 | **evaluation only** — 3,000 imgs, balanced |

No dataset is redistributed. Full audit: `docs/DATA_LICENSES.md`.

### What we learned that we did not expect

1. **The benchmark stopped measuring anything at 0.999**, and worse, it ranks training generators backwards
   relative to generalization — its 2023 fakes reward 2023 training data regardless of transfer.
2. **A real-image source is not "more data"; it is a debiasing instrument.** COCO test2017 reals alone hurt
   generalization by 0.006. Paired with 256 px GAN fakes they unlocked +0.02. Neither works alone.
3. **The "overfitting" we set out to regularize was the model fitting a data confound.** On clean data,
   training longer strictly helps. We nearly shipped a symptom-level fix for a data-level problem.
4. **One modern latent-diffusion generator in training lifted detection of a GAN from 0.69 to 0.95** — a
   cross-architecture transfer we could not find a published measurement of.

### Limitations

Pixel-space diffusion (Hourglass) remains the weakest external cell at 0.932. A 5.1% residual resolution
sensitivity remains (down from 69.5% in the unfixed arm). The 0.5 threshold needs re-tuning for a fixed
operating point. The frozen probe may cap below end-to-end fine-tuning, which was infeasible on our
hardware. Full list and what we would do next: README § Limitations.

### Team

[ ] *names and contributions*

### Links

- Code: [ ] *public GitHub URL*
- Demo video: [ ] *YouTube URL*
