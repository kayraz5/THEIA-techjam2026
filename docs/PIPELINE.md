# Pipeline — how an image enters the detector

Traced from `src/train.py`, `src/features.py`, `src/evaluate.py`, `src/data/registry.py`,
`src/degradation/transforms.py`. Rendered version (for slides):
see the *Detector Ingestion Paths* artifact.

Training and evaluation share the resize, the frozen backbone and the transform table. They differ in
**exactly one thing**: training applies *random stacked* degradations, evaluation applies *one fixed*
degradation at a time.

---

## 1. Training ingestion (`mode: frozen`)

```mermaid
flowchart TD
    A["<b>build_train(cfg)</b><br/>SID_Set parquet → PNG · WildFake label CSVs<br/><i>real_coco filtered to /test2017/ · dalle3 refused outright</i>"]
    B{"<b>ExclusionList.assert_disjoint()</b><br/>pixel-content hashes of every held-out image<br/><b>aborts the run on one overlap</b><br/>shipped: 19,500 vs 9,717 hashes → 0"}
    C["<b>RandomDegradation</b> — 1–3 distinct families,<br/>independent severities, then hflip p=0.5<br/><b>distortion_prob = 1.0 → no clean images</b><br/>reseeded per (draw, index) = random but reproducible"]
    D["<b>squish_resize(img, 384)</b><br/>stretched to 384², aspect ignored, <b>no crop</b>"]
    E["<b>SigLIP2-giant vision tower — FROZEN</b><br/>576 patch tokens × 1,536 → mean-pool → 1,536-d<br/>1.164 B params, asserted &lt; 2 B, torch.no_grad()"]
    F[("<b>.npz feature cache</b><br/>feats · labels · plans · paths")]
    G["<b>LinearHead 1,536 → 2</b><br/><b>1,537 trainable parameters</b><br/>AdamW · cosine + warmup · weighted sampler · 30 epochs"]
    H["per-epoch validation on 3 held-out feature sets:<br/>val_clean 13,841 · valsub_deg 2,000 · altreal_clean 1,000<br/><b>checkpoint on DEGRADED val AUC, never clean</b>"]
    I["<b>head_best.pt — 16 KB</b><br/>+ train_history.json, train_summary.json"]

    A --> B --> C --> D --> E --> F
    F -.->|"repeat K=2 independent draws"| C
    F --> G --> H --> I

    style B stroke:#9d3232,stroke-width:2px
    style C stroke:#8a5a12,stroke-width:2px
    style E stroke:#1f6b86,stroke-width:2px
    style G stroke:#8a5a12,stroke-width:2px
```

**Cost:** 19,500 images × 2 draws = 39,000 forward passes at ~10 img/s — paid once. Afterwards every
head retrain is seconds of CPU, which is what made ~30 ablations affordable on a laptop.

**Why the leakage gate sits before extraction:** it runs on every run with no flag to skip it, so a
violation costs zero GPU time to discover and cannot be forgotten.

---

## 2. Evaluation ingestion

```mermaid
flowchart TD
    A["<b>held-out sets — never trained on</b><br/>designated validation 13,841 (4,998 COCO val2017 + 8,843 DALL-E 3)<br/>alt-real control 1,000 (WildFake ImageNet, non-COCO)"]
    B["<b>eval_subset()</b> — seeded 2,000 for the grid<br/><i>full 13,841 still scored clean, and again deduplicated<br/>(only 3,719 of 8,843 DALL-E files are unique)</i>"]
    C["<b>apply_transform(img, name, level)</b><br/><b>ONE transform at ONE severity — never stacked, never random</b><br/>clean · jpeg 90/70/50/30 · blur .5/1/2 · resize .5/.25<br/>noise .02/.05/.1 · colour ±.2 · crop .8"]
    D["<b>squish_resize(384) → frozen SigLIP2-giant → mean-pool</b><br/><i>byte-identical to the training path;<br/>caches shared across every arm on the same backbone</i>"]
    E["<b>score twice</b><br/>AUC vs COCO reals = the benchmark number<br/>AUC vs ImageNet reals = the shortcut check"]
    F["auc_grid.csv/.png · thresholds.csv · roc.png<br/>fpr_vs_threshold.png · errors_fp/fn.png · summary.json"]

    A --> B --> C --> D --> E
    E -.->|"repeat for each of 15 conditions"| C
    E --> F

    style C stroke:#5b4b8a,stroke-width:2px
    style D stroke:#1f6b86,stroke-width:2px
    style E stroke:#5b4b8a,stroke-width:2px
```

**The shortcut check:** a detector that learned *"looks like COCO → real"* scores far better against
COCO reals than against ImageNet reals. **A large gap is a shortcut, not a success.** Shipped arm: +0.0011.

Two further evaluations reuse this path with different inputs — `scripts/eval_external.py` (Community
Forensics-Eval, RRDataset) and `scripts/resolution_control.py` (degrade only one class and re-score).

---

## 3. The one difference that matters

| | Training | Evaluation |
|---|---|---|
| transforms per image | **1–3, stacked** | **exactly 1** |
| severity | randomly drawn per family | fixed, one grid cell |
| randomness | random, seeded per (draw, index) | deterministic |
| clean images seen | **none** (`distortion_prob = 1.0`) | yes, as its own condition |
| horizontal flip | p = 0.5 | never |
| passes over the data | 2 independent draws | 15 conditions |
| **transform definitions** | **the same table — `src/degradation/transforms.py`, imported by both** ||
| **resize · backbone · pooling** | **identical code path — squish 384², frozen SigLIP2-giant, mean-pool** ||

Sharing one transform table means a robustness score cannot be an artifact of training and evaluation
drifting apart. The trade: the grid is not *independent* of the augmentation — it measures robustness to
the damage we chose to train against, which is why the **external** sets carry the generalization claim
(`docs/REPORT.md` §7.7).

---

## 4. The path we did not ship (`mode: lora`)

End-to-end LoRA changes Diagram 1 in one structural way: **the feature cache disappears.** The backbone
updates, so features cannot be reused and every epoch re-runs all 19,500 images with gradients.

Benchmarked rather than assumed: **~260 s/step at batch 16 = 0.06 img/s → ~18 h/epoch → ~22 days for 30
epochs.** On MPS the backward pass costs ~170× the forward, not the usual ~2×. Rejected on measured cost;
benchmark committed as `scripts/lora_bench.py`.
