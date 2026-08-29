# Demo video — shot list and narration (target 2:30–3:00)

Record with the terminal at a large font. One take per scene is fine; cut between them.
Everything shown is real output — nothing is mocked. Run `bash scripts/demo.sh` once before recording
so the model weights are cached and the demo runs in ~1 minute.

| # | Scene (what's on screen) | Narration (say roughly this) | ~sec |
|---|---|---|---|
| 1 | Title card / README header | "AI-generated images don't arrive pristine — they arrive compressed, resized, blurred, screenshotted. We built a detector that holds up under all of that, and we built it on one laptop." | 15 |
| 2 | `ls demo_images/` — 24 files, names visible | "Twelve held-out images the model has never seen: six real COCO photos, six DALL-E 3 renders. Each one clean, and each one under a harsh degradation — JPEG quality 30, 4× downscale, heavy blur, noise." | 20 |
| 3 | `python predict.py --image_dir demo_images --output demo_preds.json` running; param-check line + progress bar + `img/s` | "This is the deliverable. It prints the backbone's parameter count — 1.16 billion, asserted under the 2-billion cap — and scores the folder at about ten images a second on this MacBook." | 20 |
| 4 | `cat demo_preds.json` (first few entries) | "Output is the required JSON: image path and a confidence that it's AI-generated." | 10 |
| 5 | Scoreboard from `scripts/demo.sh` — 22/24, with `ai_3` missed both clean (0.08) and noisy (0.18) | "Degraded and clean versions score almost identically — that's the whole point. And look at the one it gets wrong: a DALL-E render of a guy in a t-shirt. It misses it clean *and* degraded, at the same confidence. The damage isn't what breaks it — the content is. That's the honest failure mode, and it's in our error analysis." | 30 |
| 6 | `docs/ROBUSTNESS_SUMMARY.md` table or `eval/auc_grid.png` heatmap | "Across all fifteen conditions on the full benchmark: worst cell 0.995, biggest drop from clean 0.004. The recipe we started from lost 0.11 on downscaling." | 20 |
| 7 | `docs/REPORT.md` §7.8 table (A/B/C/D) | "We could have shipped 0.9994. We shipped 0.9991 — because the benchmark is saturated and can't rank arms anymore, and on two independent external sets the arm we shipped wins by two to four points. We optimized the number we could still measure." | 30 |
| 8 | `python scripts/resolution_control.py` output | "We also ship our own failure controls. This one downscales only the real photos: the broken arm flagged 70% of them as AI; ours flags 5%. Not zero — and we say so." | 20 |
| 9 | `ls -la results/frozen_siglip2_giant_ship/head_best.pt` → 16 KB; GitHub Issues list | "The entire trained artifact is sixteen kilobytes on top of a frozen open-weight encoder. Twenty-five experiments, every one logged with its result on a GitHub issue — including the ones that falsified our own theories." | 20 |
| 10 | README deliverables table | "Code, README, robustness summary, error analysis and this video are all in the repo. Thanks." | 10 |

**Do not include**: any third-party logo, the Devpost/TikTok branding, or images from Community Forensics
(CC BY-NC-SA). The demo folder uses only COCO (CC BY 4.0) and WildFake (Apache 2.0) images.

**If a scene fails live**: every command's output is also on disk under `results/frozen_siglip2_giant_ship/eval/`
and `results/frozen_siglip2_giant_2x2/` — show the file instead.
