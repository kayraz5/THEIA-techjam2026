# `project_demo/` — live AI-content detection in a vertical video feed

A local web demo of the detector in the place the problem actually lives: a short-video feed.
Scroll the feed, and the browser samples frames from whichever clip is playing, POSTs them to a
local server running the **real** SigLIP2-giant backbone plus the shipped linear head on MPS, and
raises a notification pill when a clip reads as AI-generated.

Nothing is precomputed. Every number on screen came back from a forward pass that happened after
you scrolled to that clip, through the same code path `predict.py` uses.

## Run it

```bash
source .venv/bin/activate
uv pip install --python .venv/bin/python -r project_demo/requirements.txt

bash scripts/demo.sh                              # once: builds demo_images/ for the self-test
python project_demo/fetch_videos.py               # ~5 min, ~600 MB of source media
python project_demo/preflight.py                  # score every clip offline first
caffeinate -i python project_demo/server.py       # ~6 s model load, then SERVER READY
open http://127.0.0.1:8000
```

`caffeinate -i` matters: sleep silently pauses MPS jobs.

## What it proves, and what it does not

**Proves the pipeline is real.** At startup the server scores `demo_images/ai_0_clean.jpg` and
compares against the value committed in `demo_preds.json` — `0.99720299243927`. The result is
shown as a chip in the UI. `/api/health` reports the checkpoint path, its sha256, the feature
dimension, device and dtype, and the `[param-check] … 1.164B` line prints on the console.

**Does not prove the detector handles AI video.** It was trained on still-image generators, with
zero video frames and zero temporal-VAE outputs. The measured result on this feed:

| | flagged | |
|---|---|---|
| AI clips | **3 / 5** | both RunwayML and VideoPoet caught; **both Veo clips missed** (median p(AI) 0.11, 0.15) |
| Real clips | **4 / 5 clean** | the stock kitchen clip peaks at median-5 **0.950**, above the 0.9446 trigger, and **has fired during live playback** |

The Veo misses are the point, not a bug to hide. `docs/REPORT.md` §7.7 found that this detector's
failures are *architectural, not stylistic* — generators whose decoder differs from anything in
training simply do not carry the fingerprint it learned. A video model's temporal VAE is exactly
that case, and the demo shows it happening live rather than asserting it in a table.

The real-side split is just as sharp and equally predicted: the three handheld Wikimedia clips
score 0.0001–0.003, while the two Pexels stock clips score 0.030 and **0.913**. Graded, denoised
stock footage is the ERROR_ANALYSIS §1 false-positive profile, and it is in the feed on purpose —
so the demo shows a false positive rather than hiding one. `preflight.py` reports a `headroom`
column for exactly this: a real clip whose peak median-5 comes within 0.05 of the trigger is a
live false alarm waiting for the right segment, even if it did not latch during the offline sweep.

## Files

| | |
|---|---|
| `detector.py` | model lifecycle: one backbone, four heads, `score_jpeg_bytes()`, startup self-test |
| `server.py` | FastAPI: static, byte-range video, `/api/health`, `/api/feed`, `/api/score`, `/api/head` |
| `fetch_videos.py` | manifest → download → ffmpeg normalise → ffprobe/cropdetect validate |
| `preflight.py` | score every clip offline, emit `preflight.csv` + `preflight_traces.json` |
| `videos.json` | the manifest: URLs, labels, generators, licences, and the pre-registered selection criteria |
| `static/` | vanilla HTML/CSS/JS, no build step |

## Layout

The page is deliberately **single-viewport — it never scrolls**; only the feed inside the phone
does. The phone is drawn at iPhone 17 Pro Max proportions (440x956 pt, aspect 0.4603) with a
working Dynamic Island: the notification expands out of the island and the island hides while it
is up. The live reading (score, median-5, EMA, badge state, latency, fps, dropped frames,
sparkline, all four head scores, and the two preprocessing thumbnails) is always visible; the
detail below it is tabbed — **Pre-flight**, **Runtime**, **What it can & can't do** — so no block
is ever taller than its slot. Verified at 1280x800, 1512x862 and 1920x1080 with zero page scroll
and zero overflow in any tab. Below 1080 px wide or 620 px tall the layout unlocks and scrolls
rather than clipping.

`L` (or the `truth` button) toggles ground-truth labels on the cards.
| `videos/`, `videos_raw/` | **gitignored.** Media is fetched from the original hosts, never redistributed |

## Design decisions that are load-bearing

These were measured, not guessed. Changing them changes the answer.

- **JPEG capture quality is pinned at 0.95.** At q60 a known fake fell from 0.983 to **0.511** — a
  would-be miss. Never lower it to save bandwidth; on localhost the bandwidth is free anyway.
- **Crop to fill, never pad.** Letterboxing a clip to 9:16 cost up to **0.276** of score
  (0.983 → 0.707). The backbone mean-pools 576 patch tokens, so black bars are flat, noise-free
  tokens averaged into the representation — the exact false-positive signature. Enforced in three
  places: ffmpeg `crop` (not `pad`), full-frame canvas `drawImage`, and CSS `object-fit: cover`.
- **Frames are sent at native resolution**, and the server runs the shipped `squish_resize`.
  Resizing on the canvas first measured equivalent (|Δ| ≤ 0.002) but substitutes the browser's
  resampler for PIL's `BICUBIC`, which would stop it being the shipped preprocessing.
- **Every clip is equalised through a common 768 px intermediate** before the identical upscale to
  720×1280. Without it, every AI clip was upsampled and most reals downsampled — a confound on
  precisely the axis `docs/REPORT.md` §7.4 flags as unfixed. `fetch_videos.py --verify` prints the
  per-clip factor and it must stay ≤ 1.00 across both labels.
- **Threshold 0.9446**, from `results/frozen_siglip2_giant_ship/eval/thresholds.csv`
  (`@FPR<=0.001`). That file reports 0.00% FPR at this point, but that is in-distribution: on reals
  the model never trained on, the measured FPR is **1.23%** (Community Forensics) and **2.93%**
  (RRDataset). The UI states both pairs. Never print "0% false positives".
- **Median-of-5 with hysteresis** (raise ≥0.9446 ×2, clear ≤0.70 ×5, sticky 2.5 s). Median rather
  than EMA because a single hard-cut frame spiking to 1.0 must not fire the pill on a real clip.
  Aggregation here is **display smoothing, not error reduction** — the UI says so.
- **One frame in flight, ever.** A busy request is dropped, never queued: a queued frame is a stale
  answer displayed as a live one.

## Clip selection

Clips are chosen by the **pre-registered content criteria** in `videos.json`, never by score —
selecting clips that happen to score well is the error `docs/REPORT.md` §7.3 spends a section
retracting. The one post-pre-flight change to the shipped set is disclosed in `videos.json`
(`ship_set_changes`) and on screen, with both clips' scores.

Run `python project_demo/preflight.py --all` to score all 18 candidates across five generator
families, including the ones excluded from the feed.

## Licensing

No media is committed or redistributed. `videos.json` records where each clip comes from and
`fetch_videos.py` pulls from the original hosts at run time. AI clips are CC BY 4.0 (DeepAction
v1); real clips are CC BY / CC BY-SA (Wikimedia Commons) and the Pexels License. See
`docs/DATA_LICENSES.md`.

The UI carries no TikTok, Devpost or third-party branding, per `docs/DEMO_SCRIPT.md`.
