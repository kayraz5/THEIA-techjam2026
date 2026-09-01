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

**Does not prove the detector handles AI video, and the shipped feed is curated.** The feed scores
**5/5 AI flagged, 0/5 reals firing** — but candidates the detector got *wrong* were removed from it
on purpose. Read that 5/5 as a demonstration on material the detector handles, not as an accuracy
estimate.

The unbiased number is over **all 22 candidates**, which `preflight.py --all` still scores:

| | | |
|---|---|---|
| AI clips flagged | **8 / 13** | missed: all three **Veo** clips (median 0.008–0.146), **CogVideoX-5B** (0.480), one VideoPoet (0.528) |
| Reals at/near trigger | **2 / 9** | two Pexels **stock** clips, peak median-5 **0.950** and **0.974** |

Both failure modes are the ones the report predicts. `docs/REPORT.md` §7.7 found this detector's
failures are *architectural, not stylistic* — a generator whose decoder differs from anything in
training does not carry the fingerprint it learned, and a video model's temporal VAE is exactly
that case. Only RunwayML and VideoPoet survive among AI families that also clear the 768 px
equalisation floor, so **the shipped feed covers two architectures, not five**.

On the real side the split is just as sharp: handheld Wikimedia footage scores 0.0001–0.003, while
graded, denoised Pexels stock runs 0.030–0.913 — the ERROR_ANALYSIS §1 false-positive profile.
`preflight.py` reports a `headroom` column for this: a real clip whose peak median-5 comes within
0.05 of the trigger is a live false alarm waiting for the right segment, even if it never latched
during the offline sweep.

## Try your own file

The **Try your own** tab takes an image or a video (drag-and-drop, or click to pick).

- An **image** is scored once, immediately, with all four heads shown.
- A **video** is normalised through the *same* pipeline as the feed — equalised via the common
  768 px intermediate, cropped to fill 720×1280, trimmed to 12 s — then added to the feed as a card
  and scored live while it plays. Its ground-truth chip reads *unknown*, because it is.

Normalising uploads the same way matters: a clip scored under different resolution or letterboxing
conditions is not comparable to the shipped ones, and the two rules below are worth up to 0.28 and
0.47 of score respectively.

**Anything that fails validation is deleted, not kept** — unreadable file, unsupported codec, under
1 s, over 300 MB, or letterbox bars that survive the crop. The rejection reason is shown in the UI.
Uploads never leave the machine; they are written to the gitignored `project_demo/uploads/`.

## Files

| | |
|---|---|
| `detector.py` | model lifecycle: one backbone, four heads, `score_jpeg_bytes()`, startup self-test |
| `server.py` | FastAPI: static, byte-range video, `/api/health`, `/api/feed`, `/api/score`, `/api/head` |
| `fetch_videos.py` | manifest → download → ffmpeg normalise → ffprobe/cropdetect validate |
| `preflight.py` | score every clip offline, emit `preflight.csv` + `preflight_traces.json` |
| `uploads/` | **gitignored.** Files from the *Try your own* tab; rejects are deleted immediately |
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

## Clip selection — two stages, and the second one uses scores

**Stage 1, pre-registered, content only.** The criteria in `videos.json` gated which clips entered
the candidate pool: handheld with visible grain and high-texture content for reals; named generator,
no watermark, recorded licence for AI. Committed before anything was scored — see git history.

**Stage 2, score-based, added later at the author's instruction.** Candidates the detector got wrong
were removed from the shipped feed: five AI clips it missed, two reals it false-alarmed on.

This second stage *is* selection by score, which is the error `docs/REPORT.md` §7.3 spends a section
retracting (a source looks better because it resembles the test set). It is done deliberately and
recorded rather than hidden:

- every removal is listed in `videos.json` under `removed_for_failing`, with the score that decided it
- every removed clip stays in the manifest and is still scored by `preflight.py --all`
- the unbiased 8/13 and 2/9 figures are stated in the UI's limits panel and above

Run `python project_demo/preflight.py --all --fps 3` to reproduce the full table over all 22
candidates across five generator families.

## Licensing

No media is committed or redistributed. `videos.json` records where each clip comes from and
`fetch_videos.py` pulls from the original hosts at run time. AI clips are CC BY 4.0 (DeepAction
v1); real clips are CC BY / CC BY-SA (Wikimedia Commons) and the Pexels License. See
`docs/DATA_LICENSES.md`.

The UI carries no TikTok, Devpost or third-party branding, per `docs/DEMO_SCRIPT.md`.
