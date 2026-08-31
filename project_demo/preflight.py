"""Score every normalised clip offline and emit the truth table the demo shows on screen.

    python project_demo/preflight.py [--fps 2] [--all] [--out preflight.csv]

This is NOT precomputing demo scores — the served demo scores every frame live. This exists so
we are UNSURPRISED, not so we are right. If a clip fails, the correct response is a prepared
sentence citing docs/ERROR_ANALYSIS.md, not a quiet swap: clips are selected by the
pre-registered content criteria in videos.json, never by score (docs/REPORT.md 7.3 retracts
exactly that error).

Emits per clip: the full score trajectory, the badge rule's verdict, the flicker metric, and
the frame-to-frame correlation that determines the real effective sample size.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from project_demo.detector import Detector  # noqa: E402

VIDEO_DIR = os.path.join(HERE, "videos")
MANIFEST = os.path.join(HERE, "videos.json")

# Must match TUNING in static/app.js.
ON_THRESH, OFF_THRESH = 0.9446, 0.70
ON_UPDATES, OFF_UPDATES, WINDOW = 2, 5, 5
# Calibrated operating points from results/frozen_siglip2_giant_ship/eval/thresholds.csv.
THRESHOLDS = [0.1266, 0.5, 0.6388, 0.9446]


def extract_frames(path: str, fps: float):
    """Yield PIL frames at `fps` by piping mjpeg out of ffmpeg (no new video dependency)."""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps}",
         "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "-"],
        capture_output=True)
    buf, out, SOI, EOI = p.stdout, [], b"\xff\xd8", b"\xff\xd9"
    i = 0
    while True:
        a = buf.find(SOI, i)
        if a < 0:
            break
        b = buf.find(EOI, a)
        if b < 0:
            break
        out.append(Image.open(__import__("io").BytesIO(buf[a:b + 2])).convert("RGB"))
        i = b + 2
    return out


def badge_trace(ps: list, on=ON_THRESH, off=OFF_THRESH):
    """Replay the exact hysteresis rule the frontend uses.

    Returns (on_fraction, transitions, max_median5). The last value is the headroom metric:
    a clip whose median-5 peaks near `on` will fire on some playthroughs and not others,
    which is invisible if you only look at whether the badge latched during one sweep.
    """
    state, on_run, off_run, hist, states = False, 0, 0, [], []
    max_med = 0.0
    for p in ps:
        hist.append(p)
        med = float(np.median(hist[-WINDOW:])) if len(hist) >= 3 else None
        if med is not None:
            max_med = max(max_med, med)
            if not state:
                on_run = on_run + 1 if med >= on else 0
                if on_run >= ON_UPDATES:
                    state, on_run, off_run = True, 0, 0
            else:
                off_run = off_run + 1 if med <= off else 0
                if off_run >= OFF_UPDATES:
                    state, on_run, off_run = False, 0, 0
        states.append(state)
    trans = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
    return (sum(states) / len(states) if states else 0.0), trans, max_med


def classify(label: str, med: float, on_frac: float, trans: int) -> str:
    correct = (label == "ai" and on_frac > 0.5) or (label == "real" and on_frac < 0.02)
    if trans > 2:
        return "flickering"
    if correct:
        return "pinned_correct"
    if 0.02 <= on_frac <= 0.5:
        return "crosses_mid_clip"
    return "pinned_wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--all", action="store_true", help="include non-ship candidates")
    ap.add_argument("--out", default=os.path.join(HERE, "preflight.csv"))
    ap.add_argument("--traces", default=os.path.join(HERE, "preflight_traces.json"))
    a = ap.parse_args()

    clips = json.load(open(MANIFEST))["clips"]
    if not a.all:
        clips = [c for c in clips if c.get("ship")]
    clips = [c for c in clips if os.path.exists(os.path.join(VIDEO_DIR, c["id"] + ".mp4"))]
    if not clips:
        sys.exit("no normalised clips found — run fetch_videos.py first")

    det = Detector()
    heads = list(det.heads)
    rows, traces, lat = [], {}, []

    print(f"\n[preflight] {len(clips)} clips at {a.fps} fps, heads={heads}\n")
    for c in clips:
        path = os.path.join(VIDEO_DIR, c["id"] + ".mp4")
        frames = extract_frames(path, a.fps)
        per_head = {h: [] for h in heads}
        for f in frames:
            t0 = time.time()
            s = det._forward(f)
            lat.append((time.time() - t0) * 1000)
            for h in heads:
                per_head[h].append(s[h])
        ps = per_head[det.active]
        arr = np.array(ps)
        on_frac, trans, max_med5 = badge_trace(ps)
        corr = float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if len(arr) > 2 and arr.std() > 1e-9 else 1.0
        row = {
            "id": c["id"], "label": c["label"], "generator": c.get("generator") or "",
            "ship": int(bool(c.get("ship"))), "n_frames": len(ps),
            "min": round(float(arr.min()), 4), "p05": round(float(np.percentile(arr, 5)), 4),
            "median": round(float(np.median(arr)), 4), "mean": round(float(arr.mean()), 4),
            "p95": round(float(np.percentile(arr, 95)), 4), "max": round(float(arr.max()), 4),
            "frame_corr": round(corr, 4),
            "pill_on_fraction": round(on_frac, 3), "n_transitions": trans,
            "max_median5": round(max_med5, 4),
            "headroom": round(ON_THRESH - max_med5, 4),
            "verdict": classify(c["label"], float(np.median(arr)), on_frac, trans),
        }
        for t in THRESHOLDS:
            row[f"frac_ge_{t}"] = round(float((arr >= t).mean()), 3)
        for h in heads:
            row[f"median_{h}"] = round(float(np.median(per_head[h])), 4)
        rows.append(row)
        traces[c["id"]] = {"fps": a.fps, "label": c["label"],
                           **{h: [round(v, 4) for v in per_head[h]] for h in heads}}

        mark = {"pinned_correct": "ok", "pinned_wrong": "WRONG",
                "flickering": "FLICKER", "crosses_mid_clip": "PARTIAL"}[row["verdict"]]
        # A real clip within 0.05 of the trigger is a live false alarm waiting to happen.
        if c["label"] == "real" and row["headroom"] < 0.05:
            mark += "  <- NEAR THRESHOLD, will fire on some playthroughs"
        print(f"  {c['id']:26s} {c['label']:4s} n={len(ps):3d} med={row['median']:.4f} "
              f"[{row['min']:.3f}-{row['max']:.3f}] pill={on_frac*100:5.1f}% "
              f"trans={trans} maxmed5={max_med5:.3f} r={corr:.3f}  {mark}")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump(traces, open(a.traces, "w"), indent=1)

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 78)
    ai = [r for r in rows if r["label"] == "ai"]
    re_ = [r for r in rows if r["label"] == "real"]
    caught = sum(r["pill_on_fraction"] > 0.5 for r in ai)
    false_alarm = sum(r["pill_on_fraction"] > 0.02 for r in re_)
    print(f"badge rule: raise median-{WINDOW} >= {ON_THRESH} x{ON_UPDATES}, "
          f"clear <= {OFF_THRESH} x{OFF_UPDATES}")
    print(f"  AI clips flagged      {caught}/{len(ai)}")
    print(f"  REAL clips false-alarm {false_alarm}/{len(re_)}")
    print(f"\nper-threshold fraction of frames at or above (all clips):")
    print(f"  {'threshold':>10s} " + "".join(f"{'ai':>8s}{'real':>8s}" for _ in [0]))
    for t in THRESHOLDS:
        ma = np.mean([r[f"frac_ge_{t}"] for r in ai]) if ai else 0
        mr = np.mean([r[f"frac_ge_{t}"] for r in re_]) if re_ else 0
        print(f"  {t:>10.4f} {ma:>8.2f}{mr:>8.2f}")
    print(f"\nhead medians (AI / REAL):")
    for h in heads:
        ma = np.median([r[f"median_{h}"] for r in ai]) if ai else 0
        mr = np.median([r[f"median_{h}"] for r in re_]) if re_ else 0
        print(f"  {h:10s} {ma:.4f} / {mr:.4f}")
    mc = np.mean([r["frame_corr"] for r in rows])
    print(f"\nmean frame-to-frame correlation r={mc:.4f} — frames within a shot are highly")
    print(f"  correlated, so the effective sample size is ~1 per shot, not one per frame.")
    print(f"  Aggregation is DISPLAY SMOOTHING, not error reduction (docs/REPORT.md 7.5).")
    print(f"\nlatency p50={np.percentile(lat,50):.0f} ms  p95={np.percentile(lat,95):.0f} ms  "
          f"({1000/np.mean(lat):.1f} img/s)")
    print(f"\nwrote {a.out} and {a.traces}")
    print("=" * 78)
    notok = [r for r in rows if r["verdict"] != "pinned_correct"]
    if notok:
        print("\nclips needing a prepared sentence at the table:")
        for r in notok:
            print(f"  {r['id']:26s} {r['verdict']:18s} median={r['median']:.4f} "
                  f"pill={r['pill_on_fraction']*100:.0f}%")


if __name__ == "__main__":
    main()
