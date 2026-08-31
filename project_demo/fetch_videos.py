"""Fetch, normalise and validate the demo clips named in videos.json.

    python project_demo/fetch_videos.py            # ship set only
    python project_demo/fetch_videos.py --all      # every candidate (for preflight screening)
    python project_demo/fetch_videos.py --verify   # re-validate what is already on disk
    python project_demo/fetch_videos.py --pin      # record sha256 of each raw download

No media is redistributed: this pulls from the original hosts into the gitignored videos/
directory. See docs/DATA_LICENSES.md.

Two rules in the normalisation pass are load-bearing, not stylistic:

  * CROP TO FILL, NEVER PAD. Letterbox bars measurably wreck the score — a padded 9:16 frame
    took a known fake from 0.983 to 0.707 in testing. The backbone mean-pools 576 patch tokens,
    so black bars are flat, noise-free tokens averaged into the representation, which is the
    exact false-positive signature described in docs/ERROR_ANALYSIS.md 1.
  * ONE IDENTICAL COMMAND FOR EVERY CLIP. This kills the *comparative* confound (nobody can say
    the reals went through a harsher pipeline than the fakes). It does NOT neutralise the
    absolute effect: both classes lose high-frequency texture and drift up the score axis.
    Say that plainly rather than claiming the confound is fixed.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "videos.json")
RAW_DIR = os.path.join(HERE, "videos_raw")
OUT_DIR = os.path.join(HERE, "videos")

W, H, FPS = 720, 1280, 30
# Every clip is first scaled to this common intermediate height, then upscaled identically to
# H. Without it the source resolutions (480..4096 px tall) meant every AI clip was UPsampled
# and most reals DOWNsampled — a confound on precisely the axis docs/REPORT.md 7.4 flags as
# unfixed. 768 is at or below every shipped source, so the first step is always a downsample
# and the second is always the same 768->1280 upscale.
EQUALIZE_H = 768
UA = "tiktokMadeMeDoIt-demo/0.1 (research demo; https://github.com/)"
ID_RE = re.compile(r"^[a-z0-9_]+$")


def sh(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def probe(path: str) -> dict:
    r = sh(["ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", path])
    if r.returncode != 0:
        return {}
    d = json.loads(r.stdout)
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    a = [s for s in d.get("streams", []) if s.get("codec_type") == "audio"]
    return {"w": v.get("width"), "h": v.get("height"), "codec": v.get("codec_name"),
            "fps": v.get("avg_frame_rate"), "n_audio": len(a),
            "duration": float(d.get("format", {}).get("duration", 0) or 0),
            "size": int(d.get("format", {}).get("size", 0) or 0)}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def download(clip: dict) -> str:
    ext = os.path.splitext(clip["url"].split("?")[0])[1] or ".mp4"
    dst = os.path.join(RAW_DIR, clip["id"] + ext)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  [skip] {clip['id']}{ext} already downloaded ({os.path.getsize(dst)/1e6:.1f} MB)")
        return dst
    print(f"  [get ] {clip['id']}{ext} <- {clip['url'][:88]}")
    req = urllib.request.Request(clip["url"], headers={"User-Agent": UA})
    tmp = dst + ".part"
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dst)
    print(f"         {os.path.getsize(dst)/1e6:.1f} MB")
    return dst


def cropdetect(path: str) -> str | None:
    """Return the ffmpeg crop= expression if letterbox bars are present, else None."""
    r = sh(["ffmpeg", "-hide_banner", "-ss", "1", "-i", path, "-vf", "cropdetect=24:2:0",
            "-frames:v", "60", "-f", "null", "-"])
    crops = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
    if not crops:
        return None
    cw, ch, cx, cy = crops[-1]
    info = probe(path)
    if not info.get("w"):
        return None
    # Treat >2% trimmed on either axis as real bars.
    if int(cw) < info["w"] * 0.98 or int(ch) < info["h"] * 0.98:
        return f"crop={cw}:{ch}:{cx}:{cy}"
    return None


def normalise(clip: dict, raw: str) -> str:
    out = os.path.join(OUT_DIR, clip["id"] + ".mp4")
    trim = clip.get("trim") or {}
    filters = []
    if clip.get("watermark_crop"):          # e.g. "crop=iw:ih-80:0:0" to remove a generator mark
        filters.append(clip["watermark_crop"])
    bars = cropdetect(raw)
    if bars:
        print(f"  [bars] {clip['id']}: source is letterboxed, cropping {bars}")
        filters.append(bars)
    filters += [
        # Common resolution history for every clip, real and AI alike. See EQUALIZE_H.
        f"scale=-2:{EQUALIZE_H}",
        # increase + crop == crop-to-fill. Never `pad` — see the module docstring.
        f"scale={W}:{H}:force_original_aspect_ratio=increase",
        f"crop={W}:{H}", "setsar=1", f"fps={FPS}",
    ]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if trim.get("start"):
        cmd += ["-ss", str(trim["start"])]              # before -i for a fast seek
    cmd += ["-i", raw]
    if trim.get("duration"):
        cmd += ["-t", str(trim["duration"])]
    cmd += ["-an",                                        # strip audio (and any music licence question)
            "-vf", ",".join(filters),
            "-c:v", "libx264",                            # deterministic across machines, unlike videotoolbox
            "-preset", "veryfast", "-crf", "21",
            "-pix_fmt", "yuv420p", "-g", str(FPS),
            "-movflags", "+faststart",                    # moov atom first, so byte-range playback works
            out]
    r = sh(cmd)
    if r.returncode != 0:
        print(f"  [FAIL] {clip['id']}: ffmpeg\n{r.stderr[-600:]}")
        return ""
    return out


def verify(clips: list) -> bool:
    print("\n" + "=" * 104)
    print(f"{'id':26s} {'label':5s} {'generator':22s} {'res':11s} {'fps':6s} {'dur':>6s} "
          f"{'MB':>6s} {'scale':>7s}  ok")
    print("-" * 104)
    ok_all, n_real, n_ai = True, 0, 0
    for c in clips:
        path = os.path.join(OUT_DIR, c["id"] + ".mp4")
        if not os.path.exists(path):
            print(f"{c['id']:26s} MISSING"); ok_all = False; continue
        p = probe(path)
        raw = next((os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR)
                    if os.path.splitext(f)[0] == c["id"]), None)
        rp = probe(raw) if raw else {}
        problems = []
        if (p.get("w"), p.get("h")) != (W, H): problems.append(f"res {p.get('w')}x{p.get('h')}")
        if p.get("codec") != "h264":          problems.append(f"codec {p.get('codec')}")
        if p.get("n_audio"):                  problems.append(f"{p['n_audio']} audio streams")
        if not (3.0 <= p.get("duration", 0) <= 14.0): problems.append(f"dur {p.get('duration'):.1f}s")
        if p.get("size", 0) < 100_000:        problems.append("under 100 KB")
        if not ID_RE.match(c["id"]):          problems.append("bad id")
        if cropdetect(path):                  problems.append("LETTERBOX BARS")
        # Resolution regime. The 768->1280 step is identical for every clip, so the only
        # per-clip variation is the native->EQUALIZE_H step reported here. It must be <= 1.00
        # for every clip in BOTH classes, or the confound is back (docs/REPORT.md 5 / 7.4).
        scale = f"{EQUALIZE_H / rp['h']:.2f}x" if rp.get("h") else "?"
        n_real += c["label"] == "real"; n_ai += c["label"] == "ai"
        flag = "ok" if not problems else "FAIL: " + ", ".join(problems)
        if problems: ok_all = False
        print(f"{c['id']:26s} {c['label']:5s} {str(c.get('generator') or '-')[:22]:22s} "
              f"{p.get('w')}x{p.get('h'):<6} {str(p.get('fps'))[:6]:6s} {p.get('duration',0):6.1f} "
              f"{p.get('size',0)/1e6:6.1f} {scale:>7s}  {flag}")
    print("-" * 104)
    print(f"{len(clips)} clips: {n_real} real, {n_ai} ai")
    if len(clips) == 10 and (n_real != 5 or n_ai != 5):
        print(f"  !! ship set must be 5 real + 5 ai, got {n_real}/{n_ai}"); ok_all = False
    print(f"{'ALL CLIPS VALIDATED' if ok_all else 'VALIDATION FAILED'}")
    print("=" * 104)
    print(f"\nnote: `scale` = native height -> the common {EQUALIZE_H} px intermediate. Every clip then\n"
          f"      takes the SAME {EQUALIZE_H}->{H} upscale, so this column is the only resampling that\n"
          f"      differs per clip. All values must be <= 1.00 across BOTH labels; a mix of up- and\n"
          f"      down-sampled clips split by label is the confound docs/REPORT.md 5 / 7.4 warns about.")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="fetch every candidate, not just the ship set")
    ap.add_argument("--verify", action="store_true", help="validate what is on disk and exit")
    ap.add_argument("--pin", action="store_true", help="record sha256 of each raw download in videos.json")
    ap.add_argument("--only", default=None, help="comma-separated clip ids")
    a = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg/ffprobe not found on PATH")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    man = json.load(open(MANIFEST))
    clips = man["clips"]
    if a.only:
        want = set(a.only.split(","))
        clips = [c for c in clips if c["id"] in want]
    elif not a.all:
        clips = [c for c in clips if c.get("ship")]

    if a.verify:
        sys.exit(0 if verify(clips) else 1)

    print(f"[fetch] {len(clips)} clips -> {OUT_DIR}")
    for c in clips:
        print(f"\n- {c['id']}  ({c['label']}, {c.get('generator') or 'camera'}, {c['license']})")
        try:
            raw = download(c)
        except Exception as e:
            print(f"  [FAIL] download: {type(e).__name__}: {e}")
            continue
        if a.pin:
            c["sha256"] = sha256(raw)
        if not normalise(c, raw):
            continue
        print(f"  [ok  ] {c['id']}.mp4")

    if a.pin:
        json.dump(man, open(MANIFEST, "w"), indent=2, ensure_ascii=False)
        print(f"\n[pin] wrote sha256 for {len(clips)} clips into videos.json")

    sys.exit(0 if verify(clips) else 1)


if __name__ == "__main__":
    main()
