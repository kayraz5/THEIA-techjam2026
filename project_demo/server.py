"""Local server for the live feed demo.

    python project_demo/server.py [--host 127.0.0.1] [--port 8000] [--no-self-test]

One process, one port: it serves the frontend, the normalised clips (with byte-range support,
which Safari requires for <video>), and the scoring API. No CORS, no proxy, no second terminal.

Inference is genuinely live — there are no precomputed or cached scores anywhere in this file.
`preflight.csv`, if present, is served alongside the feed purely so the UI can show what we
already measured each clip does, including the clips the detector gets wrong.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, threading, time

from fastapi import Body, FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from project_demo.detector import Detector, DEFAULT_CONFIG  # noqa: E402
from project_demo.fetch_videos import normalise_file, probe, cropdetect  # noqa: E402

VIDEO_DIR = os.path.join(HERE, "videos")
UPLOAD_DIR = os.path.join(HERE, "uploads")
STATIC_DIR = os.path.join(HERE, "static")
MANIFEST = os.path.join(HERE, "videos.json")
PREFLIGHT = os.path.join(HERE, "preflight.csv")

MAX_FRAME_BYTES = 8 << 20    # a 720x1280 JPEG q95 is ~250 KB; 8 MB is a generous ceiling
MAX_UPLOAD_BYTES = 300 << 20 # 300 MB
UPLOAD_TRIM_S = 12.0         # uploads are trimmed to this so one clip cannot hog the demo
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".gif"}

app = FastAPI(title="AI-content detection — live feed demo", docs_url=None, redoc_url=None)

DET: Detector | None = None
# Exactly one frame is scored at a time. A second request is refused rather than queued:
# a queued frame is a stale answer displayed as a live one, which is a correctness problem.
GATE = threading.Semaphore(1)
STATS = {"scored": 0, "busy": 0}


# ---- feed ----------------------------------------------------------------

def _preflight_rows() -> dict:
    if not os.path.exists(PREFLIGHT):
        return {}
    with open(PREFLIGHT, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f) if r.get("id")}


def _manifest_clips() -> list:
    if not os.path.exists(MANIFEST):
        return []
    return json.load(open(MANIFEST)).get("clips", [])


# ---- routes --------------------------------------------------------------

@app.get("/api/health")
def health():
    if DET is None:
        return JSONResponse({"ready": False}, status_code=503)
    h = DET.health()
    h["stats"] = dict(STATS)
    h["n_clips"] = len([c for c in _manifest_clips() if c.get("ship")
                        and os.path.exists(os.path.join(VIDEO_DIR, f"{c['id']}.mp4"))])
    return h


@app.get("/api/feed")
def feed():
    """Manifest rows whose normalised mp4 actually exists, each with its pre-flight row.

    Driven by the manifest, never by globbing videos/ — scripts/demo.sh records why: an
    earlier version of that script globbed a directory and silently fell through to a
    DALL-E 2 *training* folder.
    """
    pre = _preflight_rows()
    out = []
    for c in _manifest_clips():
        # Only the ship set is served. Non-ship candidates stay on disk for preflight.py to
        # measure, but must not appear in the feed.
        if not c.get("ship"):
            continue
        path = os.path.join(VIDEO_DIR, f"{c['id']}.mp4")
        if not os.path.exists(path):
            continue
        row = dict(c)
        row["src"] = f"/videos/{c['id']}.mp4"
        row["preflight"] = pre.get(c["id"])
        out.append(row)
    return {"clips": out, "n": len(out)}


@app.get("/videos/{name}")
def video(name: str):
    """FileResponse handles Range/206, which Safari requires for <video> playback."""
    if not name.endswith(".mp4") or "/" in name or "\\" in name or name.startswith("."):
        return JSONResponse({"error": "bad name"}, status_code=400)
    path = os.path.realpath(os.path.join(VIDEO_DIR, name))
    # Every served path must resolve inside videos/ — no traversal, no hotlinking.
    if not path.startswith(os.path.realpath(VIDEO_DIR) + os.sep) or not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/head")
def set_head(body: dict):
    if DET is None:
        return JSONResponse({"ok": False, "error": "not ready"}, status_code=503)
    try:
        return {"ok": True, "active_head": DET.set_active(body.get("name", ""))}
    except KeyError:
        return JSONResponse({"ok": False, "error": f"unknown head {body.get('name')!r}",
                             "available": list(DET.heads)}, status_code=400)


# Deliberately a plain `def`, not `async def`: Starlette runs sync handlers in a threadpool,
# so the ~85 ms MPS forward never blocks the event loop.
@app.post("/api/score")
def score(body: bytes = Body(..., media_type="image/jpeg"),
          vid: str = "", t: float = -1.0, seq: int = -1):
    if DET is None:
        return JSONResponse({"ok": False, "error": "not ready"}, status_code=503)
    if not GATE.acquire(blocking=False):
        STATS["busy"] += 1
        # HTTP 200 with busy:true rather than 429, so the client's fetch path stays boring.
        return {"ok": False, "busy": True, "seq": seq, "vid": vid}
    try:
        data = body
        if not data:
            return JSONResponse({"ok": False, "error": "empty body"}, status_code=400)
        if len(data) > MAX_FRAME_BYTES:
            return JSONResponse({"ok": False, "error": "frame too large"}, status_code=413)
        t0 = time.time()
        heads = DET.score_jpeg_bytes(data)
        ms = round((time.time() - t0) * 1000, 1)
        STATS["scored"] += 1
        return {"ok": True, "seq": seq, "vid": vid, "t": t, "bytes": len(data),
                "heads": heads, "active_head": DET.active, "p": heads[DET.active], "ms": ms}
    except Exception as e:                      # a bad frame must not kill the capture loop
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=400)
    finally:
        GATE.release()


# ---- upload ---------------------------------------------------------------

def _reject(path: str, reason: str, detail: str = ""):
    """Delete the artefact and say why. Nothing that fails validation is kept on disk."""
    for f in filter(None, [path]):
        try:
            if os.path.exists(f):
                os.remove(f)
        except OSError:
            pass
    return JSONResponse({"ok": False, "error": reason, "detail": detail}, status_code=400)


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    """Score an uploaded image, or normalise an uploaded video into a playable feed card.

    Uploads traverse the SAME ffmpeg pipeline as the shipped clips (equalisation through a
    common 768 px intermediate, then crop-to-fill to 720x1280) so an uploaded clip is
    comparable to the ones in the feed rather than scored under different conditions.

    Anything that fails validation is deleted, not kept.
    """
    if DET is None:
        return JSONResponse({"ok": False, "error": "not ready"}, status_code=503)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in IMG_EXT and ext not in VID_EXT:
        return JSONResponse({"ok": False, "error": "unsupported file type",
                             "detail": f"{ext or 'no extension'} — accepted: "
                                       f"{', '.join(sorted(IMG_EXT | VID_EXT))}"}, status_code=400)

    uid = f"up_{int(time.time() * 1000):x}"
    raw = os.path.join(UPLOAD_DIR, uid + "_raw" + ext)
    n = 0
    with open(raw, "wb") as out:
        while chunk := file.file.read(1 << 20):
            n += len(chunk)
            if n > MAX_UPLOAD_BYTES:
                return _reject(raw, "file too large",
                               f"limit {MAX_UPLOAD_BYTES >> 20} MB")
            out.write(chunk)
    if n == 0:
        return _reject(raw, "empty file")

    # ---- image: score it directly, same path as predict.py ----------------
    if ext in IMG_EXT:
        try:
            with open(raw, "rb") as f:
                heads = DET.score_jpeg_bytes(f.read())
        except Exception as e:
            return _reject(raw, "could not decode image", f"{type(e).__name__}: {e}")
        os.replace(raw, os.path.join(UPLOAD_DIR, uid + ext))
        STATS["scored"] += 1
        return {"ok": True, "kind": "image", "id": uid,
                "src": f"/uploads/{uid}{ext}", "filename": file.filename,
                "heads": heads, "p": heads[DET.active], "active_head": DET.active}

    # ---- video: validate, then normalise through the shipped pipeline -----
    info = probe(raw)
    if not info.get("w"):
        return _reject(raw, "not a readable video", "ffprobe found no video stream")
    if info.get("duration", 0) < 1.0:
        return _reject(raw, "video too short", f"{info.get('duration', 0):.1f}s — need >= 1s")

    dst = os.path.join(UPLOAD_DIR, uid + ".mp4")
    trim = {"start": 0.0, "duration": min(UPLOAD_TRIM_S, info["duration"])}
    if not normalise_file(raw, dst, trim=trim, quiet=True):
        return _reject(raw, "could not transcode video",
                       "ffmpeg failed — the codec may be unsupported")

    out = probe(dst)
    problems = []
    if (out.get("w"), out.get("h")) != (720, 1280):
        problems.append(f"unexpected output size {out.get('w')}x{out.get('h')}")
    if out.get("size", 0) < 20_000:
        problems.append("transcoded output is suspiciously small")
    if cropdetect(dst):
        problems.append("letterbox bars survived the crop — scores would not be comparable")
    if problems:
        for f in (raw, dst):
            try:
                os.remove(f)
            except OSError:
                pass
        return JSONResponse({"ok": False, "error": "failed validation",
                             "detail": "; ".join(problems)}, status_code=400)

    try:
        os.remove(raw)                      # keep only the normalised copy
    except OSError:
        pass
    return {"ok": True, "kind": "video", "id": uid, "src": f"/uploads/{uid}.mp4",
            "filename": file.filename, "duration": round(out.get("duration", 0), 1),
            "scale_note": f"source {info['w']}x{info['h']} -> 768 px -> 720x1280",
            "equalised": round(768 / info["h"], 2)}


@app.get("/uploads/{name}")
def uploaded(name: str):
    if "/" in name or "\\" in name or name.startswith("."):
        return JSONResponse({"error": "bad name"}, status_code=400)
    path = os.path.realpath(os.path.join(UPLOAD_DIR, name))
    if not path.startswith(os.path.realpath(UPLOAD_DIR) + os.sep) or not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    ext = os.path.splitext(path)[1].lower()
    media = "video/mp4" if ext == ".mp4" else f"image/{ext.lstrip('.').replace('jpg', 'jpeg')}"
    return FileResponse(path, media_type=media)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def main():
    global DET
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--no-self-test", action="store_true")
    a = ap.parse_args()

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    DET = Detector(a.config, self_test=not a.no_self_test)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    n = len([c for c in _manifest_clips() if c.get("ship")
             and os.path.exists(os.path.join(VIDEO_DIR, f"{c['id']}.mp4"))])
    print(f"\n  SERVER READY  http://{a.host}:{a.port}   "
          f"({n} clips, head={DET.active}, {DET.warm_ms:.0f} ms/frame)\n", flush=True)
    # No --reload: it would load 2.3 GB of weights twice.
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port,
                log_level=("info" if os.environ.get("DEMO_ACCESS_LOG") else "warning"))


if __name__ == "__main__":
    main()
