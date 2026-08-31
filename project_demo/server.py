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

from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from project_demo.detector import Detector, DEFAULT_CONFIG  # noqa: E402

VIDEO_DIR = os.path.join(HERE, "videos")
STATIC_DIR = os.path.join(HERE, "static")
MANIFEST = os.path.join(HERE, "videos.json")
PREFLIGHT = os.path.join(HERE, "preflight.csv")

MAX_FRAME_BYTES = 8 << 20   # a 720x1280 JPEG q95 is ~250 KB; 8 MB is a generous ceiling

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
    DET = Detector(a.config, self_test=not a.no_self_test)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    n = len([c for c in _manifest_clips() if c.get("ship")
             and os.path.exists(os.path.join(VIDEO_DIR, f"{c['id']}.mp4"))])
    print(f"\n  SERVER READY  http://{a.host}:{a.port}   "
          f"({n} clips, head={DET.active}, {DET.warm_ms:.0f} ms/frame)\n", flush=True)
    # No --reload: it would load 2.3 GB of weights twice.
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
