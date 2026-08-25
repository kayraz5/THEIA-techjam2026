"""Extract selected members from a remote ZIP over HTTP Range requests (no full download).

Used to pull the WildFake DALL-E Advanced subset (8,843 files) out of the 25 GB DALLE.zip,
and small slices of other archives, on a laptop connection.

Usage:
  python scripts/remote_zip.py list  <url> [--grep SUBSTR] [--limit N]
  python scripts/remote_zip.py fetch <url> --out DIR [--grep SUBSTR] [--limit N] [--seed S] [--strip N]
"""
from __future__ import annotations
import argparse, io, os, random, struct, sys, zlib, time
import requests
from tqdm import tqdm

S = requests.Session()
S.headers["User-Agent"] = "remote-zip/0.1"

def _get(url, start, end, retries=6):
    for i in range(retries):
        try:
            r = S.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120, allow_redirects=True)
            if r.status_code in (200, 206):
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"range fetch failed {start}-{end}")

def resolve(url):
    """Follow the ModelScope redirect once; reuse the signed CDN URL for all range requests."""
    r = S.head(url, allow_redirects=True, timeout=60)
    return r.url, int(r.headers["Content-Length"])

def _size(url):
    return resolve(url)[1]

def central_directory(url):
    size = _size(url)
    tail = _get(url, max(0, size - 65557 - 20 - 56), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    assert i >= 0, "EOCD not found"
    eocd = tail[i:i + 22]
    cd_size, cd_off = struct.unpack("<II", eocd[12:20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        z64 = tail[j:j + 56]
        cd_size, cd_off = struct.unpack("<QQ", z64[40:56])
    cd = _get(url, cd_off, cd_off + cd_size - 1)
    entries, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (method, csize, usize, nlen, xlen, clen, lho) = struct.unpack("<H", cd[p+10:p+12]) + struct.unpack("<II", cd[p+20:p+28]) + struct.unpack("<HHH", cd[p+28:p+34]) + struct.unpack("<I", cd[p+42:p+46])
        name = cd[p+46:p+46+nlen].decode("utf-8", "replace")
        extra = cd[p+46+nlen:p+46+nlen+xlen]
        if 0xFFFFFFFF in (csize, usize, lho):  # zip64 extra field
            q = 0
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q+4]); body = extra[q+4:q+4+hsz]
                if hid == 1:
                    r = 0
                    if usize == 0xFFFFFFFF: usize = struct.unpack("<Q", body[r:r+8])[0]; r += 8
                    if csize == 0xFFFFFFFF: csize = struct.unpack("<Q", body[r:r+8])[0]; r += 8
                    if lho == 0xFFFFFFFF: lho = struct.unpack("<Q", body[r:r+8])[0]; r += 8
                q += 4 + hsz
        if not name.endswith("/"):
            entries.append(dict(name=name, method=method, csize=csize, usize=usize, lho=lho))
        p += 46 + nlen + xlen + clen
    return entries

def fetch_member(url, e):
    hdr = _get(url, e["lho"], e["lho"] + 29)
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    start = e["lho"] + 30 + nlen + xlen
    raw = _get(url, start, start + e["csize"] - 1)
    if e["method"] == 0:
        return raw
    if e["method"] == 8:
        return zlib.decompress(raw, -15)
    raise ValueError(f"unsupported method {e['method']}")

def fetch_windows(url, ents, out, strip, workers, max_win=16 << 20, max_gap=1 << 20):
    """Group entries (sorted by local header offset) into byte windows and fetch each in ONE request."""
    ents = sorted(ents, key=lambda e: e["lho"])
    wins, cur = [], []
    def span(es): return es[0]["lho"], es[-1]["lho"] + 30 + len(es[-1]["name"]) + 512 + es[-1]["csize"]
    for e in ents:
        if cur:
            s0, _ = span(cur); _, e1 = span(cur + [e])
            if e1 - s0 > max_win or e["lho"] - span(cur)[1] > max_gap:
                wins.append(cur); cur = []
        cur.append(e)
    if cur: wins.append(cur)
    def work(es):
        todo = []
        for e in es:
            dst = os.path.join(out, *e["name"].split("/")[strip:])
            if not (os.path.exists(dst) and os.path.getsize(dst) == e["usize"]): todo.append((e, dst))
        if not todo: return 0
        s0, e1 = span(es)
        blob = _get(url, s0, e1 - 1); n = 0
        for e, dst in todo:
            o = e["lho"] - s0
            nlen, xlen = struct.unpack("<HH", blob[o+26:o+30])
            start = o + 30 + nlen + xlen
            raw = blob[start:start + e["csize"]]
            data = raw if e["method"] == 0 else zlib.decompress(raw, -15)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f: f.write(data)
            n += len(data)
        return n
    from concurrent.futures import ThreadPoolExecutor
    print(f"{len(ents)} files in {len(wins)} windows", file=sys.stderr)
    tot = 0
    with ThreadPoolExecutor(workers) as ex:
        for n in tqdm(ex.map(work, wins), total=len(wins), unit="win"): tot += n
    print(f"wrote {tot/1e9:.2f} GB to {out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "fetch"]); ap.add_argument("url")
    ap.add_argument("--grep", default=None); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None); ap.add_argument("--strip", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.url, _ = resolve(a.url)
    ents = central_directory(a.url)
    print(f"{len(ents)} members in archive", file=sys.stderr)
    if a.grep:
        ents = [e for e in ents if a.grep in e["name"]]
    if a.limit and len(ents) > a.limit:
        random.Random(a.seed).shuffle(ents); ents = sorted(ents[:a.limit], key=lambda e: e["lho"])
    if a.cmd == "list":
        for e in ents[:200]: print(e["name"], e["usize"])
        print(f"... {len(ents)} selected, {sum(e['usize'] for e in ents)/1e9:.2f} GB", file=sys.stderr)
        return
    os.makedirs(a.out, exist_ok=True)
    fetch_windows(a.url, ents, a.out, a.strip, a.workers)

if __name__ == "__main__":
    main()
