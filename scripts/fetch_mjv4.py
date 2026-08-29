"""#20: MJv4 lives in Midjourney/Typical/part_*.zip whose members have FLAT filenames
(no directory prefix), so remote_zip.py's --grep cannot select it - the same trap MJv5 hit.
This cross-references archive member basenames against mjv4.csv, then writes the winners to the
path the CSV expects so src/data/registry.py finds them.

  python scripts/fetch_mjv4.py --n 3000
"""
import argparse, os, random, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_zip import central_directory, fetch_windows, resolve

BASE = ("https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo"
        "?Revision=master&FilePath=Images%2FDiffusion_based%2FMidjourney%2FTypical%2Fpart_{}.zip")
DEST = "data/wildfake/images/Diffusion_based/Midjourney/Typical/mj_v4"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--parts", default="1,2,3,4")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    want = set(os.path.basename(p) for p in pd.read_csv("data/wildfake/csv/mjv4.csv").Image_path)
    print(f"[mjv4] {len(want)} filenames in mjv4.csv")
    os.makedirs(DEST, exist_ok=True)
    have = set(os.listdir(DEST))
    got = len(have)

    for part in a.parts.split(","):
        if got >= a.n: break
        url, _ = resolve(BASE.format(part))
        ents = central_directory(url)
        hits = [e for e in ents if os.path.basename(e["name"]) in want
                and os.path.basename(e["name"]) not in have]
        print(f"[mjv4] part_{part}: {len(ents)} members, {len(hits)} match mjv4.csv "
              f"({len(hits)/max(1,len(ents))*100:.0f}%)")
        if not hits:
            continue
        random.Random(a.seed).shuffle(hits)
        hits = sorted(hits[: a.n - got], key=lambda e: e["lho"])
        # members are flat, so strip=0 drops them straight into DEST
        fetch_windows(url, hits, DEST, 0, 8)
        have = set(os.listdir(DEST)); got = len(have)
        print(f"[mjv4] now {got}/{a.n} on disk")
    print(f"[mjv4] DONE {got} files in {DEST}")

if __name__ == "__main__":
    main()
