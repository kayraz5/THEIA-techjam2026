"""Held-out validation exclusion list (spec §3 rule 1).

We hash decoded pixel content (not file bytes), so a re-encoded / renamed copy of a validation
image is still caught. Hash = sha1 of the RGB image downsampled to 64x64 with the original size
appended; this is robust to container/format changes but not to actual edits.

  build_exclusion_list(paths, out_txt)   -> writes one hash per line
  ExclusionList(out_txt).assert_disjoint(train_paths)   -> raises if any overlap
"""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm


def content_hash(path_or_img) -> str | None:
    try:
        img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
        img = img.convert("RGB")
        w, h = img.size
        small = img.resize((64, 64), Image.BILINEAR)
        h_ = hashlib.sha1(small.tobytes())
        h_.update(f"{w}x{h}".encode())
        return h_.hexdigest()
    except Exception:
        return None


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_many(paths: Iterable, desc="hashing", workers=8) -> list[str | None]:
    paths = list(paths)
    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(content_hash, paths), total=len(paths), desc=desc, leave=False))


def build_exclusion_list(paths: Iterable[str], out_txt: str) -> int:
    hashes = [h for h in hash_many(paths, "building exclusion list") if h]
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write("\n".join(sorted(set(hashes))) + "\n")
    return len(set(hashes))


class ExclusionList:
    def __init__(self, txt: str):
        self.path = txt
        if not os.path.exists(txt):
            raise FileNotFoundError(
                f"Exclusion list {txt} missing. Run `python -m src.data.build_val --config <cfg>` first; "
                "training refuses to start without it.")
        with open(txt) as f:
            self.hashes = {l.strip() for l in f if l.strip()}
        assert self.hashes, f"exclusion list {txt} is empty"

    def __contains__(self, h: str) -> bool:
        return h in self.hashes

    def assert_disjoint(self, items: list, desc="train") -> None:
        """items: file paths or PIL images. Raises AssertionError listing offenders if any overlap."""
        hs = hash_many(items, f"exclusion check ({desc})")
        bad = [str(p)[:120] for p, h in zip(items, hs) if h in self.hashes]
        msg = f"[exclusion-check] {desc}: {len(items)} samples checked against {len(self.hashes)} held-out hashes -> {len(bad)} overlaps"
        print(msg, flush=True)
        assert not bad, msg + "\nFirst offenders:\n  " + "\n  ".join(bad[:10])
