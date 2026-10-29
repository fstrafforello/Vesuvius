"""Download labelled 512x512 blocks from every Scroll 4 segment that has ink labels.

Per segment, into data/PHerc1667/ink_blocks/<segment>/:
  labels/            full ink labels + supervision mask (zarr v3, small)
  surface.zarr       sparse full-size array, only the chosen blocks are written; layers
                     SURFACE_LAYER-18 .. +18 (36 layers: the 32 the model uses + jitter room)
  blocks.json        [y, x] of every downloaded block (level-0 surface pixels, 512-aligned)

Blocks are picked at random among those at least MIN_SUP supervised, capped per segment.
Remote chunks are 109 layers deep, so transfer is ~3x the size kept on disk.
Re-running only fetches blocks not already present.

Usage:  uv run python src/fetch_ink_blocks.py
"""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorstore as ts

sys.path.insert(0, str(Path(__file__).parent))
from fetch_starter import BUCKET, SCAN_ID, mirror  # noqa: E402

SCROLL = "PHerc1667"
SEG_ROOT = f"{SCROLL}/segments"
SEGMENTS = {  # name: max blocks
    "20240304144031-w018_20240304144031_flatboi": 200,
    "20240304141531-w013_20240304141531_flatboi": 100,
    "20240304161941-w023_20240304161941_flatboi": 100,
    "20251223230000-w031_2025122323_flatboi": 100,   # held out for validation
    "20251208130119-w028_20251208130119156_flatboi": 50,
    "20251212185248-w029_20251212185248662_flatboi": 50,
}
BLOCK = 512
MIN_SUP = 0.6
SURFACE_LAYER, HALF = 54, 18
OUT = Path(__file__).resolve().parents[1] / "data" / SCROLL / "ink_blocks"


def fetch_segment(name, cap, rng):
    seg = f"{SEG_ROOT}/{name}"
    out = OUT / name.split("-")[1].split("_")[0]          # e.g. ink_blocks/w018
    out.mkdir(parents=True, exist_ok=True)
    mirror(f"{seg}/ink-labels/2.399um-volume-{SCAN_ID}/20260918", out / "labels")

    sup2 = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(out / "labels/supervision.zarr/2")}},
                   read=True).result().read().result() > 0   # level 2 = 1/4 res, so a block is 128x128 here
    b = BLOCK // 4
    h, w = sup2.shape[0] // b, sup2.shape[1] // b
    cover = sup2[:h * b, :w * b].reshape(h, b, w, b).mean((1, 3))
    eligible = [(int(y) * BLOCK, int(x) * BLOCK) for y, x in zip(*np.nonzero(cover >= MIN_SUP))]
    blocks_file = out / "blocks.json"
    done = json.loads(blocks_file.read_text()) if blocks_file.exists() else []
    done_set = {tuple(v) for v in done}
    todo = [v for v in eligible if v not in done_set]
    rng.shuffle(todo)
    todo = todo[:max(0, cap - len(done))]

    src = ts.open({"driver": "zarr", "kvstore": {"driver": "http",
                   "base_url": f"{BUCKET}/{seg}/surface-volumes/2.399um-0.22m-78keV-volume-{SCAN_ID}.zarr/0"}},
                  read=True).result()
    dst = ts.open({
        "driver": "zarr", "kvstore": {"driver": "file", "path": str(out / "surface.zarr")},
        "metadata": {"shape": [2 * HALF, *src.shape[1:]], "dtype": "|u1", "chunks": [2 * HALF, BLOCK, BLOCK],
                     "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}},
        "open": True, "create": True,
    }).result()
    print(f"{out.name}: {len(eligible)} eligible blocks, {len(done)} already here, fetching {len(todo)}", flush=True)

    def one(yx):
        y, x = yx
        dst[:, y:y + BLOCK, x:x + BLOCK].write(
            src[SURFACE_LAYER - HALF:SURFACE_LAYER + HALF, y:y + BLOCK, x:x + BLOCK].read().result()).result()
        return yx

    with ThreadPoolExecutor(8) as pool:
        for i, yx in enumerate(pool.map(one, todo), 1):
            done.append(list(yx))
            if i % 25 == 0 or i == len(todo):
                blocks_file.write_text(json.dumps(done))
                print(f"  {out.name}: {i}/{len(todo)}", flush=True)
    blocks_file.write_text(json.dumps(done))


def main():
    rng = random.Random(0)
    for name, cap in SEGMENTS.items():
        fetch_segment(name, cap, rng)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
