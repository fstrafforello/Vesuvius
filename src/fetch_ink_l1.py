"""Download the half-resolution (level 1) surface volume around ALL labelled areas of the
Scroll 4 segments that have ink labels.

At level 1 a 256x256 block covers the same papyrus as 512x512 at full res, and costs 1/4 of the
transfer, so we can take every block that touches supervision instead of a sample.

Per segment, into data/PHerc1667/ink_l1/<wNNN>/:
  labels/        ink labels + supervision mask (zarr v3; level 1 is used for training)
  surface.zarr   sparse level-1 array, 36 layers around the surface, only chosen blocks written
  blocks.json    [y, x] (level-1 pixels, 256-aligned) of every downloaded block

Usage:  uv run python src/fetch_ink_l1.py [--dry-run]
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorstore as ts

sys.path.insert(0, str(Path(__file__).parent))
from fetch_ink_blocks import HALF, SEG_ROOT, SEGMENTS, SURFACE_LAYER  # noqa: E402
from fetch_starter import BUCKET, SCAN_ID, mirror  # noqa: E402

BLOCK = 256        # level-1 pixels
MIN_SUP = 0.3      # take a block if at least this much of it is supervised
DILATE = 0         # plus this many neighbouring blocks, so patches near label edges have context
OUT = Path(__file__).resolve().parents[1] / "data" / "PHerc1667" / "ink_l1"
EST_MB_PER_BLOCK = 4 * 1.25   # 4 remote chunks of 109x128x128, ~1.25 MB each compressed


def short(name):
    return name.split("-")[1].split("_")[0]


def choose_blocks(out):
    sup = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(out / "labels/supervision.zarr/3")}},
                  read=True).result().read().result() > 0   # level 3 = 1/4 of level 1, a block is 64x64 here
    b = BLOCK // 4
    h, w = sup.shape[0] // b, sup.shape[1] // b
    cover = sup[:h * b, :w * b].reshape(h, b, w, b).mean((1, 3))
    pick = cover >= MIN_SUP
    grown = pick.copy()
    for dy in range(-DILATE, DILATE + 1):
        for dx in range(-DILATE, DILATE + 1):
            grown |= np.roll(np.roll(pick, dy, 0), dx, 1)
    return [(int(y) * BLOCK, int(x) * BLOCK) for y, x in zip(*np.nonzero(grown))]


def fetch_segment(name, dry):
    seg = f"{SEG_ROOT}/{name}"
    out = OUT / short(name)
    out.mkdir(parents=True, exist_ok=True)
    mirror(f"{seg}/ink-labels/2.399um-volume-{SCAN_ID}/20260918", out / "labels")
    blocks = choose_blocks(out)
    blocks_file = out / "blocks.json"
    done = json.loads(blocks_file.read_text()) if blocks_file.exists() else []
    done_set = {tuple(v) for v in done}
    todo = [v for v in blocks if v not in done_set]
    print(f"{out.name}: {len(blocks)} blocks (~{len(blocks) * EST_MB_PER_BLOCK / 1024:.1f} GB transfer), "
          f"{len(done)} already here, {len(todo)} to fetch", flush=True)
    if dry:
        return len(blocks)

    src = ts.open({"driver": "zarr", "kvstore": {"driver": "http",
                   "base_url": f"{BUCKET}/{seg}/surface-volumes/2.399um-0.22m-78keV-volume-{SCAN_ID}.zarr/1"}},
                  read=True).result()
    dst = ts.open({
        "driver": "zarr", "kvstore": {"driver": "file", "path": str(out / "surface.zarr")},
        "metadata": {"shape": [2 * HALF, *src.shape[1:]], "dtype": "|u1", "chunks": [2 * HALF, BLOCK, BLOCK],
                     "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}},
        "open": True, "create": True,
    }).result()

    def one(yx):
        y, x = yx
        for attempt in range(5):
            try:
                dst[:, y:y + BLOCK, x:x + BLOCK].write(
                    src[SURFACE_LAYER - HALF:SURFACE_LAYER + HALF, y:y + BLOCK, x:x + BLOCK].read().result()).result()
                return yx
            except Exception:
                if attempt == 4:
                    raise

    with ThreadPoolExecutor(8) as pool:
        for i, yx in enumerate(pool.map(one, todo), 1):
            done.append(list(yx))
            if i % 100 == 0 or i == len(todo):
                blocks_file.write_text(json.dumps(done))
                print(f"  {out.name}: {i}/{len(todo)}", flush=True)
    blocks_file.write_text(json.dumps(done))
    return len(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    total = sum(fetch_segment(n, args.dry_run) for n in SEGMENTS)
    print(f"total {total} blocks, ~{total * EST_MB_PER_BLOCK / 1024:.1f} GB transfer, "
          f"~{total * 2 * HALF * BLOCK * BLOCK / 1024**3 * 0.7:.1f} GB on disk")


if __name__ == "__main__":
    main()
