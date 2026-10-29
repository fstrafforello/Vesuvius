"""Full-resolution (level 0, 2.4 um) CT for a band of sheets in a named region, for the ink runs.

The band is a y-range of the region's level-1 tracing block (sheets there run roughly along x), over
the block's full z and x range plus a margin. Saved as ct_L0_<name>.zarr + .json, usable with
first_ink.py --ct <name>.

Usage:  uv run python src/surface/fetch_ct_band.py --region top --y 300 800 --name top_band
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorstore as ts

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "surface")]
from fetch_starter import remote  # noqa: E402
from scout import SCANS  # noqa: E402
from trace import D, load  # noqa: E402

MARGIN = 70     # level-0 voxels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--y", type=int, nargs=2, required=True, help="y range in the level-1 block")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    block, region, _ = load(1, a.region)
    nz, ny, nx = block.shape
    lo = np.array([region["z0"], region["y0"] + a.y[0], region["x0"]]) * 2 - MARGIN
    hi = np.array([region["z0"] + nz, region["y0"] + a.y[1], region["x0"] + nx]) * 2 + MARGIN
    shape = (hi - lo).tolist()
    print(f"CT band origin zyx {lo.tolist()} shape {shape}: {np.prod(shape) / 1e9:.1f} GB raw", flush=True)
    src = remote(f"{SCANS['PHerc0846A']}/0")
    dst = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(D / f"ct_L0_{a.name}.zarr")},
                   "metadata": {"shape": shape, "dtype": "|u1", "chunks": [128, 128, 128],
                                "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}},
                   "create": True, "delete_existing": True}).result()
    for z in range(0, shape[0], 256):
        z1 = min(shape[0], z + 256)
        dst[z:z1].write(src[lo[0] + z:lo[0] + z1, lo[1]:hi[1], lo[2]:hi[2]].read().result()).result()
        print(f"  z {z}..{z1} of {shape[0]}", flush=True)
    (D / f"ct_L0_{a.name}.json").write_text(json.dumps({"origin_zyx": lo.tolist(), "shape": shape}))
    print("fetched", D / f"ct_L0_{a.name}.zarr", flush=True)


if __name__ == "__main__":
    main()
