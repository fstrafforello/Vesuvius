"""Download a level-1 (4.8 um) tracing block of PHerc0846A around the wrap we traced at level 2.

At level 2 the sheets are only 2-4 voxels thick with neighbours 3-5 voxels away, so the tracer
slides onto touching sheets; at level 1 they are twice as thick with clearer gaps.

Region (level-2 block coordinates of trace_L2.zarr): z 250..850, y 850..1350, x 250..1300,
i.e. about 6 x 5 x 10 mm around the wrap. Saved as trace_L1.zarr + region_L1.json.

Usage:  uv run python src/surface/fetch_trace_l1.py
"""

import json
import sys
from pathlib import Path

import tensorstore as ts

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "surface")]
from fetch_starter import remote  # noqa: E402
from scout import SCANS  # noqa: E402

D = ROOT / "data" / "scout" / "PHerc0846A"
L2_BOX = {"z": (250, 850), "y": (850, 1350), "x": (250, 1300)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="region name (default: the original region)")
    ap.add_argument("--box", type=int, nargs=6, default=None, metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
                    help="level-2 block coordinates of trace_L2.zarr")
    a = ap.parse_args()
    box = {"z": tuple(a.box[0:2]), "y": tuple(a.box[2:4]), "x": tuple(a.box[4:6])} if a.box else L2_BOX
    r2 = json.loads((D / "region.json").read_text())
    # level-2 block coords -> absolute level-2 -> absolute level-1 (x2), chunk-aligned to 128
    lo = {k: (r2[f"{k}0"] + box[k][0]) * 2 // 128 * 128 for k in "zyx"}
    hi = {k: -(-((r2[f"{k}0"] + box[k][1]) * 2) // 128) * 128 for k in "zyx"}
    shape = [hi[k] - lo[k] for k in "zyx"]
    print(f"level-1 block origin zyx {[lo[k] for k in 'zyx']} shape {shape} ({shape[0] * shape[1] * shape[2] / 1e9:.2f} GB raw)", flush=True)
    src = remote(f"{SCANS['PHerc0846A']}/1")
    path = D / (f"trace_L1_{a.tag}.zarr" if a.tag else "trace_L1.zarr")
    dst = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)},
                   "metadata": {"shape": shape, "dtype": "|u1", "chunks": [128, 128, 128],
                                "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}},
                   "create": True, "delete_existing": True}).result()
    for z in range(0, shape[0], 128):
        dst[z:z + 128].write(src[lo["z"] + z:lo["z"] + z + 128, lo["y"]:hi["y"], lo["x"]:hi["x"]].read().result()).result()
        print(f"  slab {z}..{z + 128} of {shape[0]}", flush=True)
    (D / (f"region_L1_{a.tag}.json" if a.tag else "region_L1.json")).write_text(json.dumps({"scroll": "PHerc0846A", "level": 1, "um_per_voxel": 2.403 * 2,
                                                  "z0": lo["z"], "y0": lo["y"], "x0": lo["x"], "shape": shape,
                                                  "note": "level-1 voxel coords; level-0 = x2"}, indent=2))
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
