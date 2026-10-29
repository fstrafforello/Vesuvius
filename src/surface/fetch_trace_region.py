"""Step c1: download the tracing region of PHerc0846A at level 2 (9.6 um/voxel).

Region: the lower third of the scroll (about 3.7-13.5 mm up), a 2048 x 2048 (about 20 x 20 mm)
window centred on the papyrus centre of mass there, which sits on the spiral's inner wraps.
Saved as data/scout/PHerc0846A/trace_L2.zarr, plus region.json and check figures.

Usage:  uv run python src/surface/fetch_trace_region.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "surface")]
from fetch_starter import remote  # noqa: E402
from scout import OUT, SCANS, local, stretch  # noqa: E402

SCROLL = "PHerc0846A"
LEVEL = 2
UM = 2.403 * 2 ** LEVEL
Z0, Z1 = 384, 1408          # level-2 slices, chunk aligned: 3.7 - 13.5 mm
SIZE = 2048                 # in-plane window, level-2 voxels


def main():
    d = OUT / SCROLL
    l4 = local(SCROLL, 4)                                   # find the centre on the small level-4 copy
    sl = l4[Z0 // 4:Z1 // 4].read().result()
    m = sl > 0
    zz, yy, xx = np.nonzero(m[::4, ::2, ::2])
    cy, cx = int(yy.mean() * 2 * 4), int(xx.mean() * 2 * 4)  # level-4 -> level-2 (x4)
    src = remote(f"{SCANS[SCROLL]}/{LEVEL}")
    y0 = int(np.clip(cy - SIZE // 2, 0, src.shape[1] - SIZE)) // 128 * 128
    x0 = int(np.clip(cx - SIZE // 2, 0, src.shape[2] - SIZE)) // 128 * 128
    print(f"L2 volume {src.shape}; papyrus centre (L2) y={cy} x={cx}; window y {y0}..{y0 + SIZE}, x {x0}..{x0 + SIZE}, z {Z0}..{Z1}", flush=True)

    path = d / "trace_L2.zarr"
    if not path.exists():
        dst = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)},
                       "metadata": {"shape": [Z1 - Z0, SIZE, SIZE], "dtype": "|u1", "chunks": [128, 128, 128],
                                    "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}},
                       "create": True}).result()
        for z in range(Z0, Z1, 128):                          # slab by slab, to keep memory low
            dst[z - Z0:z - Z0 + 128].write(src[z:z + 128, y0:y0 + SIZE, x0:x0 + SIZE].read().result()).result()
            print(f"  slab z {z}..{z + 128} done", flush=True)
    (d / "region.json").write_text(json.dumps({"scroll": SCROLL, "level": LEVEL, "um_per_voxel": UM,
                                               "z0": Z0, "y0": y0, "x0": x0, "shape": [Z1 - Z0, SIZE, SIZE],
                                               "note": "level-2 voxel coords; level-0 = x4"}, indent=2))

    v = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}}, read=True).result()
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
    for a, f in zip(ax, (0.2, 0.5, 0.8)):
        z = int((Z1 - Z0) * f)
        a.imshow(stretch(v[z].read().result()), cmap="gray")
        a.set_title(f"z {(Z0 + z) * UM / 1000:.1f} mm (L2, {UM:.1f} um/voxel, 20 mm wide)"); a.axis("off")
    plt.tight_layout(); plt.savefig(d / "trace_L2_axial.png", dpi=70); plt.close()

    mid = v[(Z1 - Z0) // 2].read().result()
    c = SIZE // 2
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
    for a, (oy, ox) in zip(ax, ((-512, -768), (-256, 256), (256, -256))):
        a.imshow(stretch(mid[c + oy:c + oy + 512, c + ox:c + ox + 512]), cmap="gray")
        a.set_title(f"512 px = {512 * UM / 1000:.1f} mm crop"); a.axis("off")
    plt.tight_layout(); plt.savefig(d / "trace_L2_zoom.png", dpi=70); plt.close()

    sec = v[:, c, :].read().result()
    plt.figure(figsize=(12, 6)); plt.imshow(stretch(sec), cmap="gray", aspect="auto")
    plt.title("long section through the window (z vertical)"); plt.axis("off")
    plt.tight_layout(); plt.savefig(d / "trace_L2_section.png", dpi=70); plt.close()
    print("saved figures in", d)


if __name__ == "__main__":
    main()
