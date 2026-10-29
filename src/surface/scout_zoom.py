"""Closer look at a candidate height: one level-3 (19 um) slab across the whole scroll, saved
locally, plus a full slice and zoomed crops to judge whether single layers can be followed.

Usage:  uv run python src/surface/scout_zoom.py PHerc0846A --z-mm 10.9
"""

import argparse
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
from scout import OUT, SCANS, stretch  # noqa: E402

LEVEL, SLAB = 3, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scroll")
    ap.add_argument("--z-mm", type=float, required=True)
    a = ap.parse_args()
    um = 2.403 * 2 ** LEVEL
    src = remote(f"{SCANS[a.scroll]}/{LEVEL}")
    z0 = int(a.z_mm * 1000 / um) // SLAB * SLAB            # chunk-aligned
    d = OUT / a.scroll
    path = d / f"L3_z{z0}"
    if not path.exists():
        slab = src[z0:z0 + SLAB].read().result()
        ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)},
                 "metadata": {"shape": list(slab.shape), "dtype": "|u1", "chunks": [SLAB, 256, 256],
                              "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}},
                 "create": True}).result().write(slab).result()
    slab = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}}, read=True).result().read().result()
    mid = slab[SLAB // 2]
    print(f"{a.scroll}: L3 slab z {z0}..{z0 + SLAB} ({z0 * um / 1000:.1f}-{(z0 + SLAB) * um / 1000:.1f} mm), slice shape {mid.shape}")

    # full slice with a grid of 512 px boxes, and the 6 boxes with most papyrus edges (layer-rich)
    img = stretch(mid)
    box = 512
    gy, gx = mid.shape[0] // box, mid.shape[1] // box
    edge = np.abs(np.diff(img, axis=1))[:, :-0 or None]
    scores = []
    for i in range(gy):
        for j in range(gx):
            t = mid[i * box:(i + 1) * box, j * box:(j + 1) * box]
            fill = (t > 0).mean()
            if fill > 0.9:
                e = edge[i * box:(i + 1) * box, j * box:min((j + 1) * box, edge.shape[1])].mean()
                scores.append((e, i, j))
    scores.sort(reverse=True)
    pick = scores[:6]

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(img, cmap="gray")
    for n, (_, i, j) in enumerate(pick):
        ax.add_patch(plt.Rectangle((j * box, i * box), box, box, fill=False, ec="red", lw=1.5))
        ax.text(j * box + 10, i * box + 40, str(n + 1), color="red", fontsize=14)
    ax.set_title(f"{a.scroll} at {z0 * um / 1000 + SLAB * um / 2000:.1f} mm, level 3 ({um:.0f} um/voxel); boxes = 512 px = {box * um / 1000:.1f} mm")
    ax.axis("off")
    plt.tight_layout(); plt.savefig(d / f"zoom_z{z0}_overview.png", dpi=70); plt.close()

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for a_, (n, (_, i, j)) in zip(ax.flat, enumerate(pick)):
        a_.imshow(img[i * box:(i + 1) * box, j * box:(j + 1) * box], cmap="gray")
        a_.set_title(f"box {n + 1}  (y {i * box}, x {j * box} at L3)"); a_.axis("off")
    plt.tight_layout(); plt.savefig(d / f"zoom_z{z0}_boxes.png", dpi=70); plt.close()
    print("saved", d / f"zoom_z{z0}_overview.png", "and _boxes.png")


if __name__ == "__main__":
    main()
