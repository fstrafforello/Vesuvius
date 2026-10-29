"""Quick flattened preview of a trace, from the level-2 block (no full-resolution data needed).

Samples the block on the traced grid itself (rows = height, cols = along the sheet) at a few
offsets across the sheet, so we can judge whether the trace stays on ONE layer: a single sheet
gives continuous papyrus texture; a slip onto a neighbouring sheet shows as a seam or a band of gap.

Usage:  uv run python src/surface/flatten_preview.py --name test
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts
from scipy import ndimage

D = Path(__file__).resolve().parents[2] / "data" / "scout" / "PHerc0846A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="test")
    a = ap.parse_args()
    t = np.load(D / f"trace_{a.name}" / "trace_block_coords.npz")
    grid, valid, zs = t["grid"], t["valid"], t["z"]
    sys.path.insert(0, str(Path(__file__).parent))
    from trace import load, trace_level
    block, _, _ = load(trace_level(D / f"trace_{a.name}"))
    rows = np.nonzero(valid.any(1))[0]
    zlo, zhi = zs[rows[0]], zs[rows[-1]] + 1
    ylo, yhi = max(int(np.nanmin(grid[valid][:, 0])) - 4, 0), min(int(np.nanmax(grid[valid][:, 0])) + 5, block.shape[1])
    xlo, xhi = max(int(np.nanmin(grid[valid][:, 1])) - 4, 0), min(int(np.nanmax(grid[valid][:, 1])) + 5, block.shape[2])
    sub = block[zlo:zhi, ylo:yhi, xlo:xhi].read().result().astype(np.float32)

    # in-slice normals of each row's curve, to sample across the sheet
    tang = np.gradient(grid, axis=1)
    tang /= np.maximum(np.linalg.norm(tang, axis=-1, keepdims=True), 1e-6)
    nrm = np.stack([-tang[..., 1], tang[..., 0]], -1)
    zz = (zs[:, None] - zlo) * np.ones(grid.shape[1])[None]
    offs = (-3, -1.5, 0, 1.5, 3)
    imgs = []
    for o in offs:
        p = grid + o * nrm
        v = ndimage.map_coordinates(sub, [zz, p[..., 0] - ylo, p[..., 1] - xlo], order=1, mode="constant")
        imgs.append(np.where(valid, v, np.nan))
    fig, ax = plt.subplots(len(offs), 1, figsize=(14, 3.2 * len(offs)))
    for a_, o, im in zip(ax, offs, imgs):
        a_.imshow(im[rows[0]:rows[-1] + 1], cmap="gray", vmin=40, vmax=200, aspect="auto")
        a_.set_title(f"flattened trace, offset {o:+.1f} voxels across the sheet (rows = height {zs[rows[0]]}..{zs[rows[-1]]}, cols = along sheet)")
        a_.axis("off")
    plt.tight_layout(); plt.savefig(D / f"trace_{a.name}" / "flat_preview.png", dpi=60); plt.close()
    on = imgs[2][valid]
    print(f"on-sheet mean {np.nanmean(on):.0f}; fraction of valid points sampling gap (<83): {np.mean(on < 83):.1%}")
    print("saved", D / f"trace_{a.name}" / "flat_preview.png")


if __name__ == "__main__":
    main()
