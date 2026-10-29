"""Vertical check of a traced surface: does it stay on the SAME sheet from height to height?

Per-slice overlays only show that every slice sits on *a* sheet. Here, for a few columns of the
trace grid (positions along the sheet), we cut a section through the scan along the scroll axis
and across the sheet, and draw the trace's path through the heights on it. A trace that stays on
one sheet runs smoothly along one bright band; a hop is a step across dark gaps to another band.

Automatic count over ALL columns: a hop is a move across the sheet of more than HOP voxels per
slice between consecutive valid rows (real sheets here run nearly parallel to the scroll axis).

Usage:  uv run python src/surface/section_check.py --name l1_filled [--cols 6]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from trace import D, load, trace_level, trace_tag  # noqa: E402

HOP = 1.5      # voxels per slice across the sheet (level 2; scaled per level)
SLOPE = 0.3    # sideways drift a sheet can make per slice across a gap (measured: 0.1-0.25)
HALF = 60      # section half-width across the sheet (level 2 voxels; scaled per level)


def across_offsets(grid, valid, c, nrm):
    """Signed offset across the sheet (along nrm) of column c's point in every row."""
    p = grid[:, c]
    ref = np.nanmedian(p[valid[:, c]], axis=0)
    return (p - ref) @ nrm, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cols", type=int, default=6)
    a = ap.parse_args()
    t = np.load(D / f"trace_{a.name}" / "trace_block_coords.npz")
    grid, valid, zs = t["grid"].astype(np.float64), t["valid"], t["z"]
    level = trace_level(D / f"trace_{a.name}")
    block, region, _ = load(level, trace_tag(D / f"trace_{a.name}"))
    k = 2 ** (2 - level)
    hop, half = HOP * k, HALF * k

    # in-plane sheet normal per column (median tangent over the rows)
    tang = np.gradient(grid, axis=1)
    tang /= np.maximum(np.linalg.norm(tang, axis=-1, keepdims=True), 1e-6)

    # automatic hop count over all columns
    hops, steps, hop_cols = 0, 0, np.zeros(grid.shape[1], int)
    for c in range(grid.shape[1]):
        rows = np.nonzero(valid[:, c])[0]
        if len(rows) < 2:
            continue
        tc = np.nanmedian(tang[rows, c], axis=0)
        n = np.array([-tc[1], tc[0]]) / max(np.linalg.norm(tc), 1e-6)
        off, _ = across_offsets(grid, valid, c, n)
        # a sheet can tilt by up to ~1 voxel per slice; consecutive slices allow `hop`, across a gap
        # of g slices allow 1.0 * g + a margin. Anything more is a jump to another sheet.
        gap = np.diff(rows)
        limit = np.where(gap == 1, hop, SLOPE * gap + 4.0 * k)
        big = np.abs(np.diff(off[rows])) > limit
        hops += int(big.sum()); steps += len(gap); hop_cols[c] = int(big.sum())
    cols_with = (hop_cols > 0).sum() / max((valid.sum(0) > 1).sum(), 1)
    print(f"{a.name}: {hops} hops in {steps} slice-to-slice steps ({hops / max(steps, 1):.2%}); "
          f"{cols_with:.0%} of columns contain at least one hop", flush=True)

    # sections through a few well-covered columns
    cover = valid.sum(0)
    cand = np.nonzero(cover >= np.percentile(cover[cover > 0], 50))[0]
    pick = cand[np.linspace(0, len(cand) - 1, a.cols).astype(int)]
    fig, ax = plt.subplots(1, len(pick), figsize=(4 * len(pick), 9))
    s = np.arange(-half, half + 1)
    info = []
    for c in pick:
        rows = np.nonzero(valid[:, c])[0]
        tc = np.nanmedian(tang[rows, c], axis=0)
        n = np.array([-tc[1], tc[0]]) / max(np.linalg.norm(tc), 1e-6)
        off, ref = across_offsets(grid, valid, c, n)
        info.append((c, rows, n, off, ref, ref[None] + s[:, None] * n[None]))
    r0 = min(i[1][0] for i in info); r1 = max(i[1][-1] for i in info) + 1
    imgs = [np.zeros((r1 - r0, len(s)), np.float32) for _ in info]
    for j, i in enumerate(range(r0, r1)):                      # read every slice once for all sections
        sl = block[int(zs[i])].read().result().astype(np.float32)
        for img, (_, _, _, _, _, pts) in zip(imgs, info):
            img[j] = ndimage.map_coordinates(sl, pts.T, order=1, mode="constant")
    for a_, img, (c, rows, n, off, ref, _) in zip(np.atleast_1d(ax), imgs, info):
        a_.imshow(img, cmap="gray", aspect="auto", extent=(-half, half, zs[r1 - 1], zs[r0]), vmin=0,
                  vmax=np.percentile(img, 99.5))
        a_.plot(off[rows], zs[rows], "r.", ms=1)
        a_.set_title(f"column {c}: {hop_cols[c]} hops")
        a_.set_xlabel("across the sheet (voxels)"); a_.set_ylabel("slice (height)")
    plt.suptitle(f"{a.name}: vertical sections (red = trace). One smooth band = one sheet; steps = hops")
    plt.tight_layout(); plt.savefig(D / f"trace_{a.name}" / "section_check.png", dpi=55); plt.close()
    (D / f"trace_{a.name}" / "section_check.json").write_text(json.dumps(
        {"hops": hops, "steps": steps, "hop_rate": hops / max(steps, 1), "cols_with_hops": float(cols_with)}, indent=2))
    print("saved", D / f"trace_{a.name}" / "section_check.png", flush=True)


if __name__ == "__main__":
    main()
