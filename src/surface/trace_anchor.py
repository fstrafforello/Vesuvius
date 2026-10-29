"""Step d, version 2: anchor-based tracing.

Instead of carrying hundreds of indexed points from slice to slice (which loses them a few at a
time), track ONE anchor point on the sheet through the slices and, in every slice, run the full
single-slice tracer (with bridging of small breaks) from that anchor. Each slice's curve is laid
into the grid by arc length from the anchor, so columns line up across heights and nothing is
lost by accumulation.

Per slice:
  1. move the anchor across the sheet to its centre in the new slice (small snap)
  2. trace the sheet both ways from the anchor
  3. orient the curve like the previous one and register it by arc length from the anchor
  4. jump check: every point must lie within MAX_DEV voxels of the previous slice's curve
     (measured to the nearest point, i.e. across the sheet); points further away -> invalid

Output: data/scout/PHerc0846A/trace_<name>/ in the same format as trace.py.

Usage:  uv run python src/surface/trace_anchor.py --name anchor [--z 620 --seed-y 1153 --seed-x 663]
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree

from trace import D, STEP, load, local_tangent, orientation_field, ridge_snap, save_trace, trace_slice

WIDTH = 3600          # grid columns (level-2 voxels along the sheet), anchor in the middle
MAX_DEV = 3.0         # voxels a point may sit from the previous slice's curve
PAD = 120             # margin around the previous curve for the per-slice orientation field


def slice_trace(block, z, anchor, prev_dir, prev_curve, thr, half_len, bridge, max_turn, snap=True):
    """Trace slice z from the anchor. Returns (anchor, curve, anchor_index, direction) or None."""
    if prev_curve is not None and len(prev_curve):
        lo = np.maximum(np.floor(prev_curve.min(0)).astype(int) - PAD, 0)
        hi = np.minimum(np.ceil(prev_curve.max(0)).astype(int) + PAD, block.shape[1:])
    else:
        lo = np.maximum(anchor.astype(int) - half_len - PAD, 0)
        hi = np.minimum(anchor.astype(int) + half_len + PAD, block.shape[1:])
    img = ndimage.gaussian_filter(block[z, lo[0]:hi[0], lo[1]:hi[1]].read().result().astype(np.float32), 1.0)
    field = orientation_field(img)
    a = anchor - lo
    t = local_tangent(field, a, prev_dir)
    if snap:
        # small snap only (the sheet moves < 1 voxel per slice); a wide window could land on a
        # neighbouring sheet
        a, val, moved = ridge_snap(img, a, np.array([-t[1], t[0]]), r=1.0)
        if val < thr:
            return None
    t = local_tangent(field, a, prev_dir)
    curve = trace_slice(img, a, t, half_len, thr, max_turn_deg=max_turn, field=field, bridge=bridge)
    if len(curve) < 20:
        return None
    k = int(np.argmin(np.linalg.norm(curve - a, axis=1)))
    # orient like the previous slice: curve index must increase along prev_dir at the anchor
    d = curve[min(k + 3, len(curve) - 1)] - curve[max(k - 3, 0)]
    if np.dot(d, prev_dir) < 0:
        curve = curve[::-1]
        k = len(curve) - 1 - k
        d = -d
    return a + lo, curve + lo, k, d / max(np.linalg.norm(d), 1e-6)


ANCHOR_EVERY = 120    # columns between anchors (1.2 mm)
PIECE_LEN = 250       # each anchor traces this far each way (2.4 mm)


def multi_anchor_trace(block, z, ref_pos, ref_age, fresh_cols, a):
    """Trace slice z from many anchors spread along the reference; return all points (N, 2)."""
    pts = ref_pos[fresh_cols]
    lo = np.maximum(np.floor(pts.min(0)).astype(int) - PAD, 0)
    hi = np.minimum(np.ceil(pts.max(0)).astype(int) + PAD, block.shape[1:])
    img = ndimage.gaussian_filter(block[z, lo[0]:hi[0], lo[1]:hi[1]].read().result().astype(np.float32), 1.0)
    field = orientation_field(img)
    recent = fresh_cols[ref_age[fresh_cols] <= 3]
    if len(recent) < 10:
        recent = fresh_cols
    anchors = recent[::ANCHOR_EVERY] if len(recent) > ANCHOR_EVERY else recent[[len(recent) // 2]]
    pieces = []
    for c in anchors:
        p = ref_pos[c] - lo
        nb = ref_pos[min(c + 3, len(ref_pos) - 1)] - ref_pos[max(c - 3, 0)]
        nb = nb if np.all(np.isfinite(nb)) and np.linalg.norm(nb) > 0 else np.array([0.0, 1.0])
        t = local_tangent(field, p, nb / np.linalg.norm(nb))
        piece = trace_slice(img, p, t, PIECE_LEN, a.thr, max_turn_deg=a.max_turn, field=field, bridge=a.bridge)
        if len(piece) >= 30:
            pieces.append(piece + lo)
    return np.concatenate(pieces) if pieces else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="anchor")
    ap.add_argument("--z", type=int, default=620)
    ap.add_argument("--seed-y", type=float, default=1153)
    ap.add_argument("--seed-x", type=float, default=663)
    ap.add_argument("--thr", type=float, default=83)
    ap.add_argument("--half-len", type=int, default=1500)
    ap.add_argument("--bridge", type=int, default=6)
    ap.add_argument("--max-turn", type=float, default=45)
    ap.add_argument("--zspan", type=int, default=None)
    ap.add_argument("--level", type=int, default=2)
    a = ap.parse_args()

    block, region, _ = load(a.level)
    # anchor spacing and piece length are defined at level 2; keep the same physical sizes
    global ANCHOR_EVERY, PIECE_LEN, PAD, WIDTH
    k = 2 ** (2 - a.level)
    ANCHOR_EVERY, PIECE_LEN, PAD, WIDTH = ANCHOR_EVERY * k, PIECE_LEN * k, PAD * k, WIDTH * k
    nz = block.shape[0]
    span = a.zspan if a.zspan is not None else nz
    z_range = list(range(max(0, a.z - span), min(nz, a.z + span + 1)))
    i0 = z_range.index(a.z)
    grid = np.full((len(z_range), WIDTH, 2), np.nan)
    valid = np.zeros((len(z_range), WIDTH), bool)
    mid = WIDTH // 2
    t0 = time.time()

    first = slice_trace(block, a.z, np.array([a.seed_y, a.seed_x]), np.array([0.0, 1.0]), None,
                        a.thr, a.half_len, a.bridge, a.max_turn, snap=False)
    if first is None:
        raise SystemExit("no sheet at the seed")
    anchor0, curve0, k0, dir0 = first
    cols = mid + np.arange(len(curve0)) - k0
    keep = (cols >= 0) & (cols < WIDTH)
    grid[i0, cols[keep]], valid[i0, cols[keep]] = curve0[keep], True
    print(f"start slice {a.z}: {len(curve0)} points = {len(curve0) * STEP * region['um_per_voxel'] / 1000:.1f} mm", flush=True)

    for step in (1, -1):
        anchor, d, prev = anchor0, dir0, curve0
        # column-wise reference: last good position of every grid column and when it was seen
        ref_pos = np.full((WIDTH, 2), np.nan)
        ref_age = np.full(WIDTH, 10**9)
        c0 = mid + np.arange(len(curve0)) - k0
        m0 = (c0 >= 0) & (c0 < WIDTH)
        ref_pos[c0[m0]], ref_age[c0[m0]] = curve0[m0], 0
        misses, i = 0, i0 + step
        while 0 <= i < len(z_range):
            ref_age += 1
            fresh_cols = np.nonzero(ref_age <= 20)[0]
            if len(fresh_cols) < 30:
                print(f"  reference lost at slice {z_range[i]}", flush=True)
                break
            curve = multi_anchor_trace(block, z_range[i], ref_pos, ref_age, fresh_cols, a)
            if curve is None:
                misses += 1
                if misses > 5:
                    print(f"  no sheet found near the reference at slice {z_range[i]}", flush=True)
                    break
                i += step
                continue
            misses = 0
            dist, idx = cKDTree(ref_pos[fresh_cols]).query(curve)
            ok = dist <= MAX_DEV
            # keep every well-matched stretch of useful length (>= 30 points); isolated matches
            # could be a crossing with another sheet
            lab, nlab = ndimage.label(ok)
            if nlab:
                sizes = ndimage.sum(ok, lab, range(1, nlab + 1))
                ok = np.isin(lab, np.nonzero(sizes >= 30)[0] + 1)
            # column = column of the nearest reference point on this sheet (robust to detours)
            cols = fresh_cols[idx]
            order = np.argsort(dist)                          # if two points claim a column, the closer wins
            seen = np.zeros(WIDTH, bool)
            for j in order:
                if ok[j] and not seen[cols[j]]:
                    seen[cols[j]] = True
                    grid[i, cols[j]] = curve[j]; valid[i, cols[j]] = True
                    ref_pos[cols[j]], ref_age[cols[j]] = curve[j], 0
            if abs(i - i0) % 100 == 0:
                print(f"  slice {z_range[i]}: {seen.sum()} points ({seen.sum() * STEP * region['um_per_voxel'] / 1000:.1f} mm), "
                      f"{time.time() - t0:.0f}s", flush=True)
            i += step

    rows = valid.any(1)
    used = np.nonzero(valid.any(0))[0]
    grid, valid = grid[:, used[0]:used[-1] + 1], valid[:, used[0]:used[-1] + 1]
    area = valid.sum() * (region["um_per_voxel"] / 1000) ** 2 * STEP
    print(f"traced {rows.sum()} slices = {rows.sum() * region['um_per_voxel'] / 1000:.1f} mm of height, "
          f"{grid.shape[1] * STEP * region['um_per_voxel'] / 1000:.1f} mm wide; valid {valid.mean():.1%} = {area:.0f} mm^2", flush=True)

    out = save_trace(a.name, grid, valid, z_range, region, "psn trace_anchor.py",
                     {"seed_block_zyx": [a.z, a.seed_y, a.seed_x], "threshold": a.thr})

    fig, ax = plt.subplots(2, 4, figsize=(26, 13))
    rows_i = np.nonzero(rows)[0]
    for a_, f in zip(ax.flat[:7], np.linspace(0.02, 0.98, 7)):
        i = rows_i[int((len(rows_i) - 1) * f)]
        im = block[z_range[i]].read().result()
        pts = grid[i][valid[i]]
        cy, cx = pts[:, 0].mean(), pts[:, 1].mean()
        h = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 + 60
        a_.imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im, 99.5))
        a_.plot(pts[:, 1], pts[:, 0], "r.", ms=1)
        a_.set_xlim(cx - h, cx + h); a_.set_ylim(cy + h, cy - h)
        a_.set_title(f"slice {z_range[i]} ({(z_range[i] + region['z0']) * region['um_per_voxel'] / 1000:.1f} mm), {valid[i].sum()} pts")
    ax.flat[7].imshow(valid, aspect="auto", cmap="gray"); ax.flat[7].set_title("valid grid (rows = height, cols = along sheet)")
    plt.tight_layout(); plt.savefig(out / "trace_check.png", dpi=60); plt.close()
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
