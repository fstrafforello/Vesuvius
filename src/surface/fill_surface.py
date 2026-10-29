"""Step d3: turn a traced stack of curves into one continuous surface.

The tracer judges every slice on its own, so validity flickers from slice to slice. Papyrus is
continuous along the scroll height too, so here the trace is treated as a 2D surface:
  1. fill holes along the height (gaps up to MAX_ZGAP rows) and along the sheet (up to MAX_UGAP
     points) by interpolating between valid neighbours
  2. smooth the surface in 2D (validity-weighted)
  3. re-snap every point across the sheet in its own slice; keep only points that land on papyrus
     close to where the surface predicted
  4. keep the largest connected piece

Writes data/scout/PHerc0846A/trace_<name>_filled/ in the same format as trace.py (tifxyz in
level-0 coordinates + block coordinates), plus overlay checks.

Usage:  uv run python src/surface/fill_surface.py --name test [--thr 83]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from trace import D, load, ridge_snap, save_trace, trace_level  # noqa: E402

MAX_ZGAP, MAX_UGAP = 60, 30
SIGMA = 2.0
MAX_MOVE = 1.5
NEIGH = 9            # 2D window (points) for neighbourhood acceptance after re-snap
MIN_PIECE = 5000     # keep every connected piece at least this big (points)


def fill_1d(arr, ok, max_gap, axis, max_slope=None, margin=2.0):
    """Linear interpolation over invalid runs of length <= max_gap along `axis` (2 channels last).
    With max_slope, a gap is only bridged if the two ends are consistent with one sheet: they may be
    at most max_slope * gap + margin voxels apart (sheets tilt, but cannot jump)."""
    out, ok_out = arr.copy(), ok.copy()
    a = np.moveaxis(out, axis, 0); m = np.moveaxis(ok_out, axis, 0)
    for j in range(a.shape[1]):
        col, v = a[:, j], m[:, j]
        idx = np.nonzero(v)[0]
        if len(idx) < 2:
            continue
        for s, e in zip(idx[:-1], idx[1:]):
            if max_slope is not None and np.linalg.norm(col[e] - col[s]) > max_slope * (e - s) + margin:
                continue
            if 1 < e - s <= max_gap + 1:
                w = (np.arange(s + 1, e) - s) / (e - s)
                col[s + 1:e] = (1 - w)[:, None] * col[s] + w[:, None] * col[e]
                v[s + 1:e] = True
    return out, ok_out


def smooth(arr, ok, sigma):
    w = ndimage.gaussian_filter(ok.astype(np.float64), sigma)
    out = arr.copy()
    for c in (0, 1):
        num = ndimage.gaussian_filter(np.where(ok, arr[..., c], 0.0), sigma)
        out[..., c] = np.where(ok, num / np.maximum(w, 1e-6), arr[..., c])
    return out


def continuous_piece(g, ok, zs, src, k, max_row=1.0, max_col=2.5):
    """Connected component of `ok` where grid neighbours are linked only if close in 3D
    (row neighbours: <= max_row voxels per slice + 1; column neighbours: <= max_col voxels).
    Returns the component containing the trace's start slice (largest component as fallback)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    H, W = ok.shape
    idx = -np.ones((H, W), int)
    idx[ok] = np.arange(ok.sum())
    ei, ej = [], []
    dr = np.linalg.norm(g[1:] - g[:-1], axis=-1)
    m = ok[1:] & ok[:-1] & (dr <= max_row * np.diff(zs)[:, None] + 1.0)
    ei.append(idx[:-1][m]); ej.append(idx[1:][m])
    dc = np.linalg.norm(g[:, 1:] - g[:, :-1], axis=-1)
    m = ok[:, 1:] & ok[:, :-1] & (dc <= max_col)
    ei.append(idx[:, :-1][m]); ej.append(idx[:, 1:][m])
    ei, ej = np.concatenate(ei), np.concatenate(ej)
    n = int(ok.sum())
    _, lab = connected_components(coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(n, n)), directed=False)
    labels = -np.ones((H, W), int); labels[ok] = lab
    meta = json.loads((src / "meta.json").read_text())
    seed_z = meta.get("seed_block_zyx", [None])[0]
    if seed_z is not None and seed_z in set(zs.tolist()):
        row = int(np.nonzero(zs == seed_z)[0][0])
        vals, counts = np.unique(labels[row][labels[row] >= 0], return_counts=True)
        if len(vals):
            return labels == vals[np.argmax(counts)]
    vals, counts = np.unique(lab, return_counts=True)
    return labels == vals[np.argmax(counts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="test")
    ap.add_argument("--thr", type=float, default=83)
    a = ap.parse_args()
    src = D / f"trace_{a.name}"
    t = np.load(src / "trace_block_coords.npz")
    grid, valid, zs = t["grid"].copy(), t["valid"].copy(), t["z"]
    grid[~valid] = 0
    print(f"input: valid {valid.mean():.1%}", flush=True)
    # tolerances are defined at level 2 (9.6 um); scale them so they mean the same physical size
    k = 2 ** (2 - trace_level(src))
    zgap, ugap, neigh, min_piece, max_move = MAX_ZGAP * k, MAX_UGAP * k, NEIGH * k + (k > 1), MIN_PIECE * k * k, MAX_MOVE * k

    # along the height, only bridge gaps whose ends are consistent with one tilting sheet
    g, ok = fill_1d(grid, valid, zgap, axis=0, max_slope=0.3, margin=4.0 * k)
    g, ok = fill_1d(g, ok, ugap, axis=1, max_slope=1.5, margin=2.0 * k)
    print(f"after hole filling: {ok.mean():.1%}", flush=True)
    g = smooth(g, ok, SIGMA)

    block, region, _ = load(trace_level(src))
    snapped, keep = g.copy(), np.zeros_like(ok)
    for i in np.nonzero(ok.any(1))[0]:
        img = ndimage.gaussian_filter(block[int(zs[i])].read().result().astype(np.float32), 1.0)
        row, m = g[i], ok[i]
        tang = np.gradient(row, axis=0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        for j in np.nonzero(m)[0]:
            p, val, moved = ridge_snap(img, row[j], nrm[j], r=max_move + 0.5)
            if val >= a.thr and moved <= max_move:
                snapped[i, j] = p; keep[i, j] = True
    print(f"after re-snap: {keep.mean():.1%}", flush=True)
    # neighbourhood acceptance: a point that missed is still kept (at the smoothed position of its
    # neighbours) when most points around it in 2D did land on papyrus
    rate = ndimage.uniform_filter(keep.astype(np.float32), neigh) / np.maximum(
        ndimage.uniform_filter(ok.astype(np.float32), neigh), 1e-6)
    rescued = ok & ~keep & (rate >= 0.6)
    snapped = np.where(rescued[..., None], smooth(snapped, keep, 2.0), snapped)
    keep = keep | rescued
    print(f"after neighbourhood acceptance: {keep.mean():.1%} (+{rescued.mean():.1%})", flush=True)
    snapped = smooth(snapped, keep, 1.0)
    # connectivity with real 3D continuity: neighbouring grid points only count as connected if
    # they are also close in space. Pieces on different sheets then fall apart, and we keep the
    # piece that contains the start slice (the sheet we seeded on).
    keep = continuous_piece(snapped, keep, zs, src, k)
    print(f"after 3D-continuity (seed piece only): {keep.mean():.1%}", flush=True)
    rows = np.nonzero(keep.any(1))[0]
    print(f"final: {keep.mean():.1%} of grid valid; height {len(rows)} slices; "
          f"largest piece {keep.sum()} points", flush=True)

    out = save_trace(f"{a.name}_filled", snapped, keep, list(zs), region, "psn fill_surface.py")

    fig, ax = plt.subplots(2, 3, figsize=(20, 13))
    for a_, f in zip(ax.flat, np.linspace(0.05, 0.95, 6)):
        i = rows[int((len(rows) - 1) * f)]
        im = block[int(zs[i])].read().result()
        pts = snapped[i][keep[i]]
        cy, cx = pts[:, 0].mean(), pts[:, 1].mean()
        h = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 + 60
        a_.imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im, 99.5))
        a_.plot(pts[:, 1], pts[:, 0], "r.", ms=1)
        a_.set_xlim(cx - h, cx + h); a_.set_ylim(cy + h, cy - h)
        a_.set_title(f"slice {zs[i]} ({(zs[i] + region['z0']) * region['um_per_voxel'] / 1000:.1f} mm), {keep[i].sum()} pts")
    plt.tight_layout(); plt.savefig(out / "trace_check.png", dpi=65); plt.close()
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
