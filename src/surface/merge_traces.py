"""Merge several traces of the SAME sheet (started at different heights) into one surface.

Each extra trace has its own column numbering (arc length from its own seed). Where it overlaps
the base surface in the same slice, matching points give the column mapping; a trace is only
used if, where they overlap, it really lies on the base sheet (points within MATCH voxels).
Its points then fill the base surface's holes (and extend it at the ends).

Usage:  uv run python src/surface/merge_traces.py --base l1_filled --others s352_filled s605_filled ... --out merged
        then: uv run python src/surface/fill_surface.py --name merged
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from trace import D, load, save_trace, trace_level  # noqa: E402

MATCH = 2.0          # voxels: a point is "on the base sheet" if this close to a base point in the same slice
WINDOW = 150         # columns per offset estimate
MIN_PAIRS = 40


def load_trace(name):
    t = np.load(D / f"trace_{name}" / "trace_block_coords.npz")
    return t["grid"].astype(np.float64), t["valid"], t["z"]


def pairs(base_g, base_v, g, v):
    """(row, col_other, col_base, dist) for every point of `other` near a base point in its row."""
    out = []
    for i in np.nonzero(v.any(1) & base_v.any(1))[0]:
        bc = np.nonzero(base_v[i])[0]
        oc = np.nonzero(v[i])[0]
        d, j = cKDTree(base_g[i, bc]).query(g[i, oc])
        out.append(np.stack([np.full(len(oc), i), oc, bc[j], d], 1))
    return np.concatenate(out) if out else np.zeros((0, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--others", nargs="+", required=True)
    ap.add_argument("--out", default="merged")
    a = ap.parse_args()

    G, V, Z = load_trace(a.base)
    level = trace_level(D / f"trace_{a.base}")
    _, region, _ = load(level)
    print(f"base {a.base}: {V.sum()} points, width {G.shape[1]}", flush=True)
    for name in a.others:
        g, v, z = load_trace(name)
        assert np.array_equal(z, Z), "traces must cover the same slices"
        p = pairs(G, V, g, v)
        if len(p) == 0:
            print(f"{name}: no overlap with the base - skipped"); continue
        close = p[p[:, 3] <= MATCH]
        frac = len(close) / len(p)
        print(f"{name}: {v.sum()} points; {len(p)} share a slice with the base, {frac:.0%} of those lie on the base sheet", flush=True)
        if frac < 0.6 or len(close) < MIN_PAIRS:
            print("  -> not the same sheet (or too little overlap): skipped"); continue
        # direction of column numbering: +1 or -1, whichever gives a consistent offset
        best = None
        for s in (1, -1):
            off = close[:, 2] - s * close[:, 1]
            mad = np.median(np.abs(off - np.median(off)))
            if best is None or mad < best[0]:
                best = (mad, s)
        s = best[1]
        # offset per column window (arc length drifts slowly between traces), interpolated
        centres, offsets = [], []
        for c0 in range(0, g.shape[1], WINDOW):
            m = (close[:, 1] >= c0) & (close[:, 1] < c0 + WINDOW)
            if m.sum() >= MIN_PAIRS:
                centres.append(c0 + WINDOW / 2)
                offsets.append(np.median(close[m, 2] - s * close[m, 1]))
        if not centres:
            print("  -> no window with enough matches: skipped"); continue
        cols_o = np.arange(g.shape[1])
        mapped = np.rint(s * cols_o + np.interp(cols_o, centres, offsets)).astype(int)
        # grow the base grid if the other trace reaches beyond it
        lo, hi = min(0, mapped.min()), max(G.shape[1], mapped.max() + 1)
        if lo < 0 or hi > G.shape[1]:
            G2 = np.full((G.shape[0], hi - lo, 2), np.nan); V2 = np.zeros((G.shape[0], hi - lo), bool)
            G2[:, -lo:-lo + G.shape[1]], V2[:, -lo:-lo + G.shape[1]] = G, V
            G, V, mapped = G2, V2, mapped - lo
        added = 0
        for i in np.nonzero(v.any(1))[0]:
            oc = np.nonzero(v[i])[0]
            bc = mapped[oc]
            new = ~V[i, bc]
            G[i, bc[new]] = g[i, oc[new]]; V[i, bc[new]] = True
            added += int(new.sum())
        print(f"  -> merged (direction {s:+d}, {len(centres)} offset windows): +{added} points; base now {V.sum()}", flush=True)

    used = np.nonzero(V.any(0))[0]
    G, V = G[:, used[0]:used[-1] + 1], V[:, used[0]:used[-1] + 1]
    G[~V] = np.nan
    out = save_trace(a.out, G, V, list(Z), region, "psn merge_traces.py", {"merged_from": [a.base, *a.others]})
    px_mm = region["um_per_voxel"] / 1000
    print(f"saved {out}: {V.sum()} points = {V.sum() * px_mm ** 2:.0f} mm^2, grid {G.shape[0]} x {G.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
