"""Cut the clean core out of a traced surface: the part that provably stays on ONE sheet.

Starting at the column through the seed point, keep its hop-free stretch containing the seed
slice. Then go outward column by column: in each column, split the column's points into hop-free
stretches (the same hop rule as section_check.py) and keep only the stretch that best matches the
already-kept neighbouring column (same slices, within MATCH voxels in 3D, at least MIN_SHARED
shared slices). Everything else is dropped.

Usage:  uv run python src/surface/clean_core.py --name l1_filled --seed-from l1 --out l1_core
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from trace import D, load, save_trace, trace_level, trace_tag  # noqa: E402

HOP = 1.5          # voxels per slice across the sheet at level 2 (scaled per level)
SLOPE = 0.3        # sideways drift a sheet can make per slice across a gap (measured 0.1-0.25)
MATCH = 2.5        # voxels (level 2, scaled): neighbouring columns agree if this close in the same slice
MIN_SHARED = 20    # slices two neighbouring stretches must share


def stretches(grid, valid, c, k):
    """Hop-free runs of rows in column c (lists of row indices)."""
    rows = np.nonzero(valid[:, c])[0]
    if len(rows) == 0:
        return []
    p = grid[rows, c]
    jump = np.linalg.norm(np.diff(p, axis=0), axis=1)
    gap = np.diff(rows)
    limit = np.where(gap == 1, HOP * k, SLOPE * gap + 4.0 * k)
    cuts = np.nonzero(jump > limit)[0]
    return [r for r in np.split(rows, cuts + 1) if len(r)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed-from", required=True, help="trace whose meta.json holds the seed")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = np.load(D / f"trace_{a.name}" / "trace_block_coords.npz")
    grid, valid, zs = t["grid"].astype(np.float64), t["valid"], t["z"]
    level = trace_level(D / f"trace_{a.name}")
    _, region, _ = load(level, trace_tag(D / f"trace_{a.name}"))
    k = 2 ** (2 - level)
    seed_z, sy, sx = json.loads((D / f"trace_{a.seed_from}" / "meta.json").read_text())["seed_block_zyx"]
    srow = int(np.argmin(np.abs(zs - seed_z)))
    cols = np.nonzero(valid[srow])[0]
    scol = int(cols[np.argmin(np.linalg.norm(grid[srow, cols] - [sy, sx], axis=1))])

    keep = np.zeros_like(valid)
    st = [s for s in stretches(grid, valid, scol, k) if srow in set(s.tolist())]
    if not st:
        raise SystemExit("seed column has no stretch through the seed slice")
    keep[st[0], scol] = True
    for direction in (1, -1):
        prev = scol
        c = scol + direction
        misses = 0
        while 0 <= c < grid.shape[1] and misses < 10:
            best, best_n = None, 0
            prow = np.nonzero(keep[:, prev])[0]
            for s in stretches(grid, valid, c, k):
                shared = np.intersect1d(s, prow)
                if len(shared) == 0:
                    continue
                close = np.linalg.norm(grid[shared, c] - grid[shared, prev], axis=1) <= MATCH * k
                if close.sum() >= MIN_SHARED and close.mean() >= 0.8 and close.sum() > best_n:
                    best, best_n = s, int(close.sum())
            if best is not None:
                keep[best, c] = True
                prev, misses = c, 0
            else:
                misses += 1                     # allow short gaps in columns, matched against the last kept column
            c += direction
    used = np.nonzero(keep.any(0))[0]
    g, v = grid[:, used[0]:used[-1] + 1], keep[:, used[0]:used[-1] + 1]
    g = np.where(v[..., None], g, np.nan)
    mm = region["um_per_voxel"] / 1000
    print(f"clean core: {v.sum()} points = {v.sum() * mm * mm:.1f} mm^2; {len(used)} columns "
          f"({len(used) * mm:.1f} mm along the sheet), rows {np.nonzero(v.any(1))[0].min()}..{np.nonzero(v.any(1))[0].max()} "
          f"({v.any(1).sum() * mm:.1f} mm of height)", flush=True)
    out = save_trace(a.out, g, v, list(zs), region, "psn clean_core.py",
                     {"seed_block_zyx": [seed_z, sy, sx], "from": a.name})
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
