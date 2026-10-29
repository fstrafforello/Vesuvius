"""Convert a grown 3D surface (grow3d.py) into the slice-by-slice trace format, so the same checks
(section_check.py) and ink tools (first_ink.py) apply.

For every grid column (position along the sheet) the accepted points are ordered by height; each
integer slice between two neighbouring points of that column gets an interpolated position. Then
columns are refined from the grid spacing (4 voxels) to 1 voxel by interpolating between
neighbouring columns.

Usage:  uv run python src/surface/grow_to_trace.py --name g2     (writes trace_grow_g2)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from trace import D, load, save_trace  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    src = D / f"grow_{a.name}"
    t = np.load(src / "grid_block_coords.npz")
    P, V = t["P"], t["valid"]                     # rows = v (~height), cols = u (along sheet)
    meta = json.loads((src / "meta.json").read_text())
    h = int(meta["grid_spacing_level1"])
    _, region, _ = load(1, meta.get("region_tag"))
    z_all = P[..., 0][V]
    z0, z1 = int(np.floor(z_all.min())), int(np.ceil(z_all.max()))
    zs = np.arange(z0, z1 + 1)
    ncol = P.shape[1]
    G = np.full((len(zs), ncol, 2), np.nan)
    ok = np.zeros((len(zs), ncol), bool)
    for c in range(ncol):
        rows = np.nonzero(V[:, c])[0]
        if len(rows) < 2:
            continue
        for r1, r2 in zip(rows[:-1], rows[1:]):
            if r2 - r1 != 1:                      # only interpolate between grid neighbours
                continue
            za, zb = P[r1, c, 0], P[r2, c, 0]
            if abs(zb - za) < 1e-6:
                continue
            for z in range(int(np.ceil(min(za, zb))), int(np.floor(max(za, zb))) + 1):
                w = (z - za) / (zb - za)
                G[z - z0, c] = (1 - w) * P[r1, c, 1:] + w * P[r2, c, 1:]
                ok[z - z0, c] = True
    # refine columns: h voxels -> 1 voxel, interpolating only between neighbouring valid columns
    F = np.full((len(zs), (ncol - 1) * h + 1, 2), np.nan)
    Fok = np.zeros(F.shape[:2], bool)
    for c in range(ncol - 1):
        both = ok[:, c] & ok[:, c + 1]
        for j in range(h):
            w = j / h
            F[both, c * h + j] = (1 - w) * G[both, c] + w * G[both, c + 1]
            Fok[both, c * h + j] = True
    F[ok[:, -1], -1], Fok[ok[:, -1], -1] = G[ok[:, -1], -1], True
    out = save_trace(f"grow_{a.name}", F, Fok, list(zs), region, "psn grow_to_trace.py",
                     {"seed_block_zyx": meta["seed_block_zyx"], "from_grow": a.name})
    mm = region["um_per_voxel"] / 1000
    print(f"converted: {Fok.sum()} points = {Fok.sum() * mm * mm:.1f} mm^2 over slices {z0}..{z1}; saved {out}", flush=True)


if __name__ == "__main__":
    main()
