"""Densify a grown 3D surface: fill its holes from the edges inward, checking every point against the data.

Per pass, every empty grid cell with >= 3 accepted cells among its 8 neighbours gets a position
predicted by a local plane fit (5x5 neighbourhood) and is snapped at most R voxels across the sheet
onto the papyrus ridge. It is accepted when it lands on papyrus (>= THR) within MAX_SHIFT of the
prediction and keeps a sane spacing to its direct neighbours. Passes repeat until nothing changes.
Remaining holes of <= SMALL cells are then interpolated (as in fill_grid_holes.py).

Writes grow_<name>_d (same format as grow3d.py output), and reports the usable interior area
(cells at least MARGIN_MM from any hole or edge) before and after.

Usage:  uv run python src/surface/grid_densify.py --name s652
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from grow3d import Volume, snap  # noqa: E402
from trace import D, load  # noqa: E402

THR, MAX_SHIFT, SMALL, MARGIN_MM = 90.0, 1.75, 12, 0.25


def interior_mm2(valid, h_mm):
    m = int(round(MARGIN_MM / h_mm))
    return ndimage.binary_erosion(valid, iterations=m).sum() * h_mm ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--passes", type=int, default=40)
    a = ap.parse_args()
    src, dst = D / f"grow_{a.name}", D / f"grow_{a.name}_d"
    meta = json.loads((src / "meta.json").read_text())
    t = np.load(src / "grid_block_coords.npz")
    P, V, N = t["P"].astype(np.float64), t["valid"].copy(), t["N"].astype(np.float64)
    h = meta["grid_spacing_level1"]
    block, region, _ = load(1, meta.get("region_tag"))
    vol = Volume(block.read().result())
    h_mm = h * region["um_per_voxel"] / 1000
    before_area, before_int = V.sum() * h_mm ** 2, interior_mm2(V, h_mm)
    region_mask = ndimage.binary_fill_holes(ndimage.binary_closing(V, iterations=3))   # only fill inside the outline
    H, W = V.shape
    added_total = 0
    for it in range(a.passes):
        nb = ndimage.convolve(V.astype(np.int32), np.ones((3, 3), np.int32), mode="constant") - V
        cand = np.argwhere(~V & region_mask & (nb >= 3))
        added = 0
        for r, c in cand:
            r0, r1, c0, c1 = max(r - 2, 0), min(r + 3, H), max(c - 2, 0), min(c + 3, W)
            m = V[r0:r1, c0:c1]
            rr, cc = np.nonzero(m)
            if len(rr) < 4 or len(np.unique(rr)) < 2 or len(np.unique(cc)) < 2:
                continue
            X = np.stack([np.ones(len(rr)), rr + r0 - r, cc + c0 - c], 1).astype(float)
            coef, *_ = np.linalg.lstsq(X, P[r0:r1, c0:c1][m], rcond=None)
            pred = coef[0]
            n = N[r0:r1, c0:c1][m].mean(0)
            n /= max(np.linalg.norm(n), 1e-9)
            if not vol.inside(pred):
                continue
            p, val, shift = snap(vol, pred, n)
            if val < THR or shift > MAX_SHIFT:
                continue
            ok = True
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr2, cc2 = r + dr, c + dc
                if 0 <= rr2 < H and 0 <= cc2 < W and V[rr2, cc2]:
                    d = np.linalg.norm(p - P[rr2, cc2])
                    if not (0.5 * h <= d <= 1.6 * h):
                        ok = False; break
            if ok:
                P[r, c], N[r, c], V[r, c] = p, n, True
                added += 1
        added_total += added
        if added == 0:
            break
    # interpolate remaining small holes
    holes = ndimage.binary_fill_holes(V) & ~V
    lab, nh = ndimage.label(holes)
    sizes = ndimage.sum(holes, lab, range(1, nh + 1)) if nh else np.array([])
    small = np.isin(lab, np.nonzero(sizes <= SMALL)[0] + 1)
    Pf = np.where(V[..., None], P, 0.0)
    for _ in range(100):
        s = ndimage.uniform_filter(Pf, size=(3, 3, 1)) * 9
        w = ndimage.uniform_filter((V | small).astype(float), 3) * 9
        Pf = np.where(small[..., None], s / np.maximum(w[..., None], 1e-9), Pf)
    P = np.where(small[..., None], Pf, P)
    V = V | small
    after_area, after_int = V.sum() * h_mm ** 2, interior_mm2(V, h_mm)
    print(f"{a.name}: {it + 1} passes, +{added_total} data-checked points, +{int(small.sum())} interpolated | "
          f"area {before_area:.1f} -> {after_area:.1f} mm2 | usable interior {before_int:.1f} -> {after_int:.1f} mm2", flush=True)

    dst.mkdir(exist_ok=True)
    off = np.array([region["z0"], region["y0"], region["x0"]], float)
    L0 = (P + off) * 2 ** region["level"]
    for nm, ci in (("z", 0), ("y", 1), ("x", 2)):
        tifffile.imwrite(dst / f"{nm}.tif", np.where(V, L0[..., ci], -1).astype(np.float32))
    shutil.copy(src / "meta.json", dst / "meta.json")
    np.savez_compressed(dst / "grid_block_coords.npz", P=np.where(V[..., None], P, np.nan), valid=V, N=N)


if __name__ == "__main__":
    main()
