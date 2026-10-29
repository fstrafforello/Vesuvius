"""Fill small holes in a grown 3D surface grid before rendering.

Holes up to MAX_HOLE cells (fully surrounded by accepted cells) get positions interpolated from
their neighbours (iterative neighbour averaging = discrete Laplace fill). Larger holes stay holes.
Writes grow_<name>_f (same format as grow3d.py output).

Usage:  uv run python src/surface/fill_grid_holes.py --name g3
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
from trace import D  # noqa: E402

MAX_HOLE = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    src, dst = D / f"grow_{a.name}", D / f"grow_{a.name}_f"
    xyz = np.stack([tifffile.imread(src / f"{c}.tif").astype(np.float64) for c in "xyz"], -1)
    ok = (xyz[..., 0] > 0)
    holes = ndimage.binary_fill_holes(ok) & ~ok
    lab, n = ndimage.label(holes)
    sizes = ndimage.sum(holes, lab, range(1, n + 1)) if n else np.array([])
    small = np.isin(lab, np.nonzero(sizes <= MAX_HOLE)[0] + 1)
    P = np.where(ok[..., None], xyz, 0.0)
    for _ in range(200):                                   # Laplace fill of the small holes
        acc = ndimage.uniform_filter(P, size=(3, 3, 1)) * 9
        cnt = ndimage.uniform_filter((ok | small).astype(float), 3) * 9
        filled = (ok | small)
        avg = acc / np.maximum(ndimage.uniform_filter(filled.astype(float), 3)[..., None] * 9, 1e-9)
        P = np.where(small[..., None], avg, P)
    out_ok = ok | small
    dst.mkdir(exist_ok=True)
    for i, c in enumerate("xyz"):
        tifffile.imwrite(dst / f"{c}.tif", np.where(out_ok, P[..., i], -1).astype(np.float32))
    shutil.copy(src / "meta.json", dst / "meta.json")
    print(f"filled {int(small.sum())} cells in {int((sizes <= MAX_HOLE).sum())} small holes "
          f"(left {int((sizes > MAX_HOLE).sum())} larger holes); valid {ok.mean():.1%} -> {out_ok.mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
