"""Clean a grown 3D surface: keep the part connected to the seed through strictly consistent steps.

Grid neighbours count as connected only if they are within EDGE_ACROSS voxels of each other across
the sheet (along their averaged normal) and at a sane distance. The piece connected to the seed
survives; anything reached only through larger sideways steps (a slide onto a neighbouring sheet)
is cut off. Writes grow_<name>_c in the grow3d format.

Usage:  uv run python src/surface/grow_clean.py --name top_s361 [--edge 1.0]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).parent))
from trace import D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--edge", type=float, default=1.0)
    a = ap.parse_args()
    src, dst = D / f"grow_{a.name}", D / f"grow_{a.name}_c"
    t = np.load(src / "grid_block_coords.npz")
    P, V, N = t["P"], t["valid"], t["N"]
    meta = json.loads((src / "meta.json").read_text())
    h = meta["grid_spacing_level1"]
    idx = -np.ones(V.shape, int); idx[V] = np.arange(V.sum())
    ei, ej = [], []
    for axis in (0, 1):
        a1 = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        a0 = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        both = V[a1] & V[a0]
        dp = P[a1] - P[a0]
        n = N[a1] + N[a0]
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        across = np.abs((dp * n).sum(-1))
        dist = np.linalg.norm(dp, axis=-1)
        ok = both & (across <= a.edge) & (dist >= 0.6 * h) & (dist <= 1.5 * h)
        ei.append(idx[a0][ok]); ej.append(idx[a1][ok])
    ei, ej = np.concatenate(ei), np.concatenate(ej)
    n = int(V.sum())
    _, lab = connected_components(coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(n, n)), directed=False)
    labels = -np.ones(V.shape, int); labels[V] = lab
    seed = np.array(meta["seed_block_zyx"], float)
    d = np.linalg.norm(P - seed, axis=-1); d[~V] = np.inf
    sr, sc = np.unravel_index(np.argmin(d), d.shape)
    keep = labels == labels[sr, sc]
    print(f"{a.name}: kept {keep.sum()} of {V.sum()} points ({keep.sum() / V.sum():.0%}) = "
          f"{keep.sum() * (h * 4.806 / 1000) ** 2:.1f} mm^2", flush=True)
    dst.mkdir(exist_ok=True)
    for c in "xyz":
        arr = tifffile.imread(src / f"{c}.tif")
        tifffile.imwrite(dst / f"{c}.tif", np.where(keep, arr, -1).astype(np.float32))
    shutil.copy(src / "meta.json", dst / "meta.json")
    np.savez_compressed(dst / "grid_block_coords.npz", P=np.where(keep[..., None], P, np.nan), valid=keep, N=N)


if __name__ == "__main__":
    main()
