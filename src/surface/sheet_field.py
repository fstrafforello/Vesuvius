"""Step d1: papyrus map and sheet-normal field for the level-2 tracing block.

For every voxel we estimate
  papyrus  - smoothed "is this papyrus" probability (CT density above the air/gap level)
  normal   - the direction perpendicular to the local sheet, from the structure tensor:
             the eigenvector of the largest eigenvalue of the smoothed gradient outer product
             (intensity changes fastest across a sheet, slowest along it)
  planarity - how sheet-like the neighbourhood is (1 = clean single orientation, 0 = mess)

Computed in overlapping 256^3 pieces on the CPU, stored at half the block resolution (19 um) for
the normal and planarity (they vary slowly) and full block resolution for papyrus.

Usage:  uv run python src/surface/sheet_field.py
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorstore as ts
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "scout" / "PHerc0846A"
PIECE, HALO = 256, 24
SIG_GRAD, SIG_TENSOR = 1.0, 4.0


def open_store(name, shape, dtype, chunks, create=False):
    spec = {"driver": "zarr", "kvstore": {"driver": "file", "path": str(D / name)}}
    if create:
        spec.update(metadata={"shape": list(shape), "dtype": dtype, "chunks": list(chunks),
                              "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}},
                    create=True, delete_existing=True)
        return ts.open(spec).result()
    return ts.open(spec, read=True).result()


def air_level(vol):
    """Grey value separating gaps from papyrus: valley of a 2-peak histogram (Otsu)."""
    s = vol[::4, ::8, ::8].read().result().ravel()
    s = s[s > 0]
    hist = np.bincount(s, minlength=256).astype(np.float64)
    p = hist / hist.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return int(np.argmax(between))


def piece(vol, z, y, x, thr):
    sh = vol.shape
    lo = [max(0, c - HALO) for c in (z, y, x)]
    hi = [min(n, c + PIECE + HALO) for c, n in zip((z, y, x), sh)]
    a = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].read().result().astype(np.float32)
    pap = ndimage.gaussian_filter(1 / (1 + np.exp(-(a - thr) / 6.0)), 1.0)   # soft papyrus mask
    g = [ndimage.gaussian_filter(a, SIG_GRAD, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    T = {}
    for i in range(3):
        for j in range(i, 3):
            T[i, j] = ndimage.gaussian_filter(g[i] * g[j], SIG_TENSOR)[::2, ::2, ::2]
    M = np.empty(T[0, 0].shape + (3, 3), np.float32)
    for (i, j), t in T.items():
        M[..., i, j] = t; M[..., j, i] = t
    w, v = np.linalg.eigh(M)                                  # ascending eigenvalues
    normal = v[..., :, 2]                                      # (z, y, x) components
    planar = (w[..., 2] - w[..., 1]) / np.maximum(w[..., 2], 1e-6)
    cz, cy, cx = (z - lo[0], y - lo[1], x - lo[2])
    ez, ey, ex = (min(PIECE, sh[0] - z), min(PIECE, sh[1] - y), min(PIECE, sh[2] - x))
    pap = pap[cz:cz + ez, cy:cy + ey, cx:cx + ex]
    s = lambda arr: arr[cz // 2:(cz + ez) // 2, cy // 2:(cy + ey) // 2, cx // 2:(cx + ex) // 2]
    return (z, y, x), pap, s(normal), s(planar)


def main():
    vol = open_store("trace_L2.zarr", None, None, None)
    sh = vol.shape
    thr = air_level(vol)
    print(f"block {sh}; papyrus/gap threshold {thr}", flush=True)
    pap_s = open_store("papyrus_L2.zarr", sh, "|u1", (128, 128, 128), create=True)
    half = [n // 2 for n in sh]
    nrm_s = open_store("normal_L3.zarr", half + [3], "|i1", (64, 64, 64, 3), create=True)
    pla_s = open_store("planarity_L3.zarr", half, "|u1", (64, 64, 64), create=True)
    jobs = [(z, y, x) for z in range(0, sh[0], PIECE) for y in range(0, sh[1], PIECE) for x in range(0, sh[2], PIECE)]
    t0 = time.time()

    def run(job):
        (z, y, x), pap, nrm, pla = piece(vol, *job, thr)
        # eigenvectors have arbitrary sign; store a fixed convention (x-component >= 0) and let
        # the tracer re-orient locally, since sheets curve around the scroll
        nrm = np.where(nrm[..., 2:3] < 0, -nrm, nrm)
        pap_s[z:z + pap.shape[0], y:y + pap.shape[1], x:x + pap.shape[2]].write((pap * 255).astype(np.uint8)).result()
        z2, y2, x2 = z // 2, y // 2, x // 2
        nrm_s[z2:z2 + nrm.shape[0], y2:y2 + nrm.shape[1], x2:x2 + nrm.shape[2]].write(np.rint(nrm * 127).astype(np.int8)).result()
        pla_s[z2:z2 + pla.shape[0], y2:y2 + pla.shape[1], x2:x2 + pla.shape[2]].write((np.clip(pla, 0, 1) * 255).astype(np.uint8)).result()

    with ThreadPoolExecutor(6) as pool:
        for i, _ in enumerate(pool.map(run, jobs), 1):
            if i % 16 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} pieces, {el / 60:.1f} min, ~{el / i * (len(jobs) - i) / 60:.1f} min left", flush=True)
    (D / "sheet_field.json").write_text(json.dumps({"threshold": thr, "sig_grad": SIG_GRAD, "sig_tensor": SIG_TENSOR,
                                                    "normal_planarity_level": 3, "papyrus_level": 2}, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
