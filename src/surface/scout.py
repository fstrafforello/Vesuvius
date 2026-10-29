"""Scouting: download a low-resolution pyramid level of an eligible scroll and render overview views.

Into data/scout/<scroll>/L<level>/ (a local copy of that zarr level) and figures alongside:
  axial.png    - slices across the scroll axis at several heights (the spiral of the wraps)
  sections.png - two long sections through the centre (how the wraps run along the scroll)
  profile.png  - fraction of papyrus per height (where the scroll is complete vs. broken)

Usage:  uv run python src/surface/scout.py fetch  [--level 4]
        uv run python src/surface/scout.py figures [--level 4]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fetch_starter import mirror  # noqa: E402

SCANS = {
    "PHerc1203": "PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr",
    "PHerc0846A": "PHerc0846A/volumes/20260319102732-2.403um-0.2m-77keV-masked.zarr",
}
OUT = ROOT / "data" / "scout"


def local(scroll, level):
    return ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(OUT / scroll / f"L{level}")}},
                   read=True).result()


def fetch(level):
    for scroll, vol in SCANS.items():
        print(f"{scroll}: level {level}", flush=True)
        mirror(f"{vol}/{level}", OUT / scroll / f"L{level}")


def stretch(a):
    nz = a[a > 0]
    lo, hi = np.percentile(nz, [1, 99.5]) if nz.size else (0, 255)
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def figures(level):
    um = 2.403 * 2 ** level
    for scroll in SCANS:
        v = local(scroll, level).read().result()
        d = OUT / scroll
        nz, ny, nx = v.shape
        frac = (v > 0).reshape(nz, -1).mean(1)
        zs = [int(nz * f) for f in (0.15, 0.3, 0.5, 0.7, 0.85)]

        fig, ax = plt.subplots(1, len(zs), figsize=(5 * len(zs), 5.4))
        for a, z in zip(ax, zs):
            a.imshow(stretch(v[z]), cmap="gray")
            a.set_title(f"z={z} ({z * um / 1000:.1f} mm)"); a.axis("off")
        fig.suptitle(f"{scroll} - slices across the scroll axis, level {level} ({um:.0f} um/voxel)")
        plt.tight_layout(); plt.savefig(d / "axial.png", dpi=70); plt.close()

        fig, ax = plt.subplots(1, 2, figsize=(14, 9))
        ax[0].imshow(stretch(v[:, ny // 2, :]), cmap="gray", aspect="auto"); ax[0].set_title("section through centre (x-z)")
        ax[1].imshow(stretch(v[:, :, nx // 2]), cmap="gray", aspect="auto"); ax[1].set_title("section through centre (y-z)")
        for a in ax:
            a.axis("off")
        plt.tight_layout(); plt.savefig(d / "sections.png", dpi=70); plt.close()

        plt.figure(figsize=(8, 3))
        plt.plot(np.arange(nz) * um / 1000, frac)
        plt.xlabel("height along scroll (mm)"); plt.ylabel("non-empty fraction"); plt.title(scroll)
        plt.tight_layout(); plt.savefig(d / "profile.png", dpi=80); plt.close()
        print(f"{scroll}: shape {v.shape} = {nz * um / 1000:.1f} x {ny * um / 1000:.1f} x {nx * um / 1000:.1f} mm; figures in {d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "figures"])
    ap.add_argument("--level", type=int, default=4)
    a = ap.parse_args()
    {"fetch": fetch, "figures": figures}[a.step](a.level)
