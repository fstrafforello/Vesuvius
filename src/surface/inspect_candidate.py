"""Close look at the strongest consistent response on a PHerc0846A surface.

Finds the region where the harmonised ink predictions are highest across several settings,
then shows it zoomed: surface layer, depth layers, prediction; and compares the long axis of the
response with the dominant papyrus fibre direction there. A response that runs along the fibres is
more likely texture; one that crosses them is more like a stroke.

Usage:  uv run python src/surface/inspect_candidate.py --tag grow_g3_f_filled_holes_harm_b1 --name grow_g3_f
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

D = Path(__file__).resolve().parents[2] / "data" / "scout" / "PHerc0846A"
KEYS = ["p_-1_-8", "p_-1_-4", "p_-1_0", "p_1_-4", "p_1_0"]    # settings where it appeared


def orientation(img):
    """Dominant orientation (degrees, 0 = along image x) of structures in img (structure tensor)."""
    gy, gx = ndimage.sobel(img, 0), ndimage.sobel(img, 1)
    jxx, jyy, jxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
    grad = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))   # strongest gradient direction
    coherence = np.hypot(jxx - jyy, 2 * jxy) / max(jxx + jyy, 1e-9)
    return (grad + 90) % 180, coherence                         # structures run perpendicular to it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    z = np.load(D / f"first_ink_{a.tag}.npz")
    P = np.mean([z[k] for k in KEYS if k in z], 0)                # level-1 resolution
    sm = ndimage.gaussian_filter(P, 8)
    cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
    h = 160
    y0, y1, x0, x1 = max(cy - h, 0), min(cy + h, P.shape[0]), max(cx - h, 0), min(cx + h, P.shape[1])
    crop = P[y0:y1, x0:x1]
    print(f"strongest consistent response at level-1 ({cy}, {cx}); mean prob there {sm[cy, cx]:.2f}; "
          f"elsewhere median {np.median(sm[sm > 0]):.3f}", flush=True)

    layers = np.load(D / f"first_ink_{a.name}_layers.npy").astype(np.float32)   # level 0, offsets -30..30 step 6
    L = layers[:, 2 * y0:2 * y1, 2 * x0:2 * x1]
    surf = L[len(L) // 2]

    # long axis of the response blob vs fibre direction
    blob = crop > np.percentile(crop, 95)
    lab, n = ndimage.label(blob)
    big = lab == (1 + np.argmax(ndimage.sum(blob, lab, range(1, n + 1)))) if n else blob
    yy, xx = np.nonzero(big)
    cov = np.cov(np.stack([xx, -yy]))
    ev, vec = np.linalg.eigh(cov)
    blob_angle = np.degrees(np.arctan2(vec[1, 1], vec[0, 1])) % 180
    elong = np.sqrt(ev[1] / max(ev[0], 1e-9))
    fib_angle, coh = orientation(ndimage.gaussian_filter(surf, 2))
    fib_angle = (-fib_angle) % 180                                 # image y down -> math y up
    diff = abs((blob_angle - fib_angle + 90) % 180 - 90)
    size_mm = (np.ptp(xx) * 4.8 / 1000, np.ptp(yy) * 4.8 / 1000)
    print(f"response blob: {size_mm[0]:.2f} x {size_mm[1]:.2f} mm, elongation {elong:.1f}, long axis {blob_angle:.0f} deg", flush=True)
    print(f"fibres there: {fib_angle:.0f} deg (coherence {coh:.2f}) -> angle between response and fibres {diff:.0f} deg", flush=True)

    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    ax[0, 0].imshow(surf, cmap="gray"); ax[0, 0].set_title("surface layer (2.4 um)")
    ax[0, 1].imshow(L.max(0), cmap="gray"); ax[0, 1].set_title("max over layers -30..+30")
    ax[0, 2].imshow(crop, cmap="magma", vmin=0, vmax=max(crop.max(), 0.3)); ax[0, 2].set_title("harmonised ink prob (avg of settings)")
    ax[0, 3].imshow(surf, cmap="gray"); ax[0, 3].contour(np.kron(big, np.ones((2, 2))), [0.5], colors="cyan")
    ax[0, 3].set_title(f"response outline on surface; {diff:.0f} deg to fibres")
    for i, a_ in enumerate(ax[1]):
        k = [0, 3, 6, 10][i]
        a_.imshow(L[k], cmap="gray"); a_.set_title(f"layer offset {-30 + 6 * k:+d}")
    for a_ in ax.flat:
        a_.axis("off")
    plt.tight_layout(); plt.savefig(D / "diagnose" / f"candidate_{a.tag}.png", dpi=60); plt.close()
    print("saved", D / "diagnose" / f"candidate_{a.tag}.png", flush=True)


if __name__ == "__main__":
    main()
