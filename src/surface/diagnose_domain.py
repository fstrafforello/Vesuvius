"""Why does the Scroll 4 ink model misfire on PHerc0846A? And is there a model-free ink signal?

Part A - compare the material: Scroll 4 reference area (w031, official render, known labels:
ink / no-ink pixels) against our verified PHerc0846A surface (grow_g3_f, both normal directions):
  - mean brightness profile through the depth of the sheet (109 layers, 2.4 um apart)
  - brightness distribution on the layer at the surface
  - texture: radially averaged power spectrum of the surface layer (fibre scale)

Part B - model-free ink features: simple per-pixel features from the depth stack (e.g. mean of
the layers just in front of the surface minus layers behind it). Each is scored on Scroll 4 by how
well it separates labelled ink from no-ink (AUC). Features that work there are mapped on
PHerc0846A, smoothed to stroke scale, to look for stroke-shaped structure.

Usage:  uv run python src/surface/diagnose_domain.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "surface")]
from render import dense_surface, grid_normals, load_tifxyz, render  # noqa: E402

D = ROOT / "data" / "scout" / "PHerc0846A"
REF = ROOT / "data" / "PHerc1667" / "render_check" / "w031"
L1 = ROOT / "data" / "PHerc1667" / "ink_l1" / "w031"
C = 54
OUT = D / "diagnose"


def auc(score, lab):
    s = score.ravel(); y = lab.ravel()
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def spectrum(img, mask):
    """Radially averaged power spectrum of the (mask-filled, mean-removed) image, normalised."""
    x = np.where(mask, img, img[mask].mean()).astype(np.float64)
    x -= x.mean()
    f = np.abs(np.fft.fftshift(np.fft.fft2(x * np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))))) ** 2
    h, w = f.shape
    yy, xx = np.indices(f.shape)
    r = np.hypot((yy - h / 2) / h, (xx - w / 2) / w)        # cycles per pixel
    bins = np.linspace(0.005, 0.5, 60)
    idx = np.digitize(r, bins)
    p = np.array([f[idx == i].mean() if (idx == i).any() else np.nan for i in range(1, len(bins))])
    return 0.5 * (bins[1:] + bins[:-1]), p / np.nansum(p)


def features(stack):
    """Model-free per-pixel features from a (109, H, W) depth stack (float)."""
    front = stack[C - 12:C - 2].mean(0)
    surf = stack[C - 2:C + 3].mean(0)
    back = stack[C + 3:C + 13].mean(0)
    return {"surface layer": surf,
            "front minus back": front - back,
            "surface minus back": surf - back,
            "surface minus front": surf - front,
            "depth std (+-12)": stack[C - 12:C + 13].std(0),
            "local texture (std 9px)": ndimage.generic_filter(surf, np.std, size=9) if surf.size < 3e6 else
            np.sqrt(np.maximum(ndimage.uniform_filter(surf ** 2, 9) - ndimage.uniform_filter(surf, 9) ** 2, 0))}


def main():
    OUT.mkdir(exist_ok=True)
    # --- Scroll 4 reference (official render, level 0, 1024 x 1024) + labels (level 1 -> upsample x2)
    ref = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(REF / "official_l0.zarr")}}, read=True).result().read().result().astype(np.float32)
    area = json.loads((REF / "area.json").read_text())
    y1, x1 = area["l1_yx"]; n = area["size"] // 2
    lab = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(L1 / "labels/inklabels.zarr/1")}}, read=True).result()[y1:y1 + n, x1:x1 + n].read().result() > 127
    sup = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(L1 / "labels/supervision.zarr/1")}}, read=True).result()[y1:y1 + n, x1:x1 + n].read().result() > 127
    lab, sup = np.kron(lab, np.ones((2, 2), bool)), np.kron(sup, np.ones((2, 2), bool))
    ref_ok = (ref[C] > 0)
    ink, noink = lab & sup & ref_ok, ~lab & sup & ref_ok
    print(f"Scroll 4 reference: ink px {ink.sum()}, no-ink px {noink.sum()}", flush=True)

    # --- PHerc0846A verified grown surface, both normal directions
    xyz, valid, scale = load_tifxyz(D / "grow_g3_f")
    meta = json.loads((D / "ct_L0_l1_filled.json").read_text())
    ct = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(D / "ct_L0_l1_filled.zarr")}}, read=True).result().read().result()
    gr, gc = np.nonzero(valid)
    rows = np.arange(gr.min() / scale, (gr.max() - 1) / scale)
    cols = np.arange(gc.min() / scale, (gc.max() - 1) / scale)
    pos, nrm, ok = dense_surface(xyz, grid_normals(xyz, valid), valid, scale, rows, cols)
    stacks = {}
    for sign in (1.0, -1.0):
        stacks[sign] = render(pos, nrm, ok, ct, meta["origin_zyx"], range(-C, C + 1), sign).astype(np.float32)
    print(f"PHerc0846A surface {ok.shape}, valid {ok.mean():.1%}", flush=True)

    # --- Part A: depth profiles, histograms, spectra
    fig, ax = plt.subplots(1, 3, figsize=(21, 5.5))
    depth = np.arange(-C, C + 1)
    ax[0].plot(depth, ref[:, ink].mean(1), label="Scroll 4 ink")
    ax[0].plot(depth, ref[:, noink].mean(1), label="Scroll 4 no ink")
    for sign, st in stacks.items():
        ax[0].plot(depth, st[:, ok].mean(1), "--", label=f"PHerc0846A normal {sign:+.0f}")
    ax[0].set_title("mean brightness through the sheet"); ax[0].set_xlabel("layer offset from surface (2.4 um)"); ax[0].legend()
    bins = np.arange(0, 256, 4)
    ax[1].hist(ref[C][ink], bins, density=True, histtype="step", label="Scroll 4 ink")
    ax[1].hist(ref[C][noink], bins, density=True, histtype="step", label="Scroll 4 no ink")
    ax[1].hist(stacks[1.0][C][ok], bins, density=True, histtype="step", ls="--", label="PHerc0846A")
    ax[1].set_title("brightness on the surface layer"); ax[1].legend()
    f, p = spectrum(ref[C], ref_ok)
    ax[2].loglog(f, p, label="Scroll 4")
    sub = stacks[1.0][C][:1024, :1024]; m = ok[:1024, :1024]
    f2, p2 = spectrum(sub, m)
    ax[2].loglog(f2, p2, "--", label="PHerc0846A")
    ax[2].set_title("texture: power spectrum of the surface layer"); ax[2].set_xlabel("spatial frequency (cycles / 2.4 um px)"); ax[2].legend()
    plt.tight_layout(); plt.savefig(OUT / "material_comparison.png", dpi=65); plt.close()

    stats = {
        "scroll4_ink_surface_mean": float(ref[C][ink].mean()), "scroll4_noink_surface_mean": float(ref[C][noink].mean()),
        "scroll4_surface_p10_p90": np.percentile(ref[C][ref_ok], [10, 90]).tolist(),
        "pherc0846a_surface_p10_p90": np.percentile(stacks[1.0][C][ok], [10, 90]).tolist(),
    }

    # --- Part B: model-free features, scored on Scroll 4, mapped on PHerc0846A
    fr = features(ref)
    scores = {}
    for k, v in fr.items():
        a = auc(v[sup & ref_ok], lab[sup & ref_ok])
        scores[k] = max(a, 1 - a)                               # either sign may indicate ink
        print(f"  feature {k:24s}: Scroll 4 ink/no-ink AUC {scores[k]:.3f}" + (" (inverted)" if a < 0.5 else ""), flush=True)
    stats["feature_auc_scroll4"] = scores
    best = sorted(scores, key=scores.get, reverse=True)[:3]
    fig, ax = plt.subplots(len(best), 3, figsize=(21, 5 * len(best)))
    for r, k in enumerate(best):
        sm = lambda img: ndimage.gaussian_filter(img, 6)       # stroke scale smoothing
        ax[r, 0].imshow(np.where(ref_ok, sm(fr[k]), np.nan), cmap="magma"); ax[r, 0].set_title(f"Scroll 4: {k} (AUC {scores[k]:.2f})")
        ax[r, 0].contour(lab & sup, levels=[0.5], colors="cyan", linewidths=0.6)
        for c, sign in enumerate((1.0, -1.0)):
            fp = features(stacks[sign])[k]
            ax[r, 1 + c].imshow(np.where(ok, sm(fp), np.nan), cmap="magma")
            ax[r, 1 + c].set_title(f"PHerc0846A normal {sign:+.0f}: {k}")
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "model_free_features.png", dpi=55); plt.close()
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("saved figures in", OUT)


if __name__ == "__main__":
    main()
