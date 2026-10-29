"""Depth check: how does the ink model's response change as the rendered depth window slides
through the sheet?

Real ink sits at the writing surface, so its response should PEAK at one depth and fall off a
few voxels away. A response driven by papyrus texture or density stays roughly flat over depth.
As a reference we run the same sweep on a Scroll 4 area with known ink labels (w031 render-check
area), comparing labelled-ink pixels with labelled no-ink pixels.

Usage:  uv run python src/surface/depth_profile.py --name test_filled
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "surface"), str(ROOT / "src" / "ink")]
from render import dense_surface, grid_normals, load_tifxyz, render  # noqa: E402

D = ROOT / "data" / "scout" / "PHerc0846A"
REF = ROOT / "data" / "PHerc1667" / "render_check" / "w031"
L1 = ROOT / "data" / "PHerc1667" / "ink_l1" / "w031"
CENTRE = 54
SHIFTS = list(range(-20, 21, 2))


def load_models():
    import torch
    from models import build
    from train_baseline import N_LAYERS
    out = []
    from registry import SCROLL4_PAIR, run_dir
    for r in SCROLL4_PAIR:
        ck = torch.load(run_dir(r) / "model.pt", map_location="cpu", weights_only=False)
        m = build(ck["args"]["model"], N_LAYERS, ck["args"].get("encoder", "resnet34")).cuda().eval()
        m.load_state_dict(ck["model"]); out.append(m)
    return out


def predictor(models):
    import torch
    from train_baseline import JITTER, N_LAYERS, norm

    @torch.no_grad()
    def predict(vol):
        v = torch.from_numpy(vol).cuda()
        P, S = 384, 128
        H, W = v.shape[-2:]
        r = torch.minimum(torch.arange(P, device="cuda") + 1, torch.arange(P, 0, -1, device="cuda")).float()
        w = torch.outer(r, r)
        out = torch.zeros(H, W, device="cuda"); ws = torch.zeros_like(out)
        ys = sorted(set(list(range(0, max(H - P, 0) + 1, S)) + [max(H - P, 0)]))
        xs = sorted(set(list(range(0, max(W - P, 0) + 1, S)) + [max(W - P, 0)]))
        for y in ys:
            for x in xs:
                xb = norm(v[JITTER:JITTER + N_LAYERS, y:y + P, x:x + P])[None]
                with torch.autocast("cuda", torch.bfloat16):
                    p = sum(torch.sigmoid(m(xb).float())[0, 0] for m in models) / len(models)
                out[y:y + P, x:x + P] += p * w; ws[y:y + P, x:x + P] += w
        return (out / ws).cpu().numpy()
    return predict


def to_l1(full, shift):
    sl = full[CENTRE + shift - 18:CENTRE + shift + 18].astype(np.float32)
    h, w = sl.shape[1] // 2 * 2, sl.shape[2] // 2 * 2
    return np.rint(sl[:, :h, :w].reshape(36, h // 2, 2, w // 2, 2).mean((2, 4))).astype(np.uint8)


def sweep(full, predict, masks):
    """Mean probability inside each mask, for every depth shift."""
    prof = {k: [] for k in masks}
    for s in SHIFTS:
        p = predict(to_l1(full, s))
        for k, m in masks.items():
            prof[k].append(float(p[m].mean()))
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="test_filled")
    a = ap.parse_args()
    predict = predictor(load_models())

    # reference: Scroll 4 w031 area with known labels (official render, already downloaded)
    off = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(REF / "official_l0.zarr")}}, read=True).result().read().result()
    area = json.loads((REF / "area.json").read_text())
    y1, x1 = area["l1_yx"]; n = area["size"] // 2
    lab = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(L1 / "labels/inklabels.zarr/1")}}, read=True).result()[y1:y1 + n, x1:x1 + n].read().result() > 127
    sup = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(L1 / "labels/supervision.zarr/1")}}, read=True).result()[y1:y1 + n, x1:x1 + n].read().result() > 127
    ref = sweep(off, predict, {"Scroll 4: labelled ink": lab & sup, "Scroll 4: labelled no ink": ~lab & sup})
    print("reference done", flush=True)

    # ours: PHerc0846A traced patch, holes filled with papyrus grey, both normal directions
    xyz, valid, scale = load_tifxyz(D / f"trace_{a.name}")
    meta = json.loads((D / f"ct_L0_{a.name}.json").read_text())
    ct = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(D / f"ct_L0_{a.name}.zarr")}}, read=True).result().read().result()
    gr, gc = np.nonzero(valid)
    rows = np.arange(gr.min() / scale, gr.max() / scale)[:-1]
    cols = np.arange(gc.min() / scale, gc.max() / scale)[:-1]
    rows, cols = rows[: len(rows) // 2 * 2], cols[: len(cols) // 2 * 2]
    pos, nrm, ok = dense_surface(xyz, grid_normals(xyz, valid), valid, scale, rows, cols)
    ok1 = ok.reshape(len(rows) // 2, 2, len(cols) // 2, 2).all((1, 3))
    ours = {}
    for sign in (1.0, -1.0):
        full = render(pos, nrm, ok, ct, meta["origin_zyx"], range(-CENTRE, CENTRE + 1), sign)
        for li in range(full.shape[0]):
            full[li][~ok] = np.uint8(np.median(full[li][ok]))
        # where does the model respond at all? take the top 10% of the patch at zero shift as "responding"
        p0 = predict(to_l1(full, 0))
        hot = ok1 & (p0 >= np.percentile(p0[ok1], 90))
        prof = sweep(full, predict, {"all": ok1, "hot": hot})
        ours[f"PHerc0846A normal {sign:+.0f}: whole patch"] = prof["all"]
        ours[f"PHerc0846A normal {sign:+.0f}: top-10% area"] = prof["hot"]
        print(f"normal {sign:+.0f} done", flush=True)

    def peakiness(c):
        c = np.array(c)
        return float(c.max() - np.median(c))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for k, v in ref.items():
        ax[0].plot(SHIFTS, v, marker="o", label=f"{k} (peak-median {peakiness(v):.2f})")
    ax[0].set_title("Reference: Scroll 4, known ink"); ax[0].set_xlabel("depth shift (voxels, 2.4 um)"); ax[0].set_ylabel("mean ink probability"); ax[0].legend()
    for k, v in ours.items():
        ax[1].plot(SHIFTS, v, marker="o", label=f"{k} (peak-median {peakiness(v):.2f})")
    ax[1].set_title("PHerc0846A, our traced patch"); ax[1].set_xlabel("depth shift (voxels, 2.4 um)"); ax[1].legend(fontsize=8)
    for a_ in ax:
        a_.grid(alpha=0.3); a_.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(D / f"depth_profile_{a.name}.png", dpi=70); plt.close()
    (D / f"depth_profile_{a.name}.json").write_text(json.dumps({"shifts": SHIFTS, "reference": ref, "ours": ours}, indent=2))
    for k, v in {**ref, **ours}.items():
        print(f"{k:45s} max {max(v):.3f} @ {SHIFTS[int(np.argmax(v))]:+d}  min {min(v):.3f}  peak-median {peakiness(v):.3f}")
    print("saved", D / f"depth_profile_{a.name}.png")


if __name__ == "__main__":
    main()
