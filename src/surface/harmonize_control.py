"""Control for the harmonisation step: does it keep real ink detectable?

Take the Scroll 4 reference area (known labels), make it look like PHerc0846A (map each layer's
brightness onto the PHerc0846A distribution, add PHerc0846A-like fine-scale noise), then harmonise it
back exactly as first_ink.py does for PHerc0846A, and score the models against the labels.

  as-is         : the official Scroll 4 render (upper bound)
  disguised     : made to look like PHerc0846A, no harmonisation (should break, like PHerc0846A did)
  harmonised    : disguised, then harmonised back (should recover if harmonisation is sound)

Usage:  uv run python src/surface/harmonize_control.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorstore as ts
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "surface"), str(ROOT / "src" / "ink")]
import first_ink as FI  # noqa: E402
from depth_profile import CENTRE, to_l1  # noqa: E402
from models import build  # noqa: E402
from render import dense_surface, grid_normals, load_tifxyz, render  # noqa: E402
from train_baseline import JITTER, N_LAYERS, metrics, norm  # noqa: E402

D = ROOT / "data" / "scout" / "PHerc0846A"
REF = ROOT / "data" / "PHerc1667" / "render_check" / "w031"
L1 = ROOT / "data" / "PHerc1667" / "ink_l1" / "w031"
from registry import SCROLL4_PAIR as RUNS, run_dir  # noqa: E402


def models():
    out = []
    for r in RUNS:
        ck = torch.load(run_dir(r) / "model.pt", map_location="cpu", weights_only=False)
        m = build(ck["args"]["model"], N_LAYERS, ck["args"].get("encoder", "resnet34")).cuda().eval()
        m.load_state_dict(ck["model"]); out.append((m, ck["args"].get("norm", "fixed")))
    return out


@torch.no_grad()
def predict(ms, vol):
    v = torch.from_numpy(vol).cuda()
    P, S = 384, 128
    H, W = v.shape[-2:]
    r = torch.minimum(torch.arange(P, device="cuda") + 1, torch.arange(P, 0, -1, device="cuda")).float()
    w = torch.outer(r, r)
    out = torch.zeros(H, W, device="cuda"); ws = torch.zeros_like(out)
    for y in sorted(set(list(range(0, H - P + 1, S)) + [H - P])):
        for x in sorted(set(list(range(0, W - P + 1, S)) + [W - P])):
            win = v[JITTER:JITTER + N_LAYERS, y:y + P, x:x + P]
            with torch.autocast("cuda", torch.bfloat16):
                p = sum(torch.sigmoid(m(norm(win, mode)[None]).float())[0, 0] for m, mode in ms) / len(ms)
            out[y:y + P, x:x + P] += p * w; ws[y:y + P, x:x + P] += w
    return (out / ws).cpu().numpy()


def main():
    ref = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(REF / "official_l0.zarr")}}, read=True).result().read().result()
    area = json.loads((REF / "area.json").read_text())
    y1, x1 = area["l1_yx"]; n = area["size"] // 2
    lab = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(L1 / "labels/inklabels.zarr/1")}}, read=True).result()[y1:y1 + n, x1:x1 + n].read().result() > 127
    sup = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(L1 / "labels/supervision.zarr/1")}}, read=True).result()[y1:y1 + n, x1:x1 + n].read().result() > 127
    ok = ref[CENTRE] > 0

    # PHerc0846A brightness per layer + its excess fine-scale noise, from our verified surface
    xyz, valid, scale = load_tifxyz(D / "grow_g3_f")
    meta = json.loads((D / "ct_L0_l1_filled.json").read_text())
    ct = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(D / "ct_L0_l1_filled.zarr")}}, read=True).result().read().result()
    gr, gc = np.nonzero(valid)
    rows = np.arange(gr.min() / scale, (gr.max() - 1) / scale); cols = np.arange(gc.min() / scale, (gc.max() - 1) / scale)
    pos, nrm, pok = dense_surface(xyz, grid_normals(xyz, valid), valid, scale, rows, cols)
    ph = render(pos, nrm, pok, ct, meta["origin_zyx"], range(-CENTRE, CENTRE + 1), 1.0)
    noise_sd = np.std((ph[CENTRE].astype(np.float32) - ndimage.gaussian_filter(ph[CENTRE].astype(np.float32), 1.0))[pok])
    ref_noise = np.std((ref[CENTRE].astype(np.float32) - ndimage.gaussian_filter(ref[CENTRE].astype(np.float32), 1.0))[ok])
    extra = np.sqrt(max(noise_sd ** 2 - ref_noise ** 2, 0))
    print(f"fine-scale noise: PHerc0846A {noise_sd:.1f}, Scroll 4 {ref_noise:.1f} -> adding {extra:.1f}", flush=True)

    qs = np.linspace(0, 100, 101)
    rng = np.random.default_rng(0)
    disguised = np.empty_like(ref)
    for li in range(ref.shape[0]):
        a = ref[li].astype(np.float32)
        src_q = np.percentile(a[ok], qs)
        dst_q = np.percentile(ph[li][pok], qs)
        b = np.interp(a, src_q, dst_q) + rng.normal(0, extra, a.shape)
        disguised[li] = np.clip(np.where(ok, b, 0), 0, 255).astype(np.uint8)
    harmonised = FI.harmonize(disguised, ok, 1.0)

    ms = models()
    res = {}
    for name, vol in (("as-is", ref), ("disguised as PHerc0846A", disguised), ("disguised + harmonised", harmonised)):
        p = predict(ms, to_l1(vol, 0))
        m = metrics(p, lab, sup)
        res[name] = {"auc": m["auc"], "f05": m["f05"], "mean_prob": float(p[sup].mean())}
        print(f"{name:26s} AUC {m['auc']:.3f}  F0.5 {m['f05']:.3f}  mean prob {p[sup].mean():.3f}", flush=True)
    (D / "diagnose" / "harmonize_control.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
