"""Steps c2 + e: first ink look on our own traced surface of PHerc0846A.

  fetch - full-resolution (2.403 um) CT for the bounding box of the traced piece
  run   - render 109 layers along the surface normal (same renderer we checked on Scroll 4),
          downsample to level 1 like the training data, run the averaged ink model
          for both normal directions and several depth shifts, save prediction images

Usage:  uv run python src/surface/first_ink.py fetch --name test_filled
        uv run python src/surface/first_ink.py run   --name test_filled
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
from fetch_starter import remote  # noqa: E402
from render import dense_surface, grid_normals, load_tifxyz, render  # noqa: E402
from scout import SCANS  # noqa: E402

D = ROOT / "data" / "scout" / "PHerc0846A"
MARGIN = 70
CENTRE = 54
FILL_HOLES = "--fill-holes" in sys.argv
RUNS = None   # list of run folder names under runs/ to average (default: the two original models)
TAG = ""
HARMONIZE, BLUR = False, 0.0
REF_RENDER = ROOT / "data" / "PHerc1667" / "render_check" / "w031" / "official_l0.zarr"


def surface_dir(name):
    """A traced surface lives in trace_<name>; a grown 3D surface in grow_<name> (pass 'grow_<name>')."""
    return D / name if name.startswith("grow_") else D / f"trace_{name}"


def bbox(name):
    xyz, valid, scale = load_tifxyz(surface_dir(name))
    pts = xyz[valid]
    lo = np.floor(pts.min(0)).astype(int) - MARGIN
    hi = np.ceil(pts.max(0)).astype(int) + MARGIN
    return xyz, valid, scale, [int(lo[2]), int(lo[1]), int(lo[0])], [int(hi[2] - lo[2]), int(hi[1] - lo[1]), int(hi[0] - lo[0])]


def harmonize(full, ok, blur):
    """Make a render look like Scroll 4 data before the model sees it:
    1. blur in-plane (sigma `blur` px) to remove the excess fine-scale noise of this scan
    2. map each layer's brightness distribution onto the same layer of the Scroll 4 reference
       render (quantile matching), measured on valid pixels only."""
    from scipy import ndimage
    ref = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(REF_RENDER)}}, read=True).result().read().result()
    qs = np.linspace(0, 100, 101)
    out = np.empty_like(full)
    for li in range(full.shape[0]):
        layer = full[li].astype(np.float32)
        if blur > 0:
            layer = ndimage.gaussian_filter(layer, blur)
        src_q = np.percentile(layer[ok], qs)
        r = ref[li][ref[li] > 0]
        ref_q = np.percentile(r, qs) if r.size else src_q
        out[li] = np.clip(np.interp(layer, src_q, ref_q), 0, 255).astype(np.uint8)
    return out


def fetch(name):
    _, _, _, origin, shape = bbox(name)
    gb = np.prod(shape) / 1e9
    print(f"CT bbox zyx origin {origin} shape {shape}: {gb:.2f} GB raw", flush=True)
    if gb > 12:
        sys.exit("too big - restrict the piece first")
    path = D / f"ct_L0_{name}.zarr"
    src = remote(f"{SCANS['PHerc0846A']}/0")
    dst = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)},
                   "metadata": {"shape": shape, "dtype": "|u1", "chunks": [128, 128, 128],
                                "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}},
                   "create": True, "delete_existing": True}).result()
    for z in range(0, shape[0], 256):
        z1 = min(shape[0], z + 256)
        dst[z:z1].write(src[origin[0] + z:origin[0] + z1, origin[1]:origin[1] + shape[1],
                            origin[2]:origin[2] + shape[2]].read().result()).result()
        print(f"  z {z}..{z1} of {shape[0]}", flush=True)
    (D / f"ct_L0_{name}.json").write_text(json.dumps({"origin_zyx": origin, "shape": shape}))
    print("fetched", path, flush=True)


def run(name, ct_name=None):
    import torch
    from models import build
    from train_baseline import JITTER, N_LAYERS, norm

    ct_name = ct_name or name                       # reuse another trace's CT download if it covers this one
    xyz, valid, scale, _, _ = bbox(name)
    meta = json.loads((D / f"ct_L0_{ct_name}.json").read_text())
    store = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(D / f"ct_L0_{ct_name}.zarr")}}, read=True).result()
    # read only the box around this surface (+ margin for 54 layers each side): a whole band
    # download can be larger than memory
    pts = xyz[valid]
    o = np.array(meta["origin_zyx"])
    lo = np.maximum(np.floor(pts[:, ::-1].min(0)).astype(int) - 70 - o, 0)
    hi = np.minimum(np.ceil(pts[:, ::-1].max(0)).astype(int) + 70 - o, np.array(store.shape))
    ct = store[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].read().result()
    meta = {**meta, "origin_zyx": (o + lo).tolist()}
    print(f"CT box {ct.shape} ({ct.nbytes / 1e9:.1f} GB) read from {ct_name}", flush=True)

    gr, gc = np.nonzero(valid)
    r0, r1 = gr.min(), gr.max() + 1
    c0, c1 = gc.min(), gc.max() + 1
    rows = np.arange(r0 / scale, (r1 - 1) / scale)          # surface pixels at level-0 spacing
    cols = np.arange(c0 / scale, (c1 - 1) / scale)
    rows, cols = rows[: len(rows) // 2 * 2], cols[: len(cols) // 2 * 2]
    print(f"surface {len(rows)} x {len(cols)} px at 2.4 um ({len(rows) * 2.4 / 1000:.1f} x {len(cols) * 2.4 / 1000:.1f} mm)", flush=True)
    pos, nrm, ok = dense_surface(xyz, grid_normals(xyz, valid), valid, scale, rows, cols)
    # only keep surface that lies inside the downloaded CT (outside it the render would be zeros)
    inside = render(pos, nrm, ok, ct, meta["origin_zyx"], [0], 1.0)[0] > 0
    ok = ok & inside
    print(f"valid surface px {ok.mean():.1%} (inside the CT download)", flush=True)

    from registry import SCROLL4_PAIR, run_dir
    runs = [run_dir(r) for r in (RUNS or SCROLL4_PAIR)]
    models = []
    for r in runs:
        ck = torch.load(r / "model.pt", map_location="cpu", weights_only=False)
        m = build(ck["args"]["model"], N_LAYERS, ck["args"].get("encoder", "resnet34")).cuda().eval()
        m.load_state_dict(ck["model"]); models.append((m, ck["args"].get("norm", "fixed")))
    print("models:", [f"{r.name} ({mode})" for r, (_, mode) in zip(runs, models)], flush=True)

    @torch.no_grad()
    def predict(vol):
        P, S = 384, 128
        H0, W0 = vol.shape[-2:]
        if H0 < P or W0 < P:                         # surface smaller than one window: pad with grey
            pad = np.full((vol.shape[0], max(H0, P), max(W0, P)), np.median(vol), np.uint8)
            pad[:, :H0, :W0] = vol
            vol = pad
        v = torch.from_numpy(vol).cuda()
        H, W = v.shape[-2:]
        r = torch.minimum(torch.arange(P, device="cuda") + 1, torch.arange(P, 0, -1, device="cuda")).float()
        w = torch.outer(r, r)
        out = torch.zeros(H, W, device="cuda"); ws = torch.zeros_like(out)
        ys = sorted(set(list(range(0, max(H - P, 0) + 1, S)) + [max(H - P, 0)]))
        xs = sorted(set(list(range(0, max(W - P, 0) + 1, S)) + [max(W - P, 0)]))
        for y in ys:
            for x in xs:
                win = v[JITTER:JITTER + N_LAYERS, y:y + P, x:x + P]
                with torch.autocast("cuda", torch.bfloat16):
                    p = sum(torch.sigmoid(mm(norm(win, mode)[None]).float())[0, 0] for mm, mode in models) / len(models)
                out[y:y + P, x:x + P] += p * w; ws[y:y + P, x:x + P] += w
        return (out / ws).cpu().numpy()[:H0, :W0]

    results = {}
    ok1 = ok.reshape(len(rows) // 2, 2, len(cols) // 2, 2).all((1, 3))
    for sign in (1.0, -1.0):
        full = render(pos, nrm, ok, ct, meta["origin_zyx"], range(-CENTRE, CENTRE + 1), sign)
        if HARMONIZE:
            full = harmonize(full, ok, BLUR)
        if FILL_HOLES:
            # holes in our surface would be black (0); fill each layer with its own median papyrus
            # grey so sharp hole edges can't trigger the model
            for li in range(full.shape[0]):
                vals = full[li][ok]
                full[li][~ok] = np.uint8(np.median(vals)) if vals.size else 0
        for shift in (-8, -4, 0, 4, 8):
            sl = full[CENTRE + shift - 18:CENTRE + shift + 18].astype(np.float32)
            l1 = np.rint(sl.reshape(36, len(rows) // 2, 2, len(cols) // 2, 2).mean((2, 4))).astype(np.uint8)
            prob = predict(l1) * ok1
            results[(sign, shift)] = prob
            print(f"normal {sign:+.0f} shift {shift:+d}: mean ink prob {prob[ok1].mean():.3f}, "
                  f"area > 0.5: {np.mean(prob[ok1] > 0.5):.1%}", flush=True)
        if sign == 1.0:
            surf_img = full[CENTRE]
            np.save(D / f"first_ink_{name}_layers.npy", full[CENTRE - 30:CENTRE + 31:6])

    fig, ax = plt.subplots(5, 3, figsize=(18, 22))
    ax[0, 0].imshow(surf_img, cmap="gray"); ax[0, 0].set_title("our surface, layer on the trace (2.4 um)")
    layers = np.load(D / f"first_ink_{name}_layers.npy")
    ax[0, 1].imshow(layers.max(0), cmap="gray"); ax[0, 1].set_title("max over layers -30..+30")
    ax[0, 2].imshow(ok, cmap="gray"); ax[0, 2].set_title("valid surface")
    for k, ((sign, shift), prob) in enumerate(sorted(results.items())[:12]):
        a = ax.flat[3 + k]
        a.imshow(prob, cmap="magma", vmin=0, vmax=1)
        a.set_title(f"ink prob, normal {sign:+.0f}, depth shift {shift:+d}")
    for a in ax.flat:
        a.axis("off")
    tag = name + ("_filled_holes" if FILL_HOLES else "") + (f"_{TAG}" if TAG else "")
    plt.tight_layout(); plt.savefig(D / f"first_ink_{tag}.png", dpi=55); plt.close()
    np.savez_compressed(D / f"first_ink_{tag}.npz", **{f"p_{int(s)}_{sh}": p for (s, sh), p in results.items()})
    print("saved", D / f"first_ink_{tag}.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "run"])
    ap.add_argument("--name", default="test_filled")
    ap.add_argument("--fill-holes", action="store_true")
    ap.add_argument("--ct", default=None, help="name of the trace whose CT download to reuse")
    ap.add_argument("--runs", nargs="+", default=None, help="models to average: short names from src/ink/registry.py or run folders")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--harmonize", action="store_true", help="blur + match brightness to Scroll 4 before the model")
    ap.add_argument("--blur", type=float, default=1.0, help="in-plane blur sigma (px) when harmonizing")
    a = ap.parse_args()
    RUNS = a.runs
    TAG = a.tag
    HARMONIZE, BLUR = a.harmonize, a.blur
    fetch(a.name) if a.step == "fetch" else run(a.name, a.ct)
