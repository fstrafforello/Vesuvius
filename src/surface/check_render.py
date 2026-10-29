"""Check our renderer against an official surface volume, then check the ink model gives the
same answer on both.

  prep   - fetch the w031 mesh, and for a 1024x1024 (level-0) area with lots of labelled ink:
           the CT around it and the official surface volume
  check  - render the area ourselves, compare per layer with the official one (both normal
           directions), then run the averaged ink model on both and score against the labels

Usage:  uv run python src/surface/check_render.py prep
        uv run python src/surface/check_render.py check
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorstore as ts

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "ink")]

from fetch_starter import BUCKET, SCAN_ID, mirror, remote  # noqa: E402
from render import dense_surface, grid_normals, load_tifxyz, render  # noqa: E402

SEG = "PHerc1667/segments/20251223230000-w031_2025122323_flatboi"
MESH = f"{SEG}/mesh/20251223230000-on-{SCAN_ID}-2.399um.tifxyz"
SURF = f"{SEG}/surface-volumes/2.399um-0.22m-78keV-volume-{SCAN_ID}.zarr"
CT = f"PHerc1667/volumes/{SCAN_ID}-2.399um-0.2m-78keV-masked.zarr"
OUT = ROOT / "data" / "PHerc1667" / "render_check" / "w031"
L1 = ROOT / "data" / "PHerc1667" / "ink_l1" / "w031"
SIZE = 1024          # level-0 surface pixels = 2x2 level-1 blocks
CENTRE = 54          # layer index on the mesh (109 layers)
MARGIN = 70          # voxels around the surface bbox (54 layers + interpolation)


def local(path, driver="zarr"):
    return ts.open({"driver": driver, "kvstore": {"driver": "file", "path": str(path)}}, read=True).result()


def save(arr, path, chunks):
    ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)},
             "metadata": {"shape": list(arr.shape), "dtype": "|u1", "chunks": list(chunks),
                          "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}},
             "create": True, "delete_existing": True}).result().write(arr).result()


def prep():
    OUT.mkdir(parents=True, exist_ok=True)
    mirror(MESH, OUT / "mesh")

    # 2x2 neighbouring level-1 blocks we already have, with the most labelled ink
    blocks = {tuple(b) for b in json.loads((L1 / "blocks.json").read_text())}
    lab = local(L1 / "labels/inklabels.zarr/1", "zarr3")
    sup = local(L1 / "labels/supervision.zarr/1", "zarr3")
    best = None
    for y, x in blocks:
        if all((y + dy, x + dx) in blocks for dy in (0, 256) for dx in (0, 256)):
            ink = int(((lab[y:y + 512, x:x + 512].read().result() > 127) & (sup[y:y + 512, x:x + 512].read().result() > 127)).sum())
            if best is None or ink > best[0]:
                best = (ink, y, x)
    _, y1, x1 = best
    r0, c0 = 2 * y1, 2 * x1
    print(f"area: level-1 ({y1}, {x1}) = level-0 rows {r0}..{r0 + SIZE}, cols {c0}..{c0 + SIZE}; ink px (l1) {best[0]}")

    xyz, valid, scale = load_tifxyz(OUT / "mesh")
    g0, g1 = int(r0 * scale) - 2, int((r0 + SIZE) * scale) + 3
    h0, h1 = int(c0 * scale) - 2, int((c0 + SIZE) * scale) + 3
    pts = xyz[g0:g1, h0:h1][valid[g0:g1, h0:h1]]
    lo = np.floor(pts.min(0)).astype(int) - MARGIN
    hi = np.ceil(pts.max(0)).astype(int) + MARGIN
    ct_origin = [int(lo[2]), int(lo[1]), int(lo[0])]
    shape = [int(hi[2] - lo[2]), int(hi[1] - lo[1]), int(hi[0] - lo[0])]
    print(f"CT crop zyx origin {ct_origin} shape {shape} ({np.prod(shape) / 1e9:.2f} GB)", flush=True)

    ct = remote(f"{CT}/0")[ct_origin[0]:ct_origin[0] + shape[0], ct_origin[1]:ct_origin[1] + shape[1],
                           ct_origin[2]:ct_origin[2] + shape[2]].read().result()
    save(ct, OUT / "ct.zarr", (128, 128, 128))
    off = remote(f"{SURF}/0")[:, r0:r0 + SIZE, c0:c0 + SIZE].read().result()
    save(off, OUT / "official_l0.zarr", (109, 256, 256))
    (OUT / "area.json").write_text(json.dumps({"r0": r0, "c0": c0, "size": SIZE, "l1_yx": [y1, x1],
                                               "ct_origin_zyx": ct_origin, "ct_shape": shape, "scale": scale}, indent=2))
    print("prep done ->", OUT)


def corr(a, b, m):
    a, b = a[m].astype(np.float64), b[m].astype(np.float64)
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def check():
    import torch
    from models import build
    from train_baseline import JITTER, N_LAYERS, metrics, norm

    area = json.loads((OUT / "area.json").read_text())
    r0, c0, n = area["r0"], area["c0"], area["size"]
    ct = local(OUT / "ct.zarr").read().result()
    official = local(OUT / "official_l0.zarr").read().result()
    xyz, valid, scale = load_tifxyz(OUT / "mesh")
    pos, nrm, ok = dense_surface(xyz, grid_normals(xyz, valid), valid, scale, range(r0, r0 + n), range(c0, c0 + n))
    m = ok & (official[CENTRE] > 0)
    print(f"valid surface px {m.mean():.1%}")

    # 1) which normal direction matches, and does layer k line up with official layer k?
    probe = [-30, -15, 0, 15, 30]
    res = {}
    for sign in (1.0, -1.0):
        ours = render(pos, nrm, ok, ct, area["ct_origin_zyx"], probe, sign)
        res[sign] = [corr(ours[i], official[CENTRE + k], m) for i, k in enumerate(probe)]
        print(f"normal sign {sign:+.0f}: corr per offset {dict(zip(probe, np.round(res[sign], 3)))}")
    sign = max(res, key=lambda s: np.mean(res[s]))

    # fine alignment: best official layer for a few of our offsets
    ours = render(pos, nrm, ok, ct, area["ct_origin_zyx"], range(-CENTRE, CENTRE + 1), sign)
    for k in (-20, 0, 20):
        cs = [corr(ours[CENTRE + k], official[CENTRE + k + d], m) for d in range(-3, 4)]
        print(f"  our offset {k:+d}: best official shift {int(np.argmax(cs)) - 3:+d}, corr {max(cs):.3f} (no shift {cs[3]:.3f})")
    full_corr = np.mean([corr(ours[i], official[i], m) for i in range(CENTRE - 18, CENTRE + 18)])
    mad = np.mean(np.abs(ours[CENTRE - 18:CENTRE + 18].astype(np.int16) - official[CENTRE - 18:CENTRE + 18])[:, m])
    print(f"layers {CENTRE - 18}..{CENTRE + 17}: mean corr {full_corr:.4f}, mean abs diff {mad:.2f} grey levels")

    # 2) ink model on both, at level 1 (2x2 mean), layers 36..72
    def to_l1(v):
        v = v[CENTRE - 18:CENTRE + 18].astype(np.float32)
        return np.rint(v.reshape(36, n // 2, 2, n // 2, 2).mean((2, 4))).astype(np.uint8)
    ours_l1, off_l1 = to_l1(ours), to_l1(official)
    y1, x1 = area["l1_yx"]
    lab = local(L1 / "labels/inklabels.zarr/1", "zarr3")[y1:y1 + n // 2, x1:x1 + n // 2].read().result() > 127
    sup = local(L1 / "labels/supervision.zarr/1", "zarr3")[y1:y1 + n // 2, x1:x1 + n // 2].read().result() > 127
    stored = local(L1 / "surface.zarr")[:, y1:y1 + n // 2, x1:x1 + n // 2].read().result()
    print(f"sanity: our L1 downsample of official vs stored L1 blocks: mean abs diff {np.abs(off_l1.astype(int) - stored).mean():.2f}")

    from registry import SCROLL4_PAIR, run_dir
    runs = [run_dir(r) for r in SCROLL4_PAIR]
    models = []
    for r in runs:
        ck = torch.load(r / "model.pt", map_location="cpu", weights_only=False)
        mdl = build(ck["args"]["model"], N_LAYERS, ck["args"].get("encoder", "resnet34")).cuda().eval()
        mdl.load_state_dict(ck["model"]); models.append(mdl)

    @torch.no_grad()
    def predict(vol):
        v = torch.from_numpy(vol).cuda()
        P, S = 384, 128
        h = v.shape[-1]
        r = torch.minimum(torch.arange(P, device="cuda") + 1, torch.arange(P, 0, -1, device="cuda")).float()
        w = torch.outer(r, r)
        out = torch.zeros(h, h, device="cuda"); ws = torch.zeros_like(out)
        pos_ = sorted(set(list(range(0, h - P + 1, S)) + [h - P]))
        for y in pos_:
            for x in pos_:
                xb = norm(v[JITTER:JITTER + N_LAYERS, y:y + P, x:x + P])[None]
                with torch.autocast("cuda", torch.bfloat16):
                    p = sum(torch.sigmoid(mm(xb).float())[0, 0] for mm in models) / len(models)
                out[y:y + P, x:x + P] += p * w; ws[y:y + P, x:x + P] += w
        return (out / ws).cpu().numpy()

    p_off, p_ours = predict(off_l1), predict(ours_l1)
    mo, mu = metrics(p_off, lab, sup), metrics(p_ours, lab, sup)
    agree = corr(p_off, p_ours, np.ones_like(p_off, bool))
    print(f"ink on official render: AUC {mo['auc']:.4f} F0.5 {mo['f05']:.4f}")
    print(f"ink on OUR render:      AUC {mu['auc']:.4f} F0.5 {mu['f05']:.4f}")
    print(f"prediction agreement (corr) {agree:.4f}, mean abs diff {np.abs(p_off - p_ours).mean():.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].imshow(official[CENTRE], cmap="gray"); ax[0, 0].set_title("official layer 54 (L0)")
    ax[0, 1].imshow(ours[CENTRE], cmap="gray"); ax[0, 1].set_title(f"ours layer 54 (normal sign {sign:+.0f})")
    ax[0, 2].imshow(np.abs(ours[CENTRE].astype(int) - official[CENTRE]), cmap="magma", vmax=40); ax[0, 2].set_title("|difference|")
    ax[1, 0].imshow(np.where(sup, lab, 0.5), cmap="magma", vmin=0, vmax=1); ax[1, 0].set_title("labels")
    ax[1, 1].imshow(p_off, cmap="magma", vmin=0, vmax=1); ax[1, 1].set_title(f"ink on official  F0.5 {mo['f05']:.3f}")
    ax[1, 2].imshow(p_ours, cmap="magma", vmin=0, vmax=1); ax[1, 2].set_title(f"ink on ours  F0.5 {mu['f05']:.3f}")
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "render_check.png", dpi=60)
    (OUT / "render_check.json").write_text(json.dumps({"normal_sign": sign, "layer_corr": float(full_corr), "mad": float(mad),
                                                       "ink_official": mo, "ink_ours": mu, "agreement": agree}, indent=2))
    print("saved", OUT / "render_check.png")


if __name__ == "__main__":
    {"prep": prep, "check": check}[sys.argv[1]]()
