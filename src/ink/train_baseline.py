"""Ink-detection baseline on the w013 full-res tile.

2.5D approach: the surface-volume layers around the papyrus surface become the input channels of a
2D U-Net, which predicts a per-pixel ink probability. Loss is computed only where the supervision
mask says the labels are trustworthy.

Split is spatial, never random: train on the top labelled band of the tile, validate on the
bottom-right labelled block. Random pixel splits leak (neighbouring pixels are near-identical).

Usage:  uv run python src/ink/train_baseline.py [--iters 4000]
Outputs go to runs/ink_baseline_<timestamp>/.
"""

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import tensorstore as ts
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "PHerc1667" / "w013"

SURFACE_LAYER = 54          # middle of the 109 rendered layers = on the mesh
N_LAYERS = 32               # channels fed to the model
JITTER = 2                  # random +-layer shift as augmentation (depth misalignment happens)
PATCH = 256
VAL_Y, VAL_X = 2048, 2048   # validation block = tile[VAL_Y:, VAL_X:]
MARGIN = 64                 # keep training patches this far from the validation block


def open_local(name, driver):
    return ts.open({"driver": driver, "kvstore": {"driver": "file", "path": str(D / name)}}, read=True).result()


def load_tile():
    m = json.loads((D / "manifest.json").read_text())
    y0, x0 = m["surface_tile_yx"]
    t = m["tile_size"]
    lo = SURFACE_LAYER - N_LAYERS // 2 - JITTER
    vol = open_local("surface_crop.zarr", "zarr")[lo:lo + N_LAYERS + 2 * JITTER].read().result()
    lab = open_local("labels/inklabels.zarr/0", "zarr3")[y0:y0 + t, x0:x0 + t].read().result() > 0
    sup = open_local("labels/supervision.zarr/0", "zarr3")[y0:y0 + t, x0:x0 + t].read().result() > 0
    return vol, lab, sup, m


def val_mask(shape):
    v = np.zeros(shape, bool)
    v[VAL_Y:, VAL_X:] = True
    return v


def train_origins(sup, stride=32, min_sup=0.3):
    """Top-left corners of patches that are mostly supervised and well clear of the val block."""
    h, w = sup.shape
    integral = np.pad(sup.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    out = []
    for y in range(0, h - PATCH + 1, stride):
        for x in range(0, w - PATCH + 1, stride):
            if y + PATCH > VAL_Y - MARGIN and x + PATCH > VAL_X - MARGIN:
                continue
            s = integral[y + PATCH, x + PATCH] - integral[y, x + PATCH] - integral[y + PATCH, x] + integral[y, x]
            if s >= min_sup * PATCH * PATCH:
                out.append((y, x))
    return np.array(out)


def batch(vol, lab, sup, origins, bs, gen):
    """Crop, augment and normalise a batch entirely on the GPU."""
    idx = torch.randint(len(origins), (bs,), generator=gen)
    xs, ys, ms = [], [], []
    for i in idx.tolist():
        y, x = (int(v) for v in origins[i])
        y = min(max(y + int(torch.randint(-16, 17, (1,), generator=gen)), 0), vol.shape[1] - PATCH)
        x = min(max(x + int(torch.randint(-16, 17, (1,), generator=gen)), 0), vol.shape[2] - PATCH)
        d = int(torch.randint(0, 2 * JITTER + 1, (1,), generator=gen))
        xi = vol[d:d + N_LAYERS, y:y + PATCH, x:x + PATCH]
        yi = lab[y:y + PATCH, x:x + PATCH]
        mi = sup[y:y + PATCH, x:x + PATCH]
        k = int(torch.randint(4, (1,), generator=gen))
        flip = bool(torch.randint(2, (1,), generator=gen))
        xi, yi, mi = (torch.rot90(a, k, (-2, -1)) for a in (xi, yi, mi))
        if flip:
            xi, yi, mi = (torch.flip(a, (-1,)) for a in (xi, yi, mi))
        xs.append(xi); ys.append(yi); ms.append(mi)
    x = torch.stack(xs).float() / 255.0
    # brightness / contrast jitter, per sample
    a = 1 + 0.2 * (torch.rand(bs, 1, 1, 1, device=x.device) - 0.5)
    b = 0.1 * (torch.rand(bs, 1, 1, 1, device=x.device) - 0.5)
    x = ((x - 0.5) * a + 0.5 + b - 0.30) / 0.25
    return x, torch.stack(ys).float()[:, None], torch.stack(ms).float()[:, None]


def norm(x, mode="fixed"):
    """Input normalisation for the ink models. x: uint8 or 0..255 float, (C, H, W) or (B, C, H, W).

    fixed  - one global formula (what the first models were trained with)
    robust - per sample: subtract the median papyrus grey and divide by its spread (p90 - p10),
             both measured on non-zero voxels. Removes brightness/contrast differences between
             scans and scrolls before the model sees anything.
    """
    x = x.float()
    if mode == "fixed":
        return (x / 255.0 - 0.30) / 0.25
    single = x.dim() == 3
    xb = x[None] if single else x
    flat = xb.reshape(xb.shape[0], -1)[:, ::7]                      # subsample for speed
    out = []
    for i in range(xb.shape[0]):
        v = flat[i][flat[i] > 0]
        if v.numel() < 100:
            v = flat[i]
        q = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9], device=v.device))
        out.append((xb[i] - q[1]) / torch.clamp(q[2] - q[0], min=5.0) * 2.0)
    out = torch.stack(out)
    return out[0] if single else out


@torch.no_grad()
def predict(model, vol, region, stride=128):
    """Sliding-window inference over region=(y0,y1,x0,x1) with linear-ramp blending."""
    y0, y1, x0, x1 = region
    out = torch.zeros(y1 - y0, x1 - x0, device=vol.device)
    wsum = torch.zeros_like(out)
    r = torch.minimum(torch.arange(PATCH, device=vol.device) + 1, torch.arange(PATCH, 0, -1, device=vol.device)).float()
    w = torch.outer(r, r)
    ys = list(range(y0, y1 - PATCH + 1, stride)) or [y0]
    xs = list(range(x0, x1 - PATCH + 1, stride)) or [x0]
    if ys[-1] != y1 - PATCH: ys.append(y1 - PATCH)
    if xs[-1] != x1 - PATCH: xs.append(x1 - PATCH)
    coords = [(y, x) for y in ys for x in xs]
    for i in range(0, len(coords), 32):
        cs = coords[i:i + 32]
        xb = torch.stack([norm(vol[JITTER:JITTER + N_LAYERS, y:y + PATCH, x:x + PATCH]) for y, x in cs])
        with torch.autocast("cuda", torch.bfloat16):
            p = torch.sigmoid(model(xb).float())[:, 0]
        for (y, x), pi in zip(cs, p):
            out[y - y0:y - y0 + PATCH, x - x0:x - x0 + PATCH] += pi * w
            wsum[y - y0:y - y0 + PATCH, x - x0:x - x0 + PATCH] += w
    return (out / wsum).cpu().numpy()


def metrics(prob, lab, sup):
    p, t = prob[sup].astype(np.float64), lab[sup]
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = t.sum(), (~t).sum()
    auc = (ranks[t].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1)
    best = (0.0, 0.5)
    for th in np.arange(0.05, 0.96, 0.05):
        pr = p >= th
        tp = (pr & t).sum(); prec = tp / max(pr.sum(), 1); rec = tp / max(npos, 1)
        f = 1.25 * prec * rec / max(0.25 * prec + rec, 1e-9)   # F0.5, the Kaggle ink-detection metric
        best = max(best, (f, th))
    return {"auc": float(auc), "f05": float(best[0]), "threshold": float(best[1]), "ink_fraction": float(t.mean())}


def community_baseline(m, lab, sup):
    """Score the published community prediction (1/8 res JPG) on our val block, at 1/8 res."""
    jpg = next((D / "preds").glob("*2.399um*.jpg"))
    y0, x0 = m["surface_tile_yx"]; t = m["tile_size"]
    full = np.asarray(Image.open(jpg).convert("L"), dtype=np.float32) / 255.0
    crop = full[y0 // 8:(y0 + t) // 8, x0 // 8:(x0 + t) // 8]
    l8 = lab.reshape(t // 8, 8, t // 8, 8).mean((1, 3)) > 0.5
    s8 = sup.reshape(t // 8, 8, t // 8, 8).mean((1, 3)) > 0.99
    v8 = np.zeros_like(s8)
    v8[VAL_Y // 8:, VAL_X // 8:] = True
    return crop, l8, s8 & v8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    run = ROOT / "runs" / f"ink_baseline_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True)
    dev = "cuda"

    vol_np, lab_np, sup_np, m = load_tile()
    origins = train_origins(sup_np)
    vmask = val_mask(sup_np.shape)
    print(f"tile {vol_np.shape} | train patches {len(origins)} | "
          f"train sup px {(sup_np & ~vmask).sum() / 1e6:.1f}M | val sup px {(sup_np & vmask).sum() / 1e6:.1f}M")

    vol = torch.from_numpy(vol_np).to(dev)
    lab = torch.from_numpy(lab_np).to(dev)
    sup = torch.from_numpy(sup_np).to(dev)
    gen = torch.Generator().manual_seed(args.seed)

    model = smp.Unet(args.encoder, encoder_weights="imagenet", in_channels=N_LAYERS, classes=1).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, args.iters) / args.iters)))

    vr = (VAL_Y, sup_np.shape[0], VAL_X, sup_np.shape[1])
    vlab, vsup = lab_np[VAL_Y:, VAL_X:], sup_np[VAL_Y:, VAL_X:]
    history, best = [], {"f05": -1}
    t0 = time.time()
    model.train()
    for it in range(1, args.iters + 1):
        x, y, msk = batch(vol, lab, sup, origins, args.bs, gen)
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(x.contiguous(memory_format=torch.channels_last)).float()
        bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * msk).sum() / msk.sum().clamp(min=1)
        p = torch.sigmoid(logits) * msk
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * msk).sum() + 1)
        loss = bce + 0.5 * dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()

        if it % 100 == 0:
            print(f"it {it:5d} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if it % 500 == 0 or it == args.iters:
            model.eval()
            prob = predict(model, vol, vr)
            model.train()
            mt = metrics(prob, vlab, vsup)
            mt["iter"] = it
            history.append(mt)
            print(f"  val auc {mt['auc']:.4f}  F0.5 {mt['f05']:.4f} @ {mt['threshold']:.2f}", flush=True)
            if mt["f05"] > best["f05"]:
                best = mt
                torch.save({"model": model.state_dict(), "args": vars(args), "n_layers": N_LAYERS,
                            "surface_layer": SURFACE_LAYER, "metrics": mt}, run / "model.pt")

    # final: best model over the whole tile, and comparison with the community prediction
    model.load_state_dict(torch.load(run / "model.pt", weights_only=False)["model"])
    model.eval()
    full = predict(model, vol, (0, sup_np.shape[0], 0, sup_np.shape[1]))
    Image.fromarray((full * 255).astype(np.uint8)).save(run / "pred_tile.png")

    comm, l8, cmp_mask = community_baseline(m, lab_np, sup_np)
    ours8 = full.reshape(512, 8, 512, 8).mean((1, 3))
    cmp = {"ours_1_8": metrics(ours8, l8, cmp_mask), "community_1_8": metrics(comm, l8, cmp_mask)}

    result = {"best_full_res_val": best, "history": history, "compare_at_1_8_res_on_val": cmp,
              "train_minutes": round((time.time() - t0) / 60, 1), "args": vars(args)}
    (run / "metrics.json").write_text(json.dumps(result, indent=2))

    surf = vol_np[JITTER + N_LAYERS // 2]
    fig, ax = plt.subplots(2, 2, figsize=(14, 14))
    panels = [(surf, "surface layer (gray)", "gray"), (np.where(sup_np, lab_np, 0.5), "labels (gray = unsupervised)", "magma"),
              (full, f"ours  val F0.5={best['f05']:.3f} AUC={best['auc']:.3f}", "magma"),
              (np.kron(comm, np.ones((8, 8))), f"community  (1/8 res) F0.5={cmp['community_1_8']['f05']:.3f}", "magma")]
    for a, (img, title, cm) in zip(ax.flat, panels):
        a.imshow(img, cmap=cm, vmin=0, vmax=255 if img is surf else 1)
        a.add_patch(plt.Rectangle((VAL_X, VAL_Y), 2048, 2048, fill=False, ec="cyan", lw=2))
        a.set_title(title); a.axis("off")
    fig.suptitle("w013 tile - cyan box = held-out validation block")
    plt.tight_layout(); plt.savefig(run / "overview.png", dpi=80)

    print(json.dumps({"best": best, "compare": cmp}, indent=2))
    print("saved to", run)


if __name__ == "__main__":
    main()
