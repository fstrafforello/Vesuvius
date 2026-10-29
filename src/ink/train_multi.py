"""Ink detection trained on labelled blocks from several Scroll 4 segments.

Same 2.5D U-Net as train_baseline.py, but data comes from data/PHerc1667/ink_blocks/ (see
src/fetch_ink_blocks.py). Validation is a whole held-out segment, so the score measures how the
model does on papyrus it has never seen - which is what matters for unlabelled scrolls.

Usage:  uv run python src/ink/train_multi.py [--iters 12000] [--val w031]
Outputs go to runs/ink_multi_<timestamp>/.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import segmentation_models_pytorch as smp
import tensorstore as ts
import torch
import torch.nn.functional as F
from PIL import Image

from train_baseline import JITTER, N_LAYERS, PATCH, metrics, norm

ROOT = Path(__file__).resolve().parents[2]
BLOCKS = ROOT / "data" / "PHerc1667" / "ink_blocks"
BLOCK = 512
BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


def open_local(path, driver):
    return ts.open({"driver": driver, "kvstore": {"driver": "file", "path": str(path)}}, read=True).result()


def load_segment(seg_dir):
    blocks = json.loads((seg_dir / "blocks.json").read_text())
    surf = open_local(seg_dir / "surface.zarr", "zarr")
    lab = open_local(seg_dir / "labels/inklabels.zarr/0", "zarr3")
    sup = open_local(seg_dir / "labels/supervision.zarr/0", "zarr3")
    v = np.stack([surf[:, y:y + BLOCK, x:x + BLOCK].read().result() for y, x in blocks])
    l = np.stack([lab[y:y + BLOCK, x:x + BLOCK].read().result() > 0 for y, x in blocks])
    s = np.stack([sup[y:y + BLOCK, x:x + BLOCK].read().result() > 0 for y, x in blocks])
    return blocks, v, l, s


def sample(vol, lab, sup, bs, gen):
    n = vol.shape[0]
    bi = torch.randint(n, (bs,), generator=gen)
    oy = torch.randint(0, BLOCK - PATCH + 1, (bs,), generator=gen)
    ox = torch.randint(0, BLOCK - PATCH + 1, (bs,), generator=gen)
    dz = torch.randint(0, 2 * JITTER + 1, (bs,), generator=gen)
    rot = torch.randint(4, (bs,), generator=gen)
    flip = torch.randint(2, (bs,), generator=gen)
    xs, ys, ms = [], [], []
    for b, y, x, d, k, f in zip(*(t.tolist() for t in (bi, oy, ox, dz, rot, flip))):
        xi = vol[b, d:d + N_LAYERS, y:y + PATCH, x:x + PATCH]
        yi = lab[b, y:y + PATCH, x:x + PATCH]
        mi = sup[b, y:y + PATCH, x:x + PATCH]
        xi, yi, mi = (torch.rot90(a, k, (-2, -1)) for a in (xi, yi, mi))
        if f:
            xi, yi, mi = (torch.flip(a, (-1,)) for a in (xi, yi, mi))
        xs.append(xi); ys.append(yi); ms.append(mi)
    x = torch.stack(xs).float() / 255.0
    a = 1 + 0.3 * (torch.rand(bs, 1, 1, 1, device=x.device) - 0.5)
    c = 0.15 * (torch.rand(bs, 1, 1, 1, device=x.device) - 0.5)
    x = ((x - 0.5) * a + 0.5 + c - 0.30) / 0.25
    return x, torch.stack(ys).float()[:, None], torch.stack(ms).float()[:, None]


@torch.no_grad()
def predict_blocks(model, vol):
    """Predict every block with 9 overlapping 256-windows each, ramp-blended."""
    r = torch.minimum(torch.arange(PATCH, device=vol.device) + 1, torch.arange(PATCH, 0, -1, device=vol.device)).float()
    w = torch.outer(r, r)
    offs = [(y, x) for y in (0, 128, 256) for x in (0, 128, 256)]
    out = torch.zeros(vol.shape[0], BLOCK, BLOCK, device=vol.device)
    wsum = torch.zeros(BLOCK, BLOCK, device=vol.device)
    for y, x in offs:
        wsum[y:y + PATCH, x:x + PATCH] += w
    for i in range(0, vol.shape[0], 4):
        xb = torch.stack([norm(vol[b, JITTER:JITTER + N_LAYERS, y:y + PATCH, x:x + PATCH])
                          for b in range(i, min(i + 4, vol.shape[0])) for y, x in offs])
        with torch.autocast("cuda", torch.bfloat16):
            p = torch.sigmoid(model(xb).float())[:, 0].view(-1, len(offs), PATCH, PATCH)
        for j, pb in enumerate(p):
            for (y, x), pi in zip(offs, pb):
                out[i + j, y:y + PATCH, x:x + PATCH] += pi * w
    return (out / wsum).cpu().numpy()


def community_on_blocks(seg_name, seg_dir, blocks):
    """Published community prediction (1/8-res JPG) cut to the same blocks, if available."""
    cache = seg_dir / "community_pred.jpg"
    if not cache.exists():
        from xml.etree import ElementTree as ET
        prefix = f"PHerc1667/segments/{seg_name}/ink-detection/downsampled/"
        root = ET.fromstring(requests.get(BUCKET + "/", params={"list-type": "2", "prefix": prefix}, timeout=60).content)
        ns = {"s": root.tag.split("}")[0][1:]}
        keys = [k.text for k in root.iterfind("s:Contents/s:Key", ns) if "2.399um" in k.text and k.text.endswith(".jpg")]
        if not keys:
            return None
        cache.write_bytes(requests.get(f"{BUCKET}/{keys[0]}", timeout=300).content)
    Image.MAX_IMAGE_PIXELS = None
    full = np.asarray(Image.open(cache).convert("L"), dtype=np.float32) / 255.0
    b8 = BLOCK // 8
    return np.stack([full[y // 8:y // 8 + b8, x // 8:x // 8 + b8] for y, x in blocks])


def down8(a):
    n = a.shape[0]
    return a.reshape(n, BLOCK // 8, 8, BLOCK // 8, 8).mean((2, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--val", default="w031")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    run = ROOT / "runs" / f"ink_multi_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True)

    tr = {"v": [], "l": [], "s": []}
    for seg_dir in sorted(p for p in BLOCKS.iterdir() if (p / "blocks.json").exists()):
        blocks, v, l, s = load_segment(seg_dir)
        print(f"{seg_dir.name}: {len(blocks)} blocks, sup {s.mean():.2f}, ink/sup {(l & s).sum() / max(s.sum(), 1):.2f}"
              + ("  [VALIDATION]" if seg_dir.name == args.val else ""), flush=True)
        if seg_dir.name == args.val:
            val_blocks, vv, vl, vs, val_dir = blocks, v, l, s, seg_dir
        else:
            tr["v"].append(v); tr["l"].append(l); tr["s"].append(s)
    vol = torch.from_numpy(np.concatenate(tr["v"])).to(dev)
    lab = torch.from_numpy(np.concatenate(tr["l"])).to(dev)
    sup = torch.from_numpy(np.concatenate(tr["s"])).to(dev)
    vvol = torch.from_numpy(vv).to(dev)
    print(f"train blocks {vol.shape[0]} ({sup.sum().item() / 1e6:.0f}M sup px) | val blocks {vvol.shape[0]} ({vs.sum() / 1e6:.0f}M sup px)", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    model = smp.Unet(args.encoder, encoder_weights="imagenet", in_channels=N_LAYERS, classes=1).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, args.iters) / args.iters)))

    history, best, t0 = [], {"f05": -1}, time.time()
    model.train()
    for it in range(1, args.iters + 1):
        x, y, m = sample(vol, lab, sup, args.bs, gen)
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(x.contiguous(memory_format=torch.channels_last)).float()
        bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * m).sum() / m.sum().clamp(min=1)
        p = torch.sigmoid(logits) * m
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * m).sum() + 1)
        loss = bce + 0.5 * dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if it % 200 == 0:
            print(f"it {it:6d} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if it % args.eval_every == 0 or it == args.iters:
            model.eval(); prob = predict_blocks(model, vvol); model.train()
            mt = metrics(prob, vl, vs); mt["iter"] = it
            history.append(mt)
            print(f"  val[{args.val}] auc {mt['auc']:.4f}  F0.5 {mt['f05']:.4f} @ {mt['threshold']:.2f}", flush=True)
            if mt["f05"] > best["f05"]:
                best = mt
                torch.save({"model": model.state_dict(), "args": vars(args), "n_layers": N_LAYERS, "metrics": mt}, run / "model.pt")
                np.save(run / "val_prob.npy", prob.astype(np.float16))

    prob = np.load(run / "val_prob.npy").astype(np.float32)
    l8, s8 = down8(vl.astype(np.float32)) > 0.5, down8(vs.astype(np.float32)) > 0.99
    compare = {"ours_1_8": metrics(down8(prob), l8, s8)}
    comm = community_on_blocks(full_name(args.val), val_dir, val_blocks)
    if comm is not None:
        compare["community_1_8"] = metrics(comm, l8, s8)
    result = {"best_full_res_val": best, "history": history, "compare_at_1_8_res": compare,
              "train_minutes": round((time.time() - t0) / 60, 1), "args": vars(args)}
    (run / "metrics.json").write_text(json.dumps(result, indent=2))

    # figure: 6 val blocks with the most ink - surface, labels, ours, community
    order = np.argsort(-(vl & vs).sum((1, 2)))[:6]
    cols = 4 if comm is not None else 3
    fig, ax = plt.subplots(len(order), cols, figsize=(3.2 * cols, 3.2 * len(order)))
    for r, b in enumerate(order):
        ax[r, 0].imshow(vv[b, JITTER + N_LAYERS // 2], cmap="gray")
        ax[r, 1].imshow(np.where(vs[b], vl[b], 0.5), cmap="magma", vmin=0, vmax=1)
        ax[r, 2].imshow(prob[b], cmap="magma", vmin=0, vmax=1)
        if comm is not None:
            ax[r, 3].imshow(comm[b], cmap="magma", vmin=0, vmax=1)
    for c, t in enumerate(["surface", "labels (gray=unsup.)", f"ours F0.5={best['f05']:.3f}", "community (1/8)"][:cols]):
        ax[0, c].set_title(t)
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout(); plt.savefig(run / "val_blocks.png", dpi=70)
    print(json.dumps({"best": best, "compare": compare}, indent=2))
    print("saved to", run)


def full_name(short):
    """'w031' -> '20251223230000-w031_2025122323_flatboi' (the S3 segment folder)."""
    sys.path.insert(0, str(ROOT / "src"))
    from fetch_ink_blocks import SEGMENTS
    return next(n for n in SEGMENTS if n.split("-")[1].split("_")[0] == short)


if __name__ == "__main__":
    main()
