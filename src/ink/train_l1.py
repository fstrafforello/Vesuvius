"""Ink detection at half resolution (surface level 1) on all labelled Scroll 4 areas.

Why half res: ink strokes here are ~300 px wide at full res and letters ~1500 px, so a 256 px
full-res window mostly sees the inside of one stroke. At level 1 a 384 px window covers
768 px of full-res papyrus: whole strokes plus their surroundings.

Data: data/PHerc1667/ink_l1/<seg>/ (src/fetch_ink_l1.py), stored as 256x256 blocks. Training
patches are cut from 2x2 groups of neighbouring blocks, so they can straddle block borders.
Validation: one whole held-out segment (default w031), predicted block by block with its 3x3
neighbourhood as context.

The published community prediction is only read for a reference score; it never enters training.

Usage:  uv run python src/ink/train_l1.py [--iters 15000] [--patch 384]
Outputs go to runs/ink_l1_<timestamp>/.
"""

import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from models import build
from train_baseline import JITTER, N_LAYERS, metrics, norm
from train_multi import community_on_blocks, full_name

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "PHerc1667" / "ink_l1"
B = 256  # block edge, level-1 pixels
NORM = "fixed"      # input normalisation (see train_baseline.norm); set from --norm
GROUP_W = None      # per-group sampling weights (balanced across scrolls); set in main
HOLES = False       # grey-filled hole / edge augmentation; set from --holes
PHOTO = False       # extra brightness-curve augmentation; set from --photo


def open_local(path, driver):
    return ts.open({"driver": driver, "kvstore": {"driver": "file", "path": str(path)}}, read=True).result()


def load_segment(seg_dir):
    blocks = [tuple(b) for b in json.loads((seg_dir / "blocks.json").read_text())]
    surf = open_local(seg_dir / "surface.zarr", "zarr")
    lab = open_local(seg_dir / "labels/inklabels.zarr/1", "zarr3")
    sup = open_local(seg_dir / "labels/supervision.zarr/1", "zarr3")
    h, w = lab.shape
    v = np.zeros((len(blocks), N_LAYERS + 2 * JITTER, B, B), np.uint8)
    l = np.zeros((len(blocks), B, B), bool)
    s = np.zeros((len(blocks), B, B), bool)
    for i, (y, x) in enumerate(blocks):
        yy, xx = min(y + B, h), min(x + B, w)
        v[i, :, :yy - y, :xx - x] = surf[:, y:yy, x:xx].read().result()
        l[i, :yy - y, :xx - x] = lab[y:yy, x:xx].read().result() > 127
        s[i, :yy - y, :xx - x] = sup[y:yy, x:xx].read().result() > 127
    return blocks, v, l, s


class Bank:
    """All blocks of a set of segments on the GPU, plus neighbour lookup."""

    def __init__(self, segs, dev):
        vs, ls, ss, self.index, self.blocks = [], [], [], {}, []
        for k, (name, (blocks, v, l, s)) in enumerate(segs.items()):
            for i, yx in enumerate(blocks):
                self.index[(k, *yx)] = len(self.blocks)
                self.blocks.append((k, *yx))
            vs.append(v); ls.append(l); ss.append(s)
        self.vol = torch.from_numpy(np.concatenate(vs)).to(dev)
        self.lab = torch.from_numpy(np.concatenate(ls)).to(dev)
        self.sup = torch.from_numpy(np.concatenate(ss)).to(dev)
        self.dev = dev

    def neighbour(self, b, dy, dx):
        k, y, x = self.blocks[b]
        return self.index.get((k, y + dy * B, x + dx * B), -1)

    def mosaic(self, b, r):
        """(2r+1)x(2r+1) neighbourhood of block b; missing blocks are zeros (and unsupervised)."""
        n = 2 * r + 1
        vol = torch.zeros(self.vol.shape[1], n * B, n * B, dtype=torch.uint8, device=self.dev)
        lab = torch.zeros(n * B, n * B, dtype=torch.bool, device=self.dev)
        sup = torch.zeros_like(lab)
        for i in range(n):
            for j in range(n):
                nb = self.neighbour(b, i - r, j - r)
                if nb >= 0:
                    sl = (slice(i * B, (i + 1) * B), slice(j * B, (j + 1) * B))
                    vol[(slice(None), *sl)] = self.vol[nb]; lab[sl] = self.lab[nb]; sup[sl] = self.sup[nb]
        return vol, lab, sup


def sample(bank, groups, patch, bs, gen, strong=False):
    """Random patches from 2x2 block groups (512x512 mosaics), augmented on the GPU.

    strong=True adds what the full 3D model needs against memorising texture: random scale
    (0.8-1.25x), blur, noise, and dropping a few depth layers.
    """
    xs, ys, ms = [], [], []
    gi = (torch.multinomial(GROUP_W, bs, replacement=True, generator=gen) if GROUP_W is not None
          else torch.randint(len(groups), (bs,), generator=gen)).tolist()
    rnd = lambda: float(torch.rand(1, generator=gen))
    for g in gi:
        b00, b01, b10, b11 = groups[g]
        v = torch.cat([torch.cat([bank.vol[b00], bank.vol[b01]], -1), torch.cat([bank.vol[b10], bank.vol[b11]], -1)], -2)
        l = torch.cat([torch.cat([bank.lab[b00], bank.lab[b01]], -1), torch.cat([bank.lab[b10], bank.lab[b11]], -1)], -2)
        s = torch.cat([torch.cat([bank.sup[b00], bank.sup[b01]], -1), torch.cat([bank.sup[b10], bank.sup[b11]], -1)], -2)
        cs = min(2 * B, round(patch * (0.8 + 0.45 * rnd()))) if strong else patch
        y, x = (int(t) for t in torch.randint(0, 2 * B - cs + 1, (2,), generator=gen))
        d = int(torch.randint(0, 2 * JITTER + 1, (1,), generator=gen))
        xi, yi, mi = v[d:d + N_LAYERS, y:y + cs, x:x + cs], l[y:y + cs, x:x + cs], s[y:y + cs, x:x + cs]
        if cs != patch:
            xi = F.interpolate(xi[None].float(), size=(patch, patch), mode="bilinear", align_corners=False)[0]
            yi, mi = (F.interpolate(a[None, None].float(), size=(patch, patch), mode="nearest")[0, 0] > 0.5 for a in (yi, mi))
        k = int(torch.randint(4, (1,), generator=gen))
        xi, yi, mi = (torch.rot90(a, k, (-2, -1)) for a in (xi, yi, mi))
        if int(torch.randint(2, (1,), generator=gen)):
            xi, yi, mi = (torch.flip(a, (-1,)) for a in (xi, yi, mi))
        xs.append(xi.float()); ys.append(yi); ms.append(mi)
    x = torch.stack(xs) / 255.0
    a = 1 + 0.3 * (torch.rand(bs, 1, 1, 1, device=x.device) - 0.5)
    c = 0.15 * (torch.rand(bs, 1, 1, 1, device=x.device) - 0.5)
    x = (x - 0.5) * a + 0.5 + c
    if strong:
        for i in range(bs):
            if rnd() < 0.3:                                    # blur
                x[i] = TF.gaussian_blur(x[i], 7, 0.5 + rnd())
            if rnd() < 0.3:                                    # sensor-like noise
                x[i] += (0.02 + 0.04 * rnd()) * torch.randn_like(x[i])
            if rnd() < 0.5:                                    # drop 1-4 depth layers (replace with layer mean)
                for li in torch.randint(N_LAYERS, (int(torch.randint(1, 5, (1,), generator=gen)),), generator=gen).tolist():
                    x[i, li] = x[i, li].mean()
            if PHOTO and rnd() < 0.8:                          # brightness curve (gamma 0.6 - 1.6)
                x[i] = x[i].clamp(0, 1) ** float(np.exp(np.log(0.6) + rnd() * np.log(1.6 / 0.6)))
    ms = torch.stack(ms).float()
    if HOLES:
        # like our rendered surfaces: random blobs and patch borders filled with flat papyrus grey
        # (per layer median), excluded from the loss, so grey-filled boundaries stop looking like ink
        P = x.shape[-1]
        for i in range(bs):
            if rnd() < 0.6:
                noise = torch.rand(1, 1, 12, 12, generator=gen).to(x.device)
                noise = F.interpolate(noise, size=(P, P), mode="bilinear", align_corners=False)[0, 0]
                hole = noise < 0.1 + 0.3 * rnd()                    # 5-40 % of the patch
                if rnd() < 0.5:                                     # sometimes a straight surface edge too
                    e = int(P * 0.4 * rnd())
                    side = int(4 * rnd())
                    if side == 0: hole[:e] = True
                    elif side == 1: hole[P - e:] = True
                    elif side == 2: hole[:, :e] = True
                    else: hole[:, P - e:] = True
                fill = x[i].flatten(1).median(1).values[:, None]
                x[i][:, hole] = fill.expand(-1, int(hole.sum()))
                ms[i, hole] = 0
    if NORM == "fixed":
        x = (x - 0.30) / 0.25
    else:
        x = norm(x.clamp(0, 1) * 255.0, NORM)
    return x, torch.stack(ys).float()[:, None], ms[:, None]


@torch.no_grad()
def predict_bank(model, bank, patch):
    """Predict each block from its 3x3 neighbourhood (768x768), keep the centre 256x256."""
    stride = patch // 2
    size = 3 * B
    pos = list(range(0, size - patch + 1, stride))
    if pos[-1] != size - patch:
        pos.append(size - patch)
    offs = [(y, x) for y in pos for x in pos]
    r = torch.minimum(torch.arange(patch, device=bank.dev) + 1, torch.arange(patch, 0, -1, device=bank.dev)).float()
    w = torch.outer(r, r)
    wsum = torch.zeros(size, size, device=bank.dev)
    for y, x in offs:
        wsum[y:y + patch, x:x + patch] += w
    out = np.zeros((len(bank.blocks), B, B), np.float32)
    for b in range(len(bank.blocks)):
        vol, _, _ = bank.mosaic(b, 1)
        xb = torch.stack([norm(vol[JITTER:JITTER + N_LAYERS, y:y + patch, x:x + patch], NORM) for y, x in offs])
        with torch.autocast("cuda", torch.bfloat16):
            p = torch.sigmoid(model(xb).float())[:, 0]
        acc = torch.zeros(size, size, device=bank.dev)
        for (y, x), pi in zip(offs, p):
            acc[y:y + patch, x:x + patch] += pi * w
        out[b] = (acc / wsum)[B:2 * B, B:2 * B].cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--patch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--model", choices=["2d", "3d", "full3d"], default="2d")
    ap.add_argument("--val", default="w031")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=1000, help="save full resumable state every N iterations")
    ap.add_argument("--pause-at", default=None, help="HH:MM - save state and exit cleanly at this time")
    ap.add_argument("--pause-iter", type=int, default=None, help="save state and exit cleanly after this iteration")
    ap.add_argument("--resume", default=None, help="run folder to continue (its saved settings are reused)")
    ap.add_argument("--aug", choices=["basic", "strong"], default="basic")
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--dropout", type=float, default=0.0, help="decoder dropout (full3d)")
    ap.add_argument("--enc-lr-mult", type=float, default=1.0, help="learning-rate multiplier for the pretrained encoder (full3d)")
    ap.add_argument("--norm", choices=["fixed", "robust"], default="fixed", help="input normalisation")
    ap.add_argument("--photo", action="store_true", help="extra brightness-curve (gamma) augmentation")
    ap.add_argument("--scrolls", nargs="+", default=["PHerc1667"], help="scrolls whose labelled blocks to use")
    ap.add_argument("--val-scroll", default=None, help="hold out ALL segments of this scroll for validation")
    ap.add_argument("--balance", action="store_true", help="sample every training scroll equally often")
    ap.add_argument("--holes", action="store_true", help="grey-filled hole / edge augmentation")
    cli = ap.parse_args()

    state = None
    if cli.resume:
        run = Path(cli.resume).resolve()
        state = torch.load(run / "state.pt", map_location="cpu", weights_only=False)
        defaults = vars(ap.parse_args([]))     # runs saved before an option existed get its default
        args = argparse.Namespace(**{**defaults, **state["args"], "pause_at": cli.pause_at, "pause_iter": cli.pause_iter, "resume": cli.resume})
        print(f"resuming {run.name} from iteration {state['iter']}", flush=True)
    else:
        args = cli
        run = ROOT / "runs" / f"ink_l1_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
        run.mkdir(parents=True)
    assert args.patch <= 2 * B
    global NORM, PHOTO, HOLES
    NORM, PHOTO, HOLES = args.norm, args.photo, getattr(args, "holes", False)
    torch.manual_seed(args.seed)
    dev = "cuda"
    pause_at = None
    if args.pause_at:
        hh, mm = (int(v) for v in args.pause_at.split(":"))
        now = datetime.datetime.now()
        pause_at = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if pause_at <= now:
            pause_at += datetime.timedelta(days=1)
        print(f"will pause at {pause_at:%Y-%m-%d %H:%M} (or when a PAUSE file appears in {run})", flush=True)

    train_segs, val_segs, seg_scroll = {}, {}, {}
    for scroll in args.scrolls:
        root = ROOT / "data" / scroll / "ink_l1"
        for seg_dir in sorted(p for p in root.iterdir() if (p / "blocks.json").exists()):
            data = load_segment(seg_dir)
            s = data[3]
            is_val = (scroll == args.val_scroll) if args.val_scroll else (scroll == "PHerc1667" and seg_dir.name == args.val)
            key = f"{scroll}/{seg_dir.name}"
            print(f"{key}: {len(data[0])} blocks, sup {s.mean():.2f}, ink/sup {(data[2] & s).sum() / max(s.sum(), 1):.2f}"
                  + ("  [VALIDATION]" if is_val else ""), flush=True)
            (val_segs if is_val else train_segs)[key] = data
            seg_scroll[key] = scroll
    train = Bank(train_segs, dev)
    val = Bank(val_segs, dev)
    groups = [(b, n01, n10, n11) for b in range(len(train.blocks))
              if min(n01 := train.neighbour(b, 0, 1), n10 := train.neighbour(b, 1, 0), n11 := train.neighbour(b, 1, 1)) >= 0]
    # balanced sampling: every training scroll contributes equally, whatever its amount of data
    seg_keys = list(train_segs)
    g_scroll = [seg_scroll[seg_keys[train.blocks[g[0]][0]]] for g in groups]
    counts = {s: g_scroll.count(s) for s in set(g_scroll)}
    global GROUP_W
    GROUP_W = torch.tensor([1.0 / counts[s] for s in g_scroll]) if args.balance else None
    print(f"train {len(train.blocks)} blocks / {len(groups)} 2x2 groups per scroll {counts} "
          f"({train.sup.sum().item() / 1e6:.0f}M sup px at 1/2 res) | "
          f"val {len(val.blocks)} blocks ({val.sup.sum().item() / 1e6:.0f}M sup px)", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    model = build(args.model, N_LAYERS, args.encoder, dropout=args.dropout).to(dev)
    print(f"model {args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | aug {args.aug} "
          f"wd {args.wd} dropout {args.dropout} enc-lr x{args.enc_lr_mult}", flush=True)
    # separate encoder group only when its lr differs (keeps older single-group checkpoints resumable)
    enc = list(model.stages.parameters()) if hasattr(model, "stages") and args.enc_lr_mult != 1.0 else []
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    groups_opt = [{"params": rest}] + ([{"params": enc, "lr": args.lr * args.enc_lr_mult}] if enc else [])
    opt = torch.optim.AdamW(groups_opt, lr=args.lr, weight_decay=args.wd)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, args.iters) / args.iters)))

    vl, vs = val.lab.cpu().numpy(), val.sup.cpu().numpy()
    history, best, t0, start = [], {"f05": -1}, time.time(), 1
    if state:
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        gen.set_state(state["gen"]); torch.cuda.set_rng_state(state["cuda_rng"])
        history, best, start = state["history"], state["best"], state["iter"] + 1
        t0 -= state["elapsed"]
        del state

    def save_state(it):
        tmp = run / "state.pt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": gen.get_state(), "cuda_rng": torch.cuda.get_rng_state(), "iter": it,
                    "history": history, "best": best, "elapsed": time.time() - t0,
                    "args": {k: v for k, v in vars(args).items() if k not in ("resume", "pause_at", "pause_iter")}}, tmp)
        tmp.replace(run / "state.pt")

    def pause_requested(it):
        if pause_at and datetime.datetime.now() >= pause_at:
            return True
        if args.pause_iter and it >= args.pause_iter:
            return True
        return it % 25 == 0 and (run / "PAUSE").exists()

    model.train()
    for it in range(start, args.iters + 1):
        x, y, m = sample(train, groups, args.patch, args.bs, gen, strong=args.aug == "strong")
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(x).float()
        bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * m).sum() / m.sum().clamp(min=1)
        p = torch.sigmoid(logits) * m
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * m).sum() + 1)
        loss = bce + 0.5 * dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if it % 250 == 0:
            print(f"it {it:6d} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if it % args.eval_every == 0 or it == args.iters:
            model.eval(); prob = predict_bank(model, val, args.patch); model.train()
            mt = metrics(prob, vl, vs); mt["iter"] = it
            history.append(mt)
            print(f"  val[{args.val}] auc {mt['auc']:.4f}  F0.5 {mt['f05']:.4f} @ {mt['threshold']:.2f}", flush=True)
            if mt["f05"] > best["f05"]:
                best = mt
                torch.save({"model": model.state_dict(), "args": vars(args), "n_layers": N_LAYERS, "level": 1, "kind": args.model, "metrics": mt}, run / "model.pt")
                np.save(run / "val_prob.npy", prob.astype(np.float16))
        if it % args.ckpt_every == 0 and it < args.iters:
            save_state(it)
            print(f"  checkpoint saved at it {it}", flush=True)
        if it < args.iters and pause_requested(it):
            save_state(it)
            (run / "PAUSE").unlink(missing_ok=True)
            print(f"PAUSED at it {it}. Resume with:\n  uv run python src/ink/train_l1.py --resume \"{run}\"", flush=True)
            return

    # reference only: our best vs the published community prediction, both at 1/8 full res
    prob = np.load(run / "val_prob.npy").astype(np.float32)
    d4 = lambda a: a.reshape(a.shape[0], B // 4, 4, B // 4, 4).mean((2, 4))
    l8, s8 = d4(vl.astype(np.float32)) > 0.5, d4(vs.astype(np.float32)) > 0.99
    compare = {"ours_1_8": metrics(d4(prob), l8, s8)}
    comm = None
    if not args.val_scroll:                                    # community reference exists for Scroll 4 only
        val_dir = DATA / args.val
        full_blocks = [(y * 2, x * 2) for _, y, x in val.blocks]   # level-1 -> level-0 coords
        comm = community_on_blocks(full_name(args.val), val_dir, full_blocks)
    if comm is not None:
        comm = comm[:, :B // 4, :B // 4] if comm.shape[1] >= B // 4 else comm   # 512 full-res = 64 px at 1/8
        compare["community_1_8_reference"] = metrics(comm, l8, s8)
    (run / "metrics.json").write_text(json.dumps({"best_val": best, "history": history, "compare_at_1_8_res": compare,
                                                  "train_minutes": round((time.time() - t0) / 60, 1), "args": vars(args)}, indent=2))

    # figure: the 6 val blocks with most labelled ink, each shown with its 3x3 context
    order = np.argsort(-(vl & vs).sum((1, 2)))[:6]
    fig, ax = plt.subplots(len(order), 3, figsize=(10, 3.4 * len(order)))
    for r, b in enumerate(order):
        vol, lab, sup = val.mosaic(int(b), 1)
        ax[r, 0].imshow(vol[JITTER + N_LAYERS // 2].cpu(), cmap="gray")
        ax[r, 1].imshow(np.where(sup.cpu(), lab.cpu(), 0.5), cmap="magma", vmin=0, vmax=1)
        pm = np.full((3 * B, 3 * B), np.nan, np.float32)
        for i in range(3):
            for j in range(3):
                nb = val.neighbour(int(b), i - 1, j - 1)
                if nb >= 0:
                    pm[i * B:(i + 1) * B, j * B:(j + 1) * B] = prob[nb]
        ax[r, 2].imshow(pm, cmap="magma", vmin=0, vmax=1)
    for c, t in enumerate(["surface (1/2 res)", "labels (gray = unsup./missing)", f"ours  F0.5={best['f05']:.3f}"]):
        ax[0, c].set_title(t)
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout(); plt.savefig(run / "val_blocks.png", dpi=70)
    print(json.dumps({"best": best, "compare": compare}, indent=2))
    print("saved to", run)


if __name__ == "__main__":
    main()
