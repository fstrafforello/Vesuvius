"""Side-by-side view of saved validation predictions from several runs on the held-out segment.

Uses each run's val_prob.npy (best checkpoint), so no inference is needed. Produces:
  compare_blocks.png  - the blocks with most labelled ink: surface | labels | one column per run
  compare_region.png  - the densest stitched region of the segment, same columns, as rows

Usage:  uv run python compare_preds.py <run_dir> [<run_dir> ...] [--out DIR] [--val w031]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_l1 import B, DATA, load_segment

ROOT = Path(__file__).resolve().parents[2]


def stitch(blocks, arrays, y0, x0, h, w, fill=np.nan):
    out = np.full((h * B, w * B), fill, np.float32)
    for (y, x), a in zip(blocks, arrays):
        by, bx = (y - y0) // B, (x - x0) // B
        if 0 <= by < h and 0 <= bx < w:
            out[by * B:(by + 1) * B, bx * B:(bx + 1) * B] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--val", default="w031")
    ap.add_argument("--out", default=str(ROOT / "runs" / "compare"))
    ap.add_argument("--region", default="8x12", help="region size in blocks, HxW")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    blocks, vol, lab, sup = load_segment(DATA / args.val)
    surf = vol[:, vol.shape[1] // 2].astype(np.float32)
    runs = []
    for r in args.runs:
        r = Path(r)
        prob = np.load(r / "val_prob.npy").astype(np.float32)
        m = json.loads((r / "metrics.json").read_text())["best_val"] if (r / "metrics.json").exists() else \
            __import__("torch").load(r / "model.pt", map_location="cpu", weights_only=False)["metrics"]
        assert prob.shape[0] == len(blocks), f"{r.name}: block count mismatch"
        runs.append((r.name, prob, m))
        print(f"{r.name}: best it {m['iter']}  AUC {m['auc']:.3f}  F0.5 {m['f05']:.3f}")

    titles = ["surface", "labels (gray = unsup.)"] + [f"{n.split('_2026')[0]}\nF0.5 {m['f05']:.3f} AUC {m['auc']:.3f}" for n, _, m in runs]

    # 1) blocks with most labelled ink
    order = np.argsort(-(lab & sup).sum((1, 2)))[:6]
    fig, ax = plt.subplots(len(order), len(titles), figsize=(3.1 * len(titles), 3.1 * len(order)))
    for row, b in enumerate(order):
        ax[row, 0].imshow(surf[b], cmap="gray")
        ax[row, 1].imshow(np.where(sup[b], lab[b], 0.5), cmap="magma", vmin=0, vmax=1)
        for c, (_, prob, _) in enumerate(runs):
            ax[row, 2 + c].imshow(prob[b], cmap="magma", vmin=0, vmax=1)
    for c, t in enumerate(titles):
        ax[0, c].set_title(t, fontsize=9)
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout(); plt.savefig(out / "compare_blocks.png", dpi=70); plt.close()

    # 2) densest region of h x w blocks
    h, w = (int(v) for v in args.region.split("x"))
    have = {tuple(b) for b in blocks}
    ys = sorted({y for y, _ in blocks}); xs = sorted({x for _, x in blocks})
    best = max(((sum((y + i * B, x + j * B) in have for i in range(h) for j in range(w)), y, x)
                for y in ys for x in xs), key=lambda t: t[0])
    _, y0, x0 = best
    print(f"region at level-1 y={y0} x={x0}: {best[0]}/{h * w} blocks present")
    panels = [("surface (layer on the mesh)", stitch(blocks, surf, y0, x0, h, w), "gray", None),
              ("labels (gray = unsupervised / not downloaded)",
               stitch(blocks, np.where(sup, lab, 0.5), y0, x0, h, w, fill=0.5), "magma", (0, 1))]
    panels += [(t.replace("\n", "  "), stitch(blocks, p, y0, x0, h, w), "magma", (0, 1)) for t, (_, p, _) in zip(titles[2:], runs)]
    fig, ax = plt.subplots(len(panels), 1, figsize=(w * 1.3, h * 1.3 * len(panels)))
    for a, (t, img, cm, lim) in zip(ax, panels):
        a.imshow(img, cmap=cm, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
        a.set_title(t, fontsize=11); a.axis("off")
    plt.tight_layout(); plt.savefig(out / "compare_region.png", dpi=60); plt.close()
    print("saved to", out)


if __name__ == "__main__":
    main()
