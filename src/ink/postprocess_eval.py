"""Cheap improvements on saved validation predictions (no training):
  - averaging several runs' probability maps (optionally weighted)
  - removing isolated blobs smaller than a stroke after thresholding

Each block is evaluated with its 3x3 neighbourhood stitched in, so blobs crossing block
borders are measured whole. Scores use the same metrics as training (AUC, best F0.5).

Usage:  uv run python postprocess_eval.py <run_dir> <run_dir> [--val w031]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from train_baseline import metrics
from train_l1 import B, DATA, load_segment


def stitched_regions(blocks):
    """Group blocks into connected regions so post-processing sees whole strokes."""
    have = {tuple(b): i for i, b in enumerate(blocks)}
    seen, regions = set(), []
    for start in have:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            y, x = stack.pop(); comp.append((y, x))
            for dy, dx in ((B, 0), (-B, 0), (0, B), (0, -B)):
                n = (y + dy, x + dx)
                if n in have and n not in seen:
                    seen.add(n); stack.append(n)
        regions.append([have[c] for c in comp])
    return regions


def remove_small(prob, blocks, thr, min_px):
    """Zero out connected components above thr that are smaller than min_px, region by region."""
    out = prob.copy()
    for reg in stitched_regions(blocks):
        ys = [blocks[i][0] for i in reg]; xs = [blocks[i][1] for i in reg]
        y0, x0 = min(ys), min(xs)
        h, w = (max(ys) - y0) // B + 1, (max(xs) - x0) // B + 1
        canvas = np.zeros((h * B, w * B), np.float32)
        for i in reg:
            y, x = (blocks[i][0] - y0) // B, (blocks[i][1] - x0) // B
            canvas[y * B:(y + 1) * B, x * B:(x + 1) * B] = prob[i]
        lab, n = ndimage.label(canvas >= thr)
        if n:
            sizes = ndimage.sum(np.ones_like(canvas), lab, index=np.arange(1, n + 1))
            small = np.isin(lab, np.nonzero(sizes < min_px)[0] + 1)
            canvas[small] *= 0.2          # keep a little signal so AUC still ranks them low, not zero
        for i in reg:
            y, x = (blocks[i][0] - y0) // B, (blocks[i][1] - x0) // B
            out[i] = canvas[y * B:(y + 1) * B, x * B:(x + 1) * B]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--val", default="w031")
    args = ap.parse_args()

    blocks, _, lab, sup = load_segment(DATA / args.val)
    probs = {Path(r).name.split("_2026")[0]: np.load(Path(r) / "val_prob.npy").astype(np.float32) for r in args.runs}
    names = list(probs)
    results = {}

    def score(name, p):
        m = metrics(p, lab, sup)
        results[name] = m
        print(f"{name:45s} AUC {m['auc']:.4f}  F0.5 {m['f05']:.4f} @ {m['threshold']:.2f}", flush=True)
        return m

    for n in names:
        score(n, probs[n])
    candidates = {n: probs[n] for n in names}
    if len(names) >= 2:
        a, b = probs[names[0]], probs[names[1]]
        for w in (0.3, 0.5, 0.7):
            name = f"avg {w:.1f}*{names[0]} + {1 - w:.1f}*{names[1]}"
            candidates[name] = w * a + (1 - w) * b
            score(name, candidates[name])

    # blob removal on each candidate; stroke width here is ~150 px at 1/2 res, so a real stroke
    # fragment is thousands of px; try a few size cut-offs
    best_name = max(results, key=lambda k: results[k]["f05"])
    for base in sorted({best_name, *names}):
        for min_px in (500, 1500, 4000):
            for thr in (0.5,):
                score(f"{base} | drop blobs <{min_px}px", remove_small(candidates[base], blocks, thr, min_px))

    winner = max(results, key=lambda k: results[k]["f05"])
    print(f"\nbest by F0.5: {winner}  ->  {results[winner]}")
    out = Path(args.runs[-1]).parent / "compare" / "postprocess_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"results": results, "winner": winner}, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
