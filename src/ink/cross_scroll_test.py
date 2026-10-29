"""Cross-scroll test: do the Scroll 4-trained ink models find KNOWN ink on another scroll?

PHerc0841 has 3 segments with ink labels on the 2.403 um / 77 keV scan - the same scan series as
our target PHerc0846A. We fetch labelled level-1 blocks (same layout as data/PHerc1667/ink_l1) and
score the models against the labels:
  raw         - straight into the models
  harmonised  - each depth layer's brightness distribution mapped onto Scroll 4's (from our Scroll 4
                training blocks), the step that removed the false "ink everywhere" on PHerc0846A
and the robust-normalisation model on raw input.

Usage:  uv run python src/ink/cross_scroll_test.py fetch
        uv run python src/ink/cross_scroll_test.py score
"""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorstore as ts

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "ink")]
from fetch_starter import BUCKET, mirror, remote  # noqa: E402

SCROLL = "PHerc0841"
SCAN = "20260319124803"
SEGS = ["20260220213127-w00", "20260220214732-auto_grown_20260220144552896", "20260221022814-auto_grown_20260220174252405"]
OUT = ROOT / "data" / SCROLL / "ink_l1"
B, HALF, CENTRE = 256, 18, 54
MAX_BLOCKS, MIN_SUP = 60, 0.3


def fetch():
    rng = random.Random(0)
    for seg in SEGS:
        short = seg[:14]                      # unique timestamp prefix
        out = OUT / short
        out.mkdir(parents=True, exist_ok=True)
        base = f"{SCROLL}/segments/{seg}"
        mirror(f"{base}/ink-labels/2.403um-volume-{SCAN}/20260918", out / "labels")
        sup = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(out / "labels/supervision.zarr/3")}},
                      read=True).result().read().result() > 0
        b = B // 4
        h, w = sup.shape[0] // b, sup.shape[1] // b
        cover = sup[:h * b, :w * b].reshape(h, b, w, b).mean((1, 3))
        blocks = [(int(y) * B, int(x) * B) for y, x in zip(*np.nonzero(cover >= MIN_SUP))]
        rng.shuffle(blocks)
        blocks = blocks[:MAX_BLOCKS]
        src = remote(f"{base}/surface-volumes/2.403um-0.22m-77keV-volume-{SCAN}.zarr/1")
        dst = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(out / "surface.zarr")},
                       "metadata": {"shape": [2 * HALF, *src.shape[1:]], "dtype": "|u1", "chunks": [2 * HALF, B, B],
                                    "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}},
                       "open": True, "create": True}).result()

        def one(yx):
            y, x = yx
            dst[:, y:y + B, x:x + B].write(src[CENTRE - HALF:CENTRE + HALF, y:y + B, x:x + B].read().result()).result()

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(one, blocks))
        (out / "blocks.json").write_text(json.dumps(blocks))
        print(f"{short}: {len(blocks)} labelled blocks fetched", flush=True)


def reference_quantiles():
    """Per-layer brightness quantiles of Scroll 4 training blocks (non-zero voxels)."""
    from train_l1 import DATA, load_segment
    vals = [[] for _ in range(2 * HALF)]
    for seg in ("w013", "w018", "w023"):
        _, v, _, _ = load_segment(DATA / seg)
        v = v[::4]
        for li in range(2 * HALF):
            x = v[:, li][v[:, li] > 0]
            vals[li].append(x[:: max(1, len(x) // 200000)])
    qs = np.linspace(0, 100, 101)
    return np.stack([np.percentile(np.concatenate(v), qs) for v in vals])


def harmonize_blocks(vol, ref_q):
    qs = np.linspace(0, 100, 101)
    out = np.empty_like(vol)
    for li in range(vol.shape[1]):
        layer = vol[:, li].astype(np.float32)
        src_q = np.percentile(layer[layer > 0], qs)
        out[:, li] = np.where(layer > 0, np.clip(np.interp(layer, src_q, ref_q[li]), 0, 255), 0).astype(np.uint8)
    return out


def score():
    import torch
    import train_l1 as T
    from models import build
    from train_baseline import metrics

    ref_q = reference_quantiles()
    from registry import SCROLL4_PAIR, run_dir
    models = {
        "original pair": SCROLL4_PAIR,
        "robust": ["robust"],
        "multi-scroll": ["multiscroll"],      # trained on Scroll 4 + PHerc0139 + Scroll 1, PHerc0841 held out
    }
    loaded = {}
    for k, runs in models.items():
        loaded[k] = []
        for r in runs:
            ck = torch.load(run_dir(r) / "model.pt", map_location="cpu", weights_only=False)
            m = build(ck["args"]["model"], 32, ck["args"].get("encoder", "resnet34")).cuda().eval()
            m.load_state_dict(ck["model"]); loaded[k].append((m, ck["args"].get("norm", "fixed")))

    results = {}
    segs = sorted(p for p in OUT.iterdir() if (p / "blocks.json").exists())
    for seg in segs:
        blocks, v, l, s = T.load_segment(seg)
        print(f"{seg.name}: {len(blocks)} blocks, ink fraction in supervised area {(l & s).sum() / max(s.sum(), 1):.2f}", flush=True)
        variants = {"raw": v, "harmonised": harmonize_blocks(v, ref_q)}
        for vname, vol in variants.items():
            bank = T.Bank({seg.name: (blocks, vol, l, s)}, "cuda")
            for mname, ms in loaded.items():
                if mname in ("robust", "multi-scroll") and vname == "harmonised":
                    continue
                probs = []
                for m, mode in ms:
                    T.NORM = mode
                    probs.append(T.predict_bank(m, bank, 384))
                p = np.mean(probs, 0)
                mt = metrics(p, l, s)
                key = f"{seg.name} | {mname} | {vname}"
                results[key] = {**mt, "mean_prob": float(p[s].mean())}
                print(f"  {mname:14s} {vname:11s} AUC {mt['auc']:.3f}  F0.5 {mt['f05']:.3f}  mean prob {p[s].mean():.3f}", flush=True)
            del bank
            torch.cuda.empty_cache()
    (OUT / "cross_scroll_results.json").write_text(json.dumps(results, indent=2))
    print("saved", OUT / "cross_scroll_results.json")


if __name__ == "__main__":
    {"fetch": fetch, "score": score}[sys.argv[1]]()
