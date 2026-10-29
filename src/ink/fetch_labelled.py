"""Fetch labelled level-1 ink blocks from any scroll in the open-data bucket.

For every segment of the scroll that has ink labels, the labels are paired with the surface volume
of the SAME scan (labels folder 'X-volume-<scan>' <-> surface volume '...-volume-<scan>.zarr').
Up to MAX_BLOCKS 256x256 level-1 blocks with >= MIN_SUP supervision are fetched, 36 layers around
the surface, in the same layout as data/PHerc1667/ink_l1 so train_l1.py can load them.

Usage:  uv run python src/ink/fetch_labelled.py PHerc0139 PHercParis4 [--max-blocks 80]
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import tensorstore as ts

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fetch_starter import BUCKET, mirror, remote, session  # noqa: E402

B, HALF, CENTRE, MIN_SUP = 256, 18, 54, 0.3


def ls(p):
    out, tok = [], None
    while True:
        q = {"list-type": "2", "delimiter": "/", "prefix": p}
        if tok:
            q["continuation-token"] = tok
        r = ET.fromstring(session.get(BUCKET + "/", params=q, timeout=60).content)
        ns = {"s": r.tag.split("}")[0][1:]}
        out += [c.find("s:Prefix", ns).text for c in r.findall("s:CommonPrefixes", ns)]
        if r.findtext("s:IsTruncated", namespaces=ns) != "true":
            return out
        tok = r.findtext("s:NextContinuationToken", namespaces=ns)


def pairs(scroll):
    """(segment prefix, labels prefix, surface-volume prefix) for each labelled ~2.4 um segment."""
    out = []
    for seg in ls(f"{scroll}/segments/"):
        labs = [l for l in ls(seg + "ink-labels/") if l.split("/")[-2].startswith("2.")]   # ~2.4 um labels
        svs = [v for v in ls(seg + "surface-volumes/") if not v.rstrip("/").endswith("-L1.zarr")]
        for lab in labs:
            scan = lab.split("/")[-2].split("volume-")[1]
            sv = [v for v in svs if f"volume-{scan}.zarr" in v]
            dates = ls(lab)
            if sv and dates:
                out.append((seg, sorted(dates)[-1], sv[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scrolls", nargs="+")
    ap.add_argument("--max-blocks", type=int, default=80)
    a = ap.parse_args()
    rng = random.Random(0)
    for scroll in a.scrolls:
        for seg, lab, sv in pairs(scroll):
            name = seg.split("/")[-2][:14]
            out = ROOT / "data" / scroll / "ink_l1" / name
            if (out / "blocks.json").exists():
                print(f"{scroll}/{name}: already fetched", flush=True); continue
            out.mkdir(parents=True, exist_ok=True)
            mirror(lab.rstrip("/"), out / "labels")
            sup = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(out / "labels/supervision.zarr/3")}},
                          read=True).result().read().result() > 0
            b = B // 4
            h, w = sup.shape[0] // b, sup.shape[1] // b
            cover = sup[:h * b, :w * b].reshape(h, b, w, b).mean((1, 3))
            eligible = {(int(y), int(x)) for y, x in zip(*np.nonzero(cover >= MIN_SUP))}
            # pick 2x2 clusters of neighbouring blocks: training cuts 384 px windows from 2x2 groups,
            # so scattered single blocks would hardly be usable
            seeds = sorted(eligible); rng.shuffle(seeds)
            chosen = set()
            for y, x in seeds:
                quad = [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)]
                if all(q in eligible for q in quad) and not any(q in chosen for q in quad):
                    chosen.update(quad)
                if len(chosen) >= a.max_blocks:
                    break
            blocks = [(y * B, x * B) for y, x in sorted(chosen)]
            src = remote(f"{sv}1")
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
            (out / "source.json").write_text(json.dumps({"segment": seg, "labels": lab, "surface_volume": sv}))
            print(f"{scroll}/{name}: {len(blocks)} blocks from {sv.split('/')[-2]}", flush=True)


if __name__ == "__main__":
    main()
