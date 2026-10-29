"""Scan the open-data bucket for scrolls (other than Scroll 4 / PHerc1667) whose segments have ink
labels or published ink detections, and report the resolution of their surface volumes.

Usage:  uv run python src/surface/find_known_ink.py
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fetch_starter import BUCKET, session  # noqa: E402


def ls(prefix):
    out, tok = [], None
    while True:
        p = {"list-type": "2", "delimiter": "/", "prefix": prefix}
        if tok:
            p["continuation-token"] = tok
        r = ET.fromstring(session.get(BUCKET + "/", params=p, timeout=60).content)
        ns = {"s": r.tag.split("}")[0][1:]}
        out += [c.find("s:Prefix", ns).text for c in r.findall("s:CommonPrefixes", ns)]
        if r.findtext("s:IsTruncated", namespaces=ns) != "true":
            return out
        tok = r.findtext("s:NextContinuationToken", namespaces=ns)


def segment_info(seg):
    sub = [s.split("/")[-2] for s in ls(seg)]
    sv = [s.split("/")[-2] for s in ls(seg + "surface-volumes/")] if "surface-volumes" in sub else []
    return {"labels": "ink-labels" in sub, "detection": "ink-detection" in sub, "surface_volumes": sv}


def main():
    samples = [s for s in ls("") if s.startswith("PHerc") and not s.startswith("PHerc1667")]
    for s in samples:
        segs = ls(s + "segments/")
        if not segs:
            continue
        with ThreadPoolExecutor(16) as pool:
            infos = list(pool.map(segment_info, segs[:60]))
        lab = sum(i["labels"] for i in infos)
        det = sum(i["detection"] for i in infos)
        res = sorted({v.split("-")[0] for i in infos for v in i["surface_volumes"]})
        if lab or det:
            print(f"{s:20s} segments {len(segs):3d} (checked {len(infos)}): with ink labels {lab}, with ink detection {det}; "
                  f"surface-volume resolutions {res}", flush=True)
        else:
            print(f"{s:20s} segments {len(segs):3d}: no labels / detections; resolutions {res}", flush=True)


if __name__ == "__main__":
    main()
