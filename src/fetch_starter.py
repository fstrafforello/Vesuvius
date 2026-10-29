"""Download a small, self-contained starter set for one Scroll 4 (PHerc. 1667) segment.

Everything goes to data/PHerc1667/w013 in the repository. Re-running skips what is already there.

What we fetch (segment w013, rendered on the 2.399um / 78keV scan):
  mesh/              tifxyz surface: x.tif, y.tif, z.tif give the 3D position of every grid point
  labels/            human ink labels + supervision mask (where labels are trustworthy), full res
  preds/             downsampled (1/8) ink-detection predictions from the community models, as JPG
  surface_l4.zarr    the whole flattened segment, 109 layers deep, at 1/16 resolution  (~0.5-1 GB)
  surface_crop.zarr  a full-resolution 109 x 4096 x 4096 tile where the ink labels are densest
  ct_crop.zarr       a 512^3 cube of the raw CT scan around that tile, to see the papyrus in 3D
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import requests
import tensorstore as ts
import tifffile

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SCROLL = "PHerc1667"
SEG = f"{SCROLL}/segments/20240304141531-w013_20240304141531_flatboi"
SCAN_ID = "20251217075048"
MESH = f"{SEG}/mesh/20240304141531-on-{SCAN_ID}-2.399um.tifxyz"
LABELS = f"{SEG}/ink-labels/2.399um-volume-{SCAN_ID}/20260918"
SURFACE = f"{SEG}/surface-volumes/2.399um-0.22m-78keV-volume-{SCAN_ID}.zarr"
PREDS = f"{SEG}/ink-detection/downsampled"
CT = f"{SCROLL}/volumes/{SCAN_ID}-2.399um-0.2m-78keV-masked.zarr"

OUT = Path(__file__).resolve().parents[1] / "data" / SCROLL / "w013"
TILE = 4096   # full-res crop edge, in surface pixels
CUBE = 512    # raw CT cube edge, in voxels

session = requests.Session()
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=requests.adapters.Retry(
    total=8, connect=8, read=8, backoff_factor=2, status_forcelist=(500, 502, 503, 504))))


def list_keys(prefix):
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        root = ElementTree.fromstring(session.get(BUCKET + "/", params=params, timeout=60).content)
        ns = {"s": root.tag.split("}")[0].strip("{")}
        keys += [(c.find("s:Key", ns).text, int(c.find("s:Size", ns).text)) for c in root.findall("s:Contents", ns)]
        if root.findtext("s:IsTruncated", namespaces=ns) != "true":
            return keys
        token = root.findtext("s:NextContinuationToken", namespaces=ns)


def mirror(prefix, dest):
    """Copy every object under an S3 prefix to dest, keeping relative paths."""
    jobs = []
    for key, size in list_keys(prefix.rstrip("/") + "/"):
        if key.endswith("/"):
            continue
        path = dest / key[len(prefix.rstrip("/")) + 1:]
        if not (path.exists() and path.stat().st_size == size):
            jobs.append((key, path))

    def get(job):
        key, path = job
        path.parent.mkdir(parents=True, exist_ok=True)
        r = session.get(f"{BUCKET}/{key}", timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(get, jobs))
    print(f"  {dest.parent.name}/{dest.name}: {len(jobs)} new files")


def remote(path):
    """Open a remote array, whether it is stored as zarr v2 or v3."""
    for driver in ("zarr", "zarr3"):
        try:
            return ts.open({"driver": driver, "kvstore": {"driver": "http", "base_url": f"{BUCKET}/{path}"}}, read=True).result()
        except ValueError:
            continue
    raise RuntimeError(f"cannot open {path}")


def save_local(arr, name, chunks):
    path = OUT / name
    if path.exists():
        print(f"  {name}: exists, skipped")
        return
    dst = ts.open({
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(path)},
        "metadata": {"shape": list(arr.shape), "dtype": "|u1", "chunks": list(chunks),
                     "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}},
        "create": True,
    }).result()
    dst.write(arr).result()
    print(f"  {name}: {arr.shape} written")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("1/6 mesh, labels, predictions")
    mirror(MESH, OUT / "mesh")
    mirror(LABELS, OUT / "labels")
    mirror(PREDS, OUT / "preds")

    print("2/6 surface volume overview (level 4 = 1/16 resolution)")
    save_local(remote(f"{SURFACE}/4").read().result(), "surface_l4.zarr", (109, 128, 128))

    print("3/6 choosing the full-res tile with the densest ink labels")
    labels_l4 = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(OUT / "labels/inklabels.zarr/4")}}, read=True).result().read().result()
    k = TILE // 16
    ink = (labels_l4 > 0).astype(np.int64)
    integral = np.pad(ink.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    window = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    wy, wx = np.unravel_index(window[::8, ::8].argmax(), window[::8, ::8].shape)
    y0, x0 = int(wy * 8 * 16), int(wx * 8 * 16)
    print(f"  tile at y={y0} x={x0} (level-0 surface pixels), {window[wy * 8, wx * 8] / k**2:.1%} labelled ink")

    print(f"4/6 full-resolution surface tile 109 x {TILE} x {TILE} (largest download)")
    save_local(remote(f"{SURFACE}/0")[:, y0:y0 + TILE, x0:x0 + TILE].read().result(), "surface_crop.zarr", (109, 256, 256))

    print("5/6 locating the tile in 3D via the mesh")
    # tifxyz stores 3D coordinates on a grid 20x coarser than the surface image (scale 0.05)
    xs, ys, zs = (tifffile.imread(OUT / "mesh" / f"{a}.tif") for a in "xyz")
    gy, gx = (y0 + TILE // 2) // 20, (x0 + TILE // 2) // 20
    cx, cy, cz = (float(a[gy, gx]) for a in (xs, ys, zs))
    if not all(np.isfinite([cx, cy, cz])) or cx <= 0:
        sys.exit("mesh has no valid point at the tile centre; pick another tile")
    print(f"  tile centre sits at CT voxel x={cx:.0f} y={cy:.0f} z={cz:.0f}")

    print(f"6/6 raw CT cube {CUBE}^3 around that point")
    ct = remote(f"{CT}/0")
    h = CUBE // 2
    lo = [max(0, int(c) - h) for c in (cz, cy, cx)]
    save_local(ct[lo[0]:lo[0] + CUBE, lo[1]:lo[1] + CUBE, lo[2]:lo[2] + CUBE].read().result(), "ct_crop.zarr", (128, 128, 128))

    (OUT / "manifest.json").write_text(json.dumps({
        "scroll": SCROLL, "segment": SEG, "scan": CT,
        "surface_tile_yx": [y0, x0], "tile_size": TILE,
        "tile_centre_ct_xyz": [cx, cy, cz], "ct_crop_origin_zyx": lo, "ct_crop_size": CUBE,
        "license": "Scroll 4 (PHerc. 1667) - Vesuvius Challenge / EduceLab data terms, non-commercial; see https://scrollprize.org/data",
    }, indent=2))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
