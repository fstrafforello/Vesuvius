"""Render a surface volume from a tifxyz mesh and a CT volume.

A tifxyz mesh is a grid of 3D points (x.tif, y.tif, z.tif) sampled every 1/scale surface pixels.
For every surface pixel we interpolate its 3D position and the surface normal, then sample the CT
at position + k * normal for each layer offset k (in voxels). This is what turns a curved sheet
inside the scroll into a flat stack of images, one per depth through the papyrus.
"""

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage


def load_tifxyz(folder):
    folder = Path(folder)
    xyz = np.stack([tifffile.imread(folder / f"{a}.tif").astype(np.float32) for a in "xyz"], -1)
    meta = json.loads((folder / "meta.json").read_text())
    valid = np.isfinite(xyz).all(-1) & (xyz[..., 0] > 0)
    return xyz, valid, float(meta["scale"][0])


def grid_normals(xyz, valid):
    """Unit normals on the mesh grid from the cross product of the two in-surface tangents."""
    p = np.where(valid[..., None], xyz, np.nan)
    du = np.gradient(p, axis=1)       # along columns
    dv = np.gradient(p, axis=0)       # along rows
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.nan_to_num(n)


def dense_surface(xyz, normals, valid, scale, rows, cols):
    """Interpolate positions and normals at surface pixels rows x cols (level-0 surface coords)."""
    rr, cc = np.meshgrid(np.asarray(rows, np.float32) * scale, np.asarray(cols, np.float32) * scale, indexing="ij")
    coords = np.stack([rr, cc])
    pos = np.stack([ndimage.map_coordinates(xyz[..., i], coords, order=1, mode="nearest") for i in range(3)], -1)
    nrm = np.stack([ndimage.map_coordinates(normals[..., i], coords, order=1, mode="nearest") for i in range(3)], -1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    ok = ndimage.map_coordinates(valid.astype(np.float32), coords, order=1, mode="nearest") > 0.999
    return pos, nrm, ok


def render(pos, nrm, ok, ct, ct_origin_zyx, offsets, sign=1.0, order=1):
    """Sample ct (z, y, x numpy array starting at ct_origin_zyx) at pos + k * sign * nrm for k in offsets.

    pos/nrm are (H, W, 3) in (x, y, z) order, as stored in tifxyz. Returns (len(offsets), H, W) uint8.
    """
    oz, oy, ox = ct_origin_zyx
    out = np.zeros((len(offsets), *pos.shape[:2]), np.uint8)
    for i, k in enumerate(offsets):
        p = pos + (sign * k) * nrm
        coords = np.stack([p[..., 2] - oz, p[..., 1] - oy, p[..., 0] - ox])
        v = ndimage.map_coordinates(ct, coords, order=order, mode="constant", cval=0)
        out[i] = np.clip(np.rint(v), 0, 255).astype(np.uint8) * ok
    return out
