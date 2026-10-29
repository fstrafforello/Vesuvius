"""3D surface growing: grow one papyrus sheet as a 2D grid of points in 3D, from a seed.

Unlike the slice-by-slice tracer, the surface grows in ANY direction (along the sheet and along
the scroll axis), one grid point at a time, most confident points first:
  1. predict a new grid point from its accepted neighbours (smooth continuation)
  2. snap it at most R voxels across the sheet (along the local surface normal) onto the
     brightness ridge = centre of the papyrus
  3. accept only if: it lands on papyrus; distances to its grid neighbours stay close to the grid
     spacing (no stretching/folding); the surface normal stays consistent with its neighbours;
     and it does not come close to a non-neighbouring part of the surface (no overlap).
A faint spot simply is not accepted; the front grows around it from other sides instead of
bridging across - which is where the slice-by-slice tracer jumped between sheets.

Works on the level-1 tracing block (trace_L1.zarr). Output: data/scout/PHerc0846A/grow_<name>/
with tifxyz (level-0 coordinates), the grid in block coordinates, and check figures.

Usage:  uv run python src/surface/grow3d.py --name g1 [--seed 856 797 1476] [--max-points 200000]
"""

import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from trace import D, load  # noqa: E402

H = 4.0            # grid spacing, level-1 voxels (19 um)
R = 2.0            # max snap across the sheet, voxels (~10 um): less than the gap to a neighbour sheet
THR = 90.0         # papyrus brightness at the ridge
MAX_ANGLE = 25.0   # degrees a point's normal may differ from its neighbours'
ACROSS = 1.5       # voxels a point may sit across the sheet from a direct neighbour
U_MAX, V_MAX = 600, 400   # grid half-size (u along the sheet, v roughly along the scroll axis)
NEIGH4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Volume:
    def __init__(self, arr):
        self.a = arr
        self.shape = np.array(arr.shape)

    def sample(self, pts):
        return ndimage.map_coordinates(self.a, pts.T, order=1, mode="constant", cval=0.0)

    def inside(self, p, margin=6):
        return np.all(p >= margin) and np.all(p < self.shape - margin)


def structure_normal(vol, p, half=6):
    """Sheet normal at p from the 3D structure tensor of a small window (largest-eigenvalue vector)."""
    lo = np.maximum(np.round(p).astype(int) - half, 0)
    hi = np.minimum(np.round(p).astype(int) + half + 1, vol.shape)
    w = vol.a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    g = np.stack(np.gradient(ndimage.gaussian_filter(w, 1.0)), -1).reshape(-1, 3)
    vals, vecs = np.linalg.eigh(g.T @ g)
    return vecs[:, 2]


def snap(vol, p, n):
    """Move p along n (unit) to the brightness ridge within +-R; return (point, value, |shift|)."""
    t = np.arange(-R, R + 0.01, 0.5)
    prof = vol.sample(p[None] + t[:, None] * n[None])
    prof = np.convolve(prof, [0.25, 0.5, 0.25], mode="same")
    k = int(np.argmax(prof))
    return p + t[k] * n, float(prof[k]), abs(t[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="g1")
    ap.add_argument("--seed", type=float, nargs=3, default=[856, 797, 1476], help="z y x in level-1 block coords")
    ap.add_argument("--max-points", type=int, default=200000)
    ap.add_argument("--across", type=float, default=ACROSS, help="max voxels across the sheet between neighbours")
    ap.add_argument("--region", default=None, help="named region (tracing block trace_L1_<region>.zarr)")
    a = ap.parse_args()
    globals()["ACROSS"] = a.across

    block, region, _ = load(1, a.region)
    t0 = time.time()
    vol = Volume(block.read().result())
    print(f"loaded block {vol.shape} in {time.time() - t0:.0f}s", flush=True)

    G = 2 * U_MAX + 1, 2 * V_MAX + 1
    P = np.full(G + (3,), np.nan)
    N = np.full(G + (3,), np.nan)
    acc = np.zeros(G, bool)
    tries = np.zeros(G, np.int8)
    u0, v0 = U_MAX, V_MAX
    occupied = {}                                   # spatial hash: voxel bin -> list of (u, v)

    def binkey(p):
        return tuple((p // H).astype(int))

    def accept(u, v, p, n):
        P[u, v], N[u, v], acc[u, v] = p, n, True
        occupied.setdefault(binkey(p), []).append((u, v))

    # seed frame: normal from the data, v-direction = scroll axis (z) projected into the sheet
    p0 = np.array(a.seed, float)
    n0 = structure_normal(vol, p0, half=12)                    # larger window: steadier seed normal
    p0, val0, _ = snap(vol, p0, n0)
    n0 = structure_normal(vol, p0, half=12)
    ev = np.array([1.0, 0, 0]) - n0 * n0[0]; ev /= np.linalg.norm(ev)
    eu = np.cross(n0, ev); eu /= np.linalg.norm(eu)
    print(f"seed {p0.round(1)} brightness {val0:.0f}, normal {n0.round(2)}", flush=True)
    accept(u0, v0, p0, n0)

    heap = []

    def push(u, v):
        if 0 <= u < G[0] and 0 <= v < G[1] and not acc[u, v] and tries[u, v] < 4:
            nb = sum(acc[u + du, v + dv] for du, dv in NEIGH4 if 0 <= u + du < G[0] and 0 <= v + dv < G[1])
            heapq.heappush(heap, (-nb, abs(u - u0) + abs(v - v0), u, v))

    # first ring: plain extrapolation along the seed frame
    for du, dv in NEIGH4:
        push(u0 + du, v0 + dv)

    def predict(u, v):
        """Affine fit p = c + a*du + b*dv through the accepted points in the 5x5 neighbourhood."""
        ul, uh, vl, vh = max(u - 2, 0), min(u + 3, G[0]), max(v - 2, 0), min(v + 3, G[1])
        m = acc[ul:uh, vl:vh]
        if m.sum() >= 5:
            du_, dv_ = np.nonzero(m)
            if len(np.unique(du_)) >= 2 and len(np.unique(dv_)) >= 2:   # points spread both ways
                X = np.stack([np.ones(len(du_)), du_ + ul - u, dv_ + vl - v], 1).astype(float)
                Y = P[ul:uh, vl:vh][m]
                coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
                # the normal used for checks comes from the neighbours' own normals (stable),
                # the fit only provides the predicted position
                nn = N[ul:uh, vl:vh][m].mean(0)
                return coef[0], nn / np.linalg.norm(nn)
        return predict_simple(u, v)

    def predict_simple(u, v):
        preds, normals = [], []
        for du, dv in NEIGH4:
            u1, v1, u2, v2 = u - du, v - dv, u - 2 * du, v - 2 * dv
            if 0 <= u1 < G[0] and 0 <= v1 < G[1] and acc[u1, v1]:
                normals.append(N[u1, v1])
                if 0 <= u2 < G[0] and 0 <= v2 < G[1] and acc[u2, v2]:
                    preds.append(2 * P[u1, v1] - P[u2, v2])
                else:                                # one neighbour only: step along the local frame
                    n = N[u1, v1]
                    e_v = np.array([1.0, 0, 0]) - n * n[0]; e_v /= np.linalg.norm(e_v)
                    e_u = np.cross(n, e_v); e_u /= np.linalg.norm(e_u)
                    if np.dot(e_u, eu) < 0:
                        e_u = -e_u
                    preds.append(P[u1, v1] + H * (du * e_u + dv * e_v))
        if not preds:
            return None, None
        n = np.mean(normals, 0)
        return np.mean(preds, 0), n / np.linalg.norm(n)

    rejected = {"dark": 0, "stretch": 0, "bend": 0, "overlap": 0, "edge": 0}
    while heap and acc.sum() < a.max_points:
        _, _, u, v = heapq.heappop(heap)
        if acc[u, v]:
            continue
        nb = int(acc[max(u - 1, 0):u + 2, max(v - 1, 0):v + 2].sum())   # of the 8 surrounding cells
        if nb < 3 and abs(u - u0) + abs(v - v0) > 2:      # grow only where >= 3 neighbours vouch
            rejected["few_nb"] = rejected.get("few_nb", 0) + 1
            continue
        pred, n = predict(u, v)
        if pred is None:
            rejected["no_pred"] = rejected.get("no_pred", 0) + 1
            continue
        tries[u, v] += 1
        if not vol.inside(pred):
            rejected["edge"] += 1; continue
        # refine the normal from the data, but keep it consistent with the surface around it
        nd = structure_normal(vol, pred)
        if np.dot(nd, n) < 0:
            nd = -nd
        first_ring = abs(u - u0) + abs(v - v0) <= 1             # only the seed to compare against:
        if not first_ring and np.dot(nd, n) < np.cos(np.radians(MAX_ANGLE)):   # skip bend/across there
            rejected["bend"] += 1; continue
        p, val, shift = snap(vol, pred, nd)
        if val < THR or shift >= R - 0.25:
            rejected["dark"] += 1; continue
        ok, why = True, ""
        # average normal of the accepted direct neighbours: steadier than any single one
        nbn = [N[u + du, v + dv] for du, dv in NEIGH4 if 0 <= u + du < G[0] and 0 <= v + dv < G[1] and acc[u + du, v + dv]]
        navg = np.mean([x if np.dot(x, nd) >= 0 else -x for x in nbn], 0) if nbn else nd.copy()
        navg /= max(np.linalg.norm(navg), 1e-9)
        for du, dv in NEIGH4:                        # grid must not stretch or fold
            uu, vv = u + du, v + dv
            if 0 <= uu < G[0] and 0 <= vv < G[1] and acc[uu, vv]:
                d = np.linalg.norm(p - P[uu, vv])
                if not (0.6 * H <= d <= 1.5 * H):
                    ok, why = False, "stretch"; break
                # one sheet cannot move more than ACROSS voxels sideways between neighbours 4 voxels
                # apart; more means the point landed on an adjacent sheet
                if not first_ring and abs(np.dot(p - P[uu, vv], navg)) > ACROSS:
                    ok, why = False, "across"; break
        if not ok:
            rejected[why] = rejected.get(why, 0) + 1; continue
        b = binkey(p)                                # must not touch a non-neighbouring part of itself
        clash = False
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for (uu, vv) in occupied.get((b[0] + dz, b[1] + dy, b[2] + dx), []):
                        if abs(uu - u) + abs(vv - v) > 3 and np.linalg.norm(P[uu, vv] - p) < 0.8 * H:
                            clash = True
        if clash:
            rejected["overlap"] += 1; continue
        # store a smoothed normal (data normal blended with the neighbours' average)
        ns = nd + navg
        accept(u, v, p, ns / max(np.linalg.norm(ns), 1e-9))
        # once the first ring is complete, re-derive the seed normal from the ring's actual geometry
        if first_ring and all(acc[u0 + du, v0 + dv] for du, dv in NEIGH4):
            ng = np.cross(P[u0 + 1, v0] - P[u0 - 1, v0], P[u0, v0 + 1] - P[u0, v0 - 1])
            ng /= max(np.linalg.norm(ng), 1e-9)
            if np.dot(ng, N[u0, v0]) < 0:
                ng = -ng
            for du, dv in ((0, 0),) + NEIGH4:
                N[u0 + du, v0 + dv] = ng
        for du, dv in NEIGH4 + ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            push(u + du, v + dv)
        if acc.sum() % 20000 == 0:
            print(f"  {acc.sum()} points ({acc.sum() * (H * region['um_per_voxel'] / 1000) ** 2:.1f} mm^2), "
                  f"rejected {rejected}, {time.time() - t0:.0f}s", flush=True)

    mm2 = acc.sum() * (H * region["um_per_voxel"] / 1000) ** 2
    print(f"grown: {acc.sum()} points = {mm2:.1f} mm^2; rejected {rejected}; {time.time() - t0:.0f}s", flush=True)
    # hop check: between grid neighbours the offset ACROSS the sheet must stay small
    across = []
    for axis in (0, 1):
        a1 = acc[1:] & acc[:-1] if axis == 0 else acc[:, 1:] & acc[:, :-1]
        dp = (P[1:] - P[:-1]) if axis == 0 else (P[:, 1:] - P[:, :-1])
        nn = (N[1:] + N[:-1]) if axis == 0 else (N[:, 1:] + N[:, :-1])
        nn /= np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-9)
        across.append(np.abs(np.sum(dp * nn, -1))[a1])
    across = np.concatenate(across)
    print(f"hop check: across-sheet step between neighbours median {np.median(across):.2f} voxels, "
          f"{np.mean(across > 2.0):.2%} of neighbour pairs > 2 voxels (a hop)", flush=True)

    # crop and save: tifxyz in level-0 coordinates (block + region offset, x2), grid spacing H level-1
    us, vs = np.nonzero(acc.any(1))[0], np.nonzero(acc.any(0))[0]
    P, acc, N = P[us[0]:us[-1] + 1, vs[0]:vs[-1] + 1], acc[us[0]:us[-1] + 1, vs[0]:vs[-1] + 1], N[us[0]:us[-1] + 1, vs[0]:vs[-1] + 1]
    # tifxyz rows = v (along the scroll axis), cols = u (along the sheet)
    Pt, At = np.transpose(P, (1, 0, 2)), acc.T
    f = 2 ** region["level"]
    out = D / f"grow_{a.name}"
    out.mkdir(exist_ok=True)
    off = np.array([region["z0"], region["y0"], region["x0"]], float)
    L0 = (Pt + off) * f
    for nm, ci in (("z", 0), ("y", 1), ("x", 2)):
        tifffile.imwrite(out / f"{nm}.tif", np.where(At, L0[..., ci], -1).astype(np.float32))
    (out / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [1 / (H * f)] * 2, "type": "seg",
                                               "uuid": f"psn_grow_{a.name}", "source": "psn grow3d.py", "level": 1,
                                               "region_tag": a.region,
                                               "seed_block_zyx": list(a.seed), "grid_spacing_level1": H}, indent=2))
    np.savez_compressed(out / "grid_block_coords.npz", P=Pt, valid=At, N=np.transpose(N, (1, 0, 2)))

    # checks: (1) the surface's crossings with a few slices; (2) the grid seen in 3D extent
    zs = Pt[..., 0][At]
    fig, ax = plt.subplots(2, 3, figsize=(24, 13))
    for a_, f_ in zip(ax.flat, np.linspace(0.05, 0.95, 6)):
        z = int(np.percentile(zs, 100 * f_))
        near = At & (np.abs(Pt[..., 0] - z) < H / 2)
        im = vol.a[z]
        a_.imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im, 99.5))
        pts = Pt[near]
        a_.plot(pts[:, 2], pts[:, 1], "r.", ms=2)
        if len(pts):
            cy, cx = pts[:, 1].mean(), pts[:, 2].mean()
            h = max(np.ptp(pts[:, 1]), np.ptp(pts[:, 2])) / 2 + 80
            a_.set_xlim(cx - h, cx + h); a_.set_ylim(cy + h, cy - h)
        a_.set_title(f"slice {z} ({(z + region['z0']) * region['um_per_voxel'] / 1000:.2f} mm): {near.sum()} surface points")
    plt.tight_layout(); plt.savefig(out / "slices_check.png", dpi=60); plt.close()
    plt.figure(figsize=(10, 6)); plt.imshow(At, aspect="auto", cmap="gray")
    plt.title(f"grown grid (rows = along scroll axis, cols = along sheet), {mm2:.1f} mm^2"); plt.tight_layout()
    plt.savefig(out / "grid.png", dpi=60); plt.close()
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
