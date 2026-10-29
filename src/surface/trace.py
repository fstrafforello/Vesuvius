"""Step d2: trace one papyrus layer through the level-2 block.

Method (layers here run nearly straight along the scroll axis, so this is robust):
  1. In the start slice, follow the sheet as a 2D curve from a seed point in both directions:
     step along the sheet, then snap across it to the sheet centre (intensity ridge).
  2. Carry the curve slice by slice up and down the scroll: every point moves only across the
     sheet (along the curve normal) to the ridge in the new slice, then the curve is lightly
     smoothed. Points keep their index, so the result is a grid: rows = height, cols = position
     along the sheet - the same kind of grid as the official tifxyz meshes.
  3. Points stop (become invalid) where the sheet fades, the snap jumps too far, or the curve
     folds - these are where the trace would otherwise jump to a neighbouring layer.

Output: data/scout/PHerc0846A/trace_<name>/ with x.tif, y.tif, z.tif in LEVEL-0 voxel
coordinates of the full scan (so our renderer can use it directly), meta.json, and checks.

Usage:  uv run python src/surface/trace.py --seed-y Y --seed-x X [--z MID] [--half-len 520]
        (seed in level-2 block coordinates; default seed = automatic, left of the centre)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts
import tifffile
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "scout" / "PHerc0846A"
STEP = 1.0          # spacing of curve points along the sheet, level-2 voxels (= 4 level-0 voxels)
SNAP = 2            # max snap across the sheet per step/slice, level-2 voxels
SMOOTH = 2.0        # curve smoothing (points) after each slice
VERBOSE = False


def block_files(level, tag=None):
    """Tracing block and its region file for a pyramid level (2 = 9.6 um, 1 = 4.8 um).
    tag selects a named region (e.g. 'top'); None is the original region."""
    if tag:
        return D / f"trace_L{level}_{tag}.zarr", D / f"region_L{level}_{tag}.json"
    return (D / f"trace_L{level}.zarr", D / ("region.json" if level == 2 else f"region_L{level}.json"))


def trace_level(trace_dir):
    """Level a saved trace was made at (older traces have no field: level 2)."""
    return json.loads((Path(trace_dir) / "meta.json").read_text()).get("level", 2)


def trace_tag(trace_dir):
    """Named region a saved trace/surface belongs to (None = original region)."""
    return json.loads((Path(trace_dir) / "meta.json").read_text()).get("region_tag")


def load(level=2, tag=None):
    zarr_path, region_path = block_files(level, tag)
    block = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(zarr_path)}}, read=True).result()
    region = json.loads(region_path.read_text())
    region.setdefault("level", level)
    region["tag"] = tag
    thr = json.loads((D / "sheet_field.json").read_text())["threshold"] if (D / "sheet_field.json").exists() else 60
    return block, region, thr


def ridge_snap(img, p, n, r=SNAP):
    """Move p along n (both 2D, (y, x)) to the brightest point within +-r; return new p and value."""
    ts_ = np.arange(-r, r + 0.01, 0.5)
    pts = p[None] + ts_[:, None] * n[None]
    v = ndimage.map_coordinates(img, pts.T, order=1, mode="nearest")
    k = int(np.argmax(v))
    # sub-voxel peak: parabola through the max and its neighbours
    if 0 < k < len(v) - 1:
        a, b, c = v[k - 1], v[k], v[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        t = ts_[k] + np.clip(d, -0.5, 0.5) * 0.5
    else:
        t = ts_[k]
    return p + t * n, float(v[k]), abs(t)


def normals(curve):
    """Unit normals (y, x) of a 2D polyline, from central-difference tangents."""
    t = np.gradient(curve, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
    return np.stack([-t[:, 1], t[:, 0]], 1), t


def orientation_field(img, sigma=3.0):
    """Per-pixel sheet tangent angle in a slice, from the 2D structure tensor."""
    gy = ndimage.gaussian_filter(img, 1.0, order=(1, 0))
    gx = ndimage.gaussian_filter(img, 1.0, order=(0, 1))
    jyy = ndimage.gaussian_filter(gy * gy, sigma)
    jxx = ndimage.gaussian_filter(gx * gx, sigma)
    jyx = ndimage.gaussian_filter(gy * gx, sigma)
    # doubled-angle components, so they can be interpolated without the 180-degree ambiguity
    return np.stack([jyy - jxx, 2 * jyx])


def local_tangent(field, p, prev):
    c = np.array([ndimage.map_coordinates(field[i], p[:, None], order=1, mode="nearest")[0] for i in (0, 1)])
    theta = 0.5 * np.arctan2(c[1], c[0])                    # direction of strongest gradient = sheet normal
    t = np.array([-np.sin(theta), np.cos(theta)])           # perpendicular to it = along the sheet
    return t if np.dot(t, prev) >= 0 else -t


def find_seed(img, field, thr, zone, n_try=300, half_len=400, rng=0):
    """Try many bright points in zone=(y0, y1, x0, x1), trace each, return the longest curve."""
    r = np.random.default_rng(rng)
    y0, y1, x0, x1 = zone
    cand = np.argwhere(img[y0:y1, x0:x1] > thr + 30) + [y0, x0]
    best = None
    for p in cand[r.choice(len(cand), min(n_try, len(cand)), replace=False)]:
        p = p.astype(np.float64)
        t = local_tangent(field, p, np.array([0.0, 1.0]))
        c = trace_slice(img, p, t, half_len, thr, field=field)
        if best is None or len(c) > len(best[1]):
            best = (p, c)
    return best


def sheet_profile(img, p, t, half=4):
    """Brightness profile across the sheet at p (normal to tangent t), mean-removed and scaled."""
    n = np.array([-t[1], t[0]])
    s = np.arange(-half, half + 0.01, 1.0)
    v = ndimage.map_coordinates(img, (p[None] + s[:, None] * n[None]).T, order=1, mode="nearest")
    v = v - v.mean()
    return v / max(np.linalg.norm(v), 1e-6)


def bridge_ok(img, field, p_before, t_before, prof_before, p_after, half, max_angle=20, min_corr=0.6):
    """A bridge across a break is only accepted if the sheet found on the far side runs the same way
    and has the same cross-section (thickness / fibre layers) as before the break."""
    t_after = local_tangent(field, p_after, t_before)
    if np.dot(t_after, t_before) < np.cos(np.radians(max_angle)):
        return False
    return float(np.dot(sheet_profile(img, p_after, t_after, half), prof_before)) >= min_corr


SAFE_BRIDGE = True    # verify direction + cross-section on the far side of every bridge
PROFILE_HALF = 4      # voxels each side for the cross-section (set per level by callers)


def trace_slice(img, seed, direction, half_len, thr, step=1.5, max_turn_deg=30, field=None, bridge=6):
    """Follow the sheet from seed both ways (half_len level-2 voxels each way): at every step,
    take the local sheet direction from the structure tensor, move along it, and snap across the
    sheet to its centre. Stops where the sheet fades or the direction turns sharply (a likely jump
    to another sheet). Returns the curve resampled to STEP spacing."""
    field = orientation_field(img) if field is None else field
    cos_max = np.cos(np.radians(max_turn_deg))
    out = {}
    for sgn in (1, -1):
        p, t = seed.astype(np.float64), sgn * direction / np.linalg.norm(direction)
        pts, low = [], 0
        for _ in range(int(half_len / step)):
            # stop at the edge of the data: beyond it the image only repeats its border values
            if not (3 <= p[0] < img.shape[0] - 3 and 3 <= p[1] < img.shape[1] - 3):
                if low:
                    pts = pts[:-low]
                low = 0
                break
            if low == 0:                                    # inside a bridge, keep the direction we had
                tl = local_tangent(field, p, t)
                if np.dot(tl, t) < cos_max:
                    break
                t = tl
            q, val, _ = ridge_snap(img, p, np.array([-t[1], t[0]]), r=1.5)
            if val < thr:
                # possible small break in the sheet: carry straight on for up to `bridge` steps and
                # only keep going if the sheet is found again on the other side
                if low == 0 and pts:                        # remember the sheet just before the break
                    pre = (pts[-1].copy(), t.copy(), sheet_profile(img, pts[-1], t, PROFILE_HALF))
                low += 1
                if low > bridge:
                    pts = pts[:-low] if low < len(pts) else []
                    break
                pts.append(p.copy()); p = p + step * t
                continue
            if low and SAFE_BRIDGE and pts and not bridge_ok(img, field, pre[0], pre[1], pre[2], q, PROFILE_HALF):
                pts = pts[:-low]                             # the far side is a different sheet: stop here
                low = 0
                break
            low = 0
            p = q
            pts.append(p.copy())
            p = p + step * t
        else:
            if low:
                pts = pts[:-low]
        out[sgn] = pts
    pts = out[-1][::-1][:-1] + out[1] if out[-1] else out[1]
    curve = np.array(pts)
    if len(curve) < 2:
        return curve.reshape(-1, 2)
    # resample to even STEP spacing along the arc
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    si = np.arange(0, s[-1], STEP)
    return np.stack([np.interp(si, s, curve[:, 0]), np.interp(si, s, curve[:, 1])], 1)


def alignment(img, pts, ok, pad=24):
    """|cos| of the angle between the curve direction and the image's sheet direction, per point."""
    lo = np.maximum(np.floor(pts[ok].min(0)).astype(int) - pad, 0)
    hi = np.minimum(np.ceil(pts[ok].max(0)).astype(int) + pad, img.shape)
    field = orientation_field(img[lo[0]:hi[0], lo[1]:hi[1]])
    _, t = normals(pts)
    out = np.ones(len(pts))
    for k in np.nonzero(ok)[0]:
        tl = local_tangent(field, pts[k] - lo, t[k])
        out[k] = abs(np.dot(tl, t[k]))
    return out


def regrow(img, pts, ok, thr, max_turn_deg=30, pad=24):
    """Refill lost indices in this slice by tracing outward along the sheet from the ends of the
    valid stretches (in place). Internal gaps are only accepted if the regrown piece arrives next
    to the other side; outer ends are extended as far as the sheet goes."""
    valid_idx = np.nonzero(ok)[0]
    lo = np.floor(pts[ok].min(0)).astype(int) - pad
    hi = np.ceil(pts[ok].max(0)).astype(int) + pad
    lo = np.maximum(lo, 0); hi = np.minimum(hi, img.shape)
    sub = img[lo[0]:hi[0], lo[1]:hi[1]]
    field = orientation_field(sub)
    cos_max = np.cos(np.radians(max_turn_deg))
    n = len(pts)

    def walk(start, end_idx, step_dir):
        """From valid index `start`, walk toward end_idx (exclusive) one index per STEP."""
        prev =pts[start] - pts[start - step_dir] if 0 <= start - step_dir < n and ok[start - step_dir] else None
        t = prev / max(np.linalg.norm(prev), 1e-6) if prev is not None else None
        p = pts[start] - lo
        out, low = [], 0
        k = start + step_dir
        while k != end_idx:
            if t is None:
                t = local_tangent(field, p, np.array([1.0, 0.0]))
            tl = local_tangent(field, p, t)
            if np.dot(tl, t) < cos_max:
                break
            t = tl
            q, val, _ = ridge_snap(sub, p + STEP * t, np.array([-t[1], t[0]]), r=1.5)
            # stop at the edge of the cropped data (in image coordinates: lo + q)
            if not (3 <= q[0] + lo[0] < img.shape[0] - 3 and 3 <= q[1] + lo[1] < img.shape[1] - 3):
                break
            low = low + 1 if val < thr else 0
            if low >= 3:
                out = out[:-2]
                break
            out.append(q.copy()); p = q
            k += step_dir
        return [o + lo for o in out]

    # runs of invalid indices
    lab, nlab = ndimage.label(~ok)
    for r in range(1, nlab + 1):
        gap = np.nonzero(lab == r)[0]
        g0, g1 = gap[0], gap[-1]
        left_ok, right_ok = g0 - 1 >= 0 and ok[g0 - 1], g1 + 1 < n and ok[g1 + 1]
        if left_ok and right_ok:                              # internal gap: bridge must land next to the far side
            new = walk(g0 - 1, g1 + 1, 1)
            if len(new) == len(gap) and np.linalg.norm(pts[g1 + 1] - new[-1]) < 2.5 * STEP:
                pts[gap] = new; ok[gap] = True
        elif left_ok:                                         # outer end on the right
            new = walk(g0 - 1, g1 + 1, 1)
            if new:
                pts[g0:g0 + len(new)] = new; ok[g0:g0 + len(new)] = True
        elif right_ok:                                        # outer end on the left
            new = walk(g1 + 1, g0 - 1, -1)
            if new:
                pts[g1 - len(new) + 1:g1 + 1] = new[::-1]; ok[g1 - len(new) + 1:g1 + 1] = True


def propagate(block, curve0, z0, z_range, thr):
    """Carry the start curve through slices z0 +- range. Returns grid (nz, npts, 2) and valid mask."""
    nz = len(z_range)
    grid = np.full((nz, len(curve0), 2), np.nan)
    valid = np.zeros((nz, len(curve0)), bool)
    i0 = list(z_range).index(z0)
    grid[i0], valid[i0] = curve0, True
    for direction in (1, -1):
        cur, ok = curve0.copy(), np.ones(len(curve0), bool)
        streak = np.zeros(len(curve0), int)                 # consecutive slices a point has missed
        i = i0 + direction
        while 0 <= i < nz and ok.any():
            img = ndimage.gaussian_filter(block[z_range[i]].read().result().astype(np.float32), 1.0)
            n, _ = normals(cur)
            shift = np.zeros(len(cur))                       # signed move across the sheet
            hit = np.zeros(len(cur), bool)
            for k in np.nonzero(ok)[0]:
                p, val, moved = ridge_snap(img, cur[k], n[k], r=2.0)
                if val >= thr and moved < 1.9:              # found the sheet close by
                    shift[k] = np.dot(p - cur[k], n[k]); hit[k] = True
            # misses borrow the (smoothed) movement of neighbouring hits
            if hit.sum() < 2:
                break
            idx = np.arange(len(cur))
            shift_f = np.interp(idx, idx[hit], shift[hit])
            shift_f = ndimage.gaussian_filter1d(shift_f, SMOOTH, mode="nearest")
            new = cur + shift_f[:, None] * n
            inside = (new[:, 0] >= 3) & (new[:, 0] < img.shape[0] - 3) & (new[:, 1] >= 3) & (new[:, 1] < img.shape[1] - 3)
            ok &= inside                                     # never keep points outside the data
            # judge stretches, not single points: where fewer than half of the points within
            # +-15 (0.3 mm) found the sheet, for more than 3 slices running, the stretch has lost it
            rate = ndimage.uniform_filter1d(hit.astype(np.float32), 31, mode="nearest")
            streak = np.where(rate >= 0.5, 0, streak + 1)
            n_ok0 = ok.sum()
            ok &= streak <= 3
            n_ok1 = ok.sum()
            # jump check: the layer stretches and shrinks gradually with height, so judge the CHANGE
            # in spacing between neighbouring valid points from one slice to the next (a jump to
            # another sheet shows up as a sudden gap), plus a generous cap on total stretch
            both = ok[1:] & ok[:-1]
            gap = np.linalg.norm(np.diff(new, axis=0), axis=1)
            prev_gap = np.linalg.norm(np.diff(cur, axis=0), axis=1)
            stretched = both & ((gap - prev_gap > 2.0) | (gap > 5 * STEP))
            bad = np.zeros_like(ok); bad[1:] |= stretched; bad[:-1] |= stretched
            ok &= ~bad
            n_ok2 = ok.sum()
            # close small holes (<= 6 points) between valid stretches: those points follow their
            # neighbours' movement anyway, and holes would otherwise cut the layer into pieces
            ok = ok | (ndimage.binary_closing(ok, structure=np.ones(7, bool)) & (streak <= 6))
            # keep valid stretches of at least 50 points (0.5 mm); shorter islands are unreliable
            lab, nlab = ndimage.label(ok)
            if nlab:
                sizes = ndimage.sum(ok, lab, range(1, nlab + 1))
                ok = np.isin(lab, np.nonzero(sizes >= 50)[0] + 1)
            n_ok3 = ok.sum()
            if ok.any() and not ok.all():
                regrow(img, new, ok, thr)
                streak[ok] = 0
            # alignment check: the curve must run ALONG the sheets. Where its direction differs from
            # the image's sheet direction by > 35 degrees it is cutting across layers -> invalid
            if ok.sum() > 2:
                aligned = alignment(img, new, ok) > np.cos(np.radians(35))
                bad_al = ok & ~aligned
                # judge over short stretches so single noisy points don't cut the curve
                frac_bad = ndimage.uniform_filter1d(bad_al.astype(np.float32), 9, mode="nearest")
                ok &= frac_bad < 0.5
                lab, nlab = ndimage.label(ok)
                if nlab:
                    sizes = ndimage.sum(ok, lab, range(1, nlab + 1))
                    ok = np.isin(lab, np.nonzero(sizes >= 50)[0] + 1)
            if VERBOSE and abs(i - i0) <= 12:
                print(f"    slice {z_range[i]}: valid {n_ok0} hit {int((hit & (n_ok0 > 0)).sum())} -> after streak/long-miss {n_ok1} "
                      f"-> after fold {n_ok2} -> after short-run {n_ok3} -> after regrow {ok.sum()}", flush=True)
            grid[i][ok], valid[i] = new[ok], ok
            cur = np.where(ok[:, None], new, cur)
            i += direction
    return grid, valid


def save_trace(name, grid, valid, z_range, region, source, extra=None):
    """Write a trace as tifxyz in LEVEL-0 scan coordinates (+ block coordinates for our tools).
    Block (z, y, x) + region offset gives level-L coordinates; x 2**L gives level 0. The grid has
    one point per level-L voxel both ways, i.e. scale 1 / 2**L in level-0 surface pixels."""
    f = 2 ** region["level"]
    out = D / f"trace_{name}"
    out.mkdir(exist_ok=True)
    zz = (np.array(z_range)[:, None] + region["z0"]) * np.ones(grid.shape[1])[None]
    for nm, arr in (("x", (grid[..., 1] + region["x0"]) * f), ("y", (grid[..., 0] + region["y0"]) * f), ("z", zz * f)):
        tifffile.imwrite(out / f"{nm}.tif", np.where(valid, arr, -1).astype(np.float32))
    (out / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [1 / f, 1 / f], "type": "seg",
                                               "uuid": f"psn_{name}", "source": source, "level": region["level"],
                                               "region_tag": region.get("tag"), **(extra or {})}, indent=2))
    np.savez_compressed(out / "trace_block_coords.npz", grid=grid, valid=valid, z=np.array(z_range))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2, help="tracing resolution: 2 = 9.6 um, 1 = 4.8 um")
    ap.add_argument("--z", type=int, default=None, help="start slice in the block (default: middle)")
    ap.add_argument("--seed-y", type=float, default=None)
    ap.add_argument("--seed-x", type=float, default=None)
    ap.add_argument("--half-len", type=int, default=520, help="points each way from the seed (520 = 5 mm)")
    ap.add_argument("--zspan", type=int, default=None, help="slices each way from the start (default: whole block)")
    ap.add_argument("--name", default="first")
    ap.add_argument("--thr", type=float, default=None, help="papyrus/gap grey level (default: from sheet_field.json)")
    ap.add_argument("--max-turn", type=float, default=30, help="largest direction change per step (deg) before stopping")
    ap.add_argument("--bridge", type=int, default=6, help="steps (x1.5 voxels) a trace may cross a faint break in the sheet")
    ap.add_argument("--unsafe-bridge", action="store_true", help="old behaviour: accept any sheet after a bridge")
    ap.add_argument("--zone", type=int, nargs=4, default=[1050, 1300, 450, 1450],
                    help="seed search area y0 y1 x0 x1 in the start slice (block coords)")
    a = ap.parse_args()

    block, region, thr = load(a.level)
    thr = a.thr if a.thr is not None else thr
    nz, ny, nx = block.shape
    global SAFE_BRIDGE, PROFILE_HALF
    SAFE_BRIDGE = not a.unsafe_bridge
    PROFILE_HALF = 2 * 2 ** (2 - a.level)                   # ~19 um each side at any level
    z0 = a.z if a.z is not None else nz // 2
    img = ndimage.gaussian_filter(block[z0].read().result().astype(np.float32), 1.0)
    if a.seed_y is None:                                    # automatic: a bright sheet ~4 mm left of the centre
        sy, sx = ny / 2, nx / 2 - 400
        win = img[int(sy) - 20:int(sy) + 20, int(sx) - 20:int(sx) + 20]
        dy, dx = np.unravel_index(np.argmax(win), win.shape)
        seed = np.array([sy - 20 + dy, sx - 20 + dx])
    else:
        seed = np.array([a.seed_y, a.seed_x])
    # initial direction along the sheet from the structure tensor in this slice
    gy, gx = ndimage.gaussian_filter(img, 1, order=(1, 0)), ndimage.gaussian_filter(img, 1, order=(0, 1))
    sl = (slice(int(seed[0]) - 8, int(seed[0]) + 9), slice(int(seed[1]) - 8, int(seed[1]) + 9))
    J = np.array([[np.sum(gy[sl] ** 2), np.sum(gy[sl] * gx[sl])], [np.sum(gy[sl] * gx[sl]), np.sum(gx[sl] ** 2)]])
    w, v = np.linalg.eigh(J)
    tangent = v[:, 0]                                       # smallest change = along the sheet
    print(f"start slice {z0}, seed (y,x) {seed.round(1)}, threshold {thr}, tangent {tangent.round(2)}", flush=True)

    if a.seed_y is None:
        field = orientation_field(img)
        # trial traces cover the same physical length (~4 mm each way) at every level
        seed, short = find_seed(img, field, thr, tuple(a.zone), half_len=400 * 2 ** (2 - a.level))
        print(f"seed search: best seed (y,x) {seed.round(1)} ({len(short)} pts in the short test trace)", flush=True)
        # the seed search traces only +-400 voxels; redo the trace from the chosen seed at full length
        tangent = local_tangent(field, seed, np.array([0.0, 1.0]))
    curve0 = trace_slice(img, seed, tangent, a.half_len, thr, max_turn_deg=a.max_turn, bridge=a.bridge)
    print(f"start curve: {len(curve0)} points = {len(curve0) * STEP * region['um_per_voxel'] / 1000:.1f} mm along the sheet", flush=True)
    span = a.zspan if a.zspan is not None else nz
    z_range = list(range(max(0, z0 - span), min(nz, z0 + span + 1)))
    grid, valid = propagate(block, curve0, z0, z_range, thr)
    rows = valid.any(1)
    print(f"traced {rows.sum()} slices = {rows.sum() * region['um_per_voxel'] / 1000:.1f} mm of height; "
          f"valid points {valid.mean():.1%} of the grid", flush=True)

    out = save_trace(a.name, grid, valid, z_range, region, "psn trace.py",
                     {"seed_block_zyx": [z0, *seed.tolist()], "threshold": thr})

    # checks: the curve on 3 slices, and the valid mask
    fig, ax = plt.subplots(2, 4, figsize=(24, 13))
    ax = ax.ravel()
    ax[3], ax[7] = ax[7], ax[3]
    for a_, f in zip([ax[0], ax[1], ax[2], ax[4], ax[5], ax[6]], (0.05, 0.25, 0.45, 0.6, 0.8, 0.95)):
        i = int((len(z_range) - 1) * f)
        z = z_range[i]
        im = block[z].read().result()
        pts = grid[i][valid[i]]
        if len(pts):
            cy, cx = pts[:, 0].mean(), pts[:, 1].mean()
            h = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 + 60
        else:
            cy, cx, h = ny / 2, nx / 2, 300
        a_.imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im, 99.5))
        a_.plot(pts[:, 1], pts[:, 0], "r-", lw=0.8)
        a_.set_xlim(cx - h, cx + h); a_.set_ylim(cy + h, cy - h)
        a_.set_title(f"slice {z} ({(z + region['z0']) * region['um_per_voxel'] / 1000:.1f} mm)")
    ax[3].imshow(valid, aspect="auto", cmap="gray"); ax[3].set_title("valid grid (rows = height, cols = along sheet)")
    plt.tight_layout(); plt.savefig(out / "trace_check.png", dpi=70); plt.close()
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
