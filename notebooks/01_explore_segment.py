# %% [markdown]
# # 01 · From a rolled-up scroll to a flat page
#
# Scroll 4 (PHerc. 1667), segment **w013**. Run `python src/fetch_starter.py` first.
#
# **The pipeline in one paragraph.** The scroll is CT-scanned into a 3D volume of X-ray density.
# Inside it the papyrus sheet is wound into a crushed spiral. *Segmentation* finds one patch of that
# sheet and represents it as a **mesh**: a 2D grid where each grid point stores an (x, y, z) position
# inside the scan. Because the mesh is a grid, it already *is* a flattening: grid row/column become
# page row/column. *Rendering* then samples the CT along the surface normal at every grid point,
# producing a **surface volume**: a stack of flat images, one per depth offset through the sheet.
# Ink (carbon, almost the same X-ray density as papyrus) is too faint to see by eye in most scrolls,
# so an ML model looks at the stack and predicts where ink is. That 2D prediction is what gets read.
#
# ```
#  CT volume (z,y,x)  --segmentation-->  mesh: grid[u,v] -> (x,y,z)
#        |                                       |
#        +------- sample along normals ----------+
#                         v
#        surface volume (depth, v, u)  --ink model-->  ink map (v, u)  -->  text
# ```

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts
import tifffile
from PIL import Image

ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
D = ROOT / "data" / "PHerc1667" / "w013"
M = json.loads((D / "manifest.json").read_text())
plt.rcParams.update({"figure.dpi": 110, "image.cmap": "gray"})


def load(name, driver="zarr"):
    return ts.open({"driver": driver, "kvstore": {"driver": "file", "path": str(D / name)}}, read=True).result().read().result()


def stretch(a, lo=1, hi=99):
    p1, p99 = np.percentile(a[a > 0] if (a > 0).any() else a, [lo, hi])
    return np.clip((a.astype(np.float32) - p1) / max(p99 - p1, 1), 0, 1)


M

# %% [markdown]
# ## 1 · The mesh: a sheet of papyrus as a grid of 3D points
# `x.tif`, `y.tif`, `z.tif` are three images of the same shape. Pixel (row, col) of each holds one
# coordinate of the surface point at grid position (row, col). Invalid points are -1.
# The grid is 20x coarser than the rendered surface (meta.json `scale: 0.05`).

# %%
xs, ys, zs = (tifffile.imread(D / "mesh" / f"{a}.tif") for a in "xyz")
valid = (xs > 0) & np.isfinite(xs)
print("grid shape", xs.shape, f"| valid points {valid.mean():.1%}")
print("CT bounding box  x", xs[valid].min(), xs[valid].max(), " y", ys[valid].min(), ys[valid].max(), " z", zs[valid].min(), zs[valid].max())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, arr, name in zip(ax, (xs, ys, zs), "xyz"):
    im = a.imshow(np.where(valid, arr, np.nan), cmap="viridis")
    a.set_title(f"{name}(u, v)")
    plt.colorbar(im, ax=a, shrink=0.7)
fig.suptitle("Each grid cell stores where that bit of papyrus lives in the scan")
plt.show()

# %% [markdown]
# Viewed in 3D the grid is a curved strip. The columns follow the spiral around the scroll axis (z)
# and the rows run along the length of the scroll. Colour = grid column, i.e. position along the page.

# %%
s = 12
P = np.stack([xs[::s, ::s], ys[::s, ::s], zs[::s, ::s]], -1)[valid[::s, ::s]]
U = np.indices(xs.shape)[1][::s, ::s][valid[::s, ::s]]
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection="3d")
ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=U, s=0.3, cmap="turbo")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (scroll axis)")
ax.set_title("Segment w013 in scan space")
plt.show()

# %% [markdown]
# Looking straight down the scroll axis shows the spiral: the sheet wraps around the centre several times.

# %%
plt.figure(figsize=(6, 6))
plt.scatter(P[:, 0], P[:, 1], c=U, s=0.2, cmap="turbo")
plt.gca().set_aspect("equal"); plt.gca().invert_yaxis()
plt.title("Top-down view (x, y): the wrap of the papyrus")
plt.show()

# %% [markdown]
# ## 2 · The raw CT around our tile, with the mesh cutting through it
# The cube was cut around the point where our full-res tile sits. In a horizontal (z) slice the
# papyrus shows as bright curved bands. The red dots are mesh points within ±10 voxels of the
# slice. Each trace of dots should hug exactly one band. You may see two traces: the segment spirals
# around the scroll several times, so it can cross this cube on successive wraps. The hard part of
# segmentation is following *one* layer where neighbouring layers touch or merge (see the left
# panel, where the bands squeeze together) without jumping to the wrong one.

# %%
ct = load("ct_crop.zarr")
oz, oy, ox = M["ct_crop_origin_zyx"]
n = M["ct_crop_size"]
k = n // 2

near = valid & (np.abs(zs - (oz + k)) < 10) & (xs >= ox) & (xs < ox + n) & (ys >= oy) & (ys < oy + n)
fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
ax[0].imshow(stretch(ct[k])); ax[0].scatter(xs[near] - ox, ys[near] - oy, s=2, c="red")
ax[0].set_title(f"z slice {oz + k} (across the scroll)")
ax[1].imshow(stretch(ct[:, k])); ax[1].set_title("y slice (along the scroll)")
ax[2].imshow(stretch(ct[:, :, k])); ax[2].set_title("x slice (along the scroll)")
for a in ax:
    a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3 · The flattened page: surface volume overview
# 109 layers sampled along the surface normal, centred on the mesh (layer ~54). Below: the whole
# segment at 1/16 resolution. Left, the layer on the surface. Right, the max over the middle layers,
# which brings out the fibre texture.

# %%
sv = load("surface_l4.zarr")
mid = sv.shape[0] // 2
fig, ax = plt.subplots(2, 1, figsize=(16, 7))
ax[0].imshow(stretch(sv[mid])); ax[0].set_title(f"layer {mid} (on the mesh), whole segment, 1/16 res")
ax[1].imshow(stretch(sv[mid - 8:mid + 8].max(0))); ax[1].set_title("max over layers mid±8")
y0, x0 = M["surface_tile_yx"]; t = M["tile_size"]
for a in ax:
    a.add_patch(plt.Rectangle((x0 / 16, y0 / 16), t / 16, t / 16, fill=False, ec="red", lw=1.5))
    a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4 · Ink: labels vs. model predictions
# **inklabels**: where humans marked ink. **supervision**: where those labels are trustworthy
# (outside it, "no label" does not mean "no ink"). **preds**: current community model output (1/8 res).
# This is the training data for ink detection. Labels and surface volume are pixel-aligned.

# %%
lab4 = load("labels/inklabels.zarr/4", "zarr3")
sup4 = load("labels/supervision.zarr/4", "zarr3")
preds = sorted((D / "preds").glob("*.jpg"))
fig, ax = plt.subplots(2 + len(preds), 1, figsize=(16, 4 * (2 + len(preds))))
ax[0].imshow(lab4, cmap="magma"); ax[0].set_title(f"ink labels (labelled ink: {(lab4 > 0).mean():.2%} of page)")
ax[1].imshow(sup4, cmap="magma"); ax[1].set_title("supervision mask")
for a, p in zip(ax[2:], preds):
    a.imshow(np.asarray(Image.open(p).convert("L")), cmap="magma"); a.set_title(p.name[:90], fontsize=8)
for a in ax:
    a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5 · Full resolution: the tile where we'll train first
# At full resolution you can see papyrus fibres. Ink is a slight change in texture and density,
# usually only visible a few layers *into* the sheet. Scroll the depth: left to right is deeper
# through the papyrus. Last panel: labels drawn over the surface.

# %%
tile = load("surface_crop.zarr")
lab_t = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(D / "labels/inklabels.zarr/0")}}, read=True).result()[y0:y0 + t, x0:x0 + t].read().result()
c = slice(t // 2 - 512, t // 2 + 512)
depths = [mid - 12, mid - 4, mid, mid + 4, mid + 12]
fig, ax = plt.subplots(1, len(depths) + 1, figsize=(4 * (len(depths) + 1), 4.4))
for a, d in zip(ax, depths):
    a.imshow(stretch(tile[d, c, c])); a.set_title(f"layer {d}")
ax[-1].imshow(stretch(tile[mid, c, c])); ax[-1].imshow(np.ma.masked_equal(lab_t[c, c], 0), cmap="autumn", alpha=0.45)
ax[-1].set_title("labels over layer %d" % mid)
for a in ax:
    a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Where this leads
# * **Ink detection (First Letters prizes):** train a model on `(surface volume stack -> ink label)` pairs
#   like the tile above, then run it on segments that have no labels.
# * **Segmentation / unrolling (Grand Prize):** produce meshes like section 1 automatically,
#   with less than 8 hours of human input for a whole scroll.
#
# Next notebook: `02_ink_baseline`, training a model on this tile.
