"""Re-score saved ink predictions on interior pixels only (away from surface boundaries and holes).

Models react to the transition between papyrus and the grey fill used outside / inside holes of
our surfaces. Keeping only pixels at least MARGIN_MM from any invalid pixel removes that effect.

Usage:  uv run python src/surface/interior_rescore.py --tag ms
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

D = Path(__file__).resolve().parents[2] / "data" / "scout" / "PHerc0846A"
PX_MM = 2 * 2.403 / 1000            # prediction maps are at level 1
MARGIN_MM = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ms")
    a = ap.parse_args()
    runs = json.loads((D / f"rerun_ink_{a.tag}.json").read_text())
    margin = int(round(MARGIN_MM / PX_MM))
    summary, maps = {}, {}
    for name, info in runs.items():
        f = D / f"first_ink_{name}_filled_holes_{a.tag}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        valid = np.any([z[k] > 0 for k in z.files], 0)
        interior = ndimage.binary_erosion(valid, iterations=margin)
        if interior.sum() < 1000:
            summary[name] = {"grade": info["grade"], "interior_mm2": float(interior.sum() * PX_MM ** 2), "note": "too little interior"}
            continue
        per = {}
        for k in z.files:
            p = z[k][interior]
            per[k] = {"mean": float(p.mean()), "frac_gt_0.5": float((p > 0.5).mean())}
        best = max(per, key=lambda k: per[k]["frac_gt_0.5"])
        summary[name] = {"grade": info["grade"], "interior_mm2": float(interior.sum() * PX_MM ** 2),
                         "max_frac_gt_0.5": per[best]["frac_gt_0.5"], "at": best,
                         "median_frac_gt_0.5_over_settings": float(np.median([v["frac_gt_0.5"] for v in per.values()])),
                         "settings": per}
        maps[name] = (np.where(interior, z[best], np.nan), best)
        summary[name]["interior_px"] = int(interior.sum())
        print(f"{name:22s} {info['grade']:8s} interior {summary[name]['interior_mm2']:5.1f} mm2 | "
              f"area>0.5 max {per[best]['frac_gt_0.5']:6.1%} ({best}), median over settings "
              f"{summary[name]['median_frac_gt_0.5_over_settings']:6.1%}", flush=True)
    (D / f"interior_rescore_{a.tag}.json").write_text(json.dumps(summary, indent=2))
    n = len(maps)
    cols = 4
    rows = math.ceil(n / cols)
    fig, ax = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    for a_, (name, (m, k)) in zip(np.atleast_1d(ax).ravel(), maps.items()):
        a_.imshow(m, cmap="magma", vmin=0, vmax=1)
        a_.set_title(f"{name.replace('grow_', '')} ({summary[name]['grade']}), {k}", fontsize=10)
    for a_ in np.atleast_1d(ax).ravel():
        a_.axis("off")
    plt.tight_layout(); plt.savefig(D / "diagnose" / f"interior_maps_{a.tag}.png", dpi=45); plt.close()
    print("saved", D / "diagnose" / f"interior_maps_{a.tag}.png", flush=True)


if __name__ == "__main__":
    main()
