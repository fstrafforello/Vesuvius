"""Multi-sheet survey: grow several sheets (with seed retries), then run the harmonised ink model.

For every target sheet (a ridge position y at column x, slice z), try seeds at that ridge and at
a few nearby positions along the same sheet until one grows past MIN_POINTS. Each grown surface is
pinhole-filled and run through first_ink.py with harmonisation. Results are listed in survey.json.

Usage:  uv run python src/surface/survey.py --targets 581 652 699 869 924 964 [--x 1200 --z 856]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from trace import D, load  # noqa: E402

MIN_POINTS = 5000
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def run(args):
    r = subprocess.run([PY, "-u", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout + r.stderr


def ridge_near(block, z, y, x, win=6):
    col = ndimage.gaussian_filter1d(block[z, y - win:y + win + 1, x].read().result().astype(np.float32), 1.0)
    return y - win + int(np.argmax(col))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", required=True)
    ap.add_argument("--x", type=int, default=1200)
    ap.add_argument("--z", type=int, default=856)
    ap.add_argument("--region", default=None, help="named region (default: original)")
    ap.add_argument("--ct", default="l1_filled", help="CT download to use for the ink runs")
    ap.add_argument("--phase", choices=["all", "grow", "ink"], default="all",
                    help="grow: grow + check only; ink: ink runs on already grown surfaces")
    a = ap.parse_args()
    block, _, _ = load(1, a.region)
    prefix = f"{a.region}_" if a.region else ""
    out_json = D / f"survey{'_' + a.region if a.region else ''}.json"
    results = json.loads(out_json.read_text()) if a.phase == "ink" and out_json.exists() else {}
    for y in a.targets:
        name = f"{prefix}s{y}"
        if a.phase == "ink":
            r = results.get(name, {})
            if r.get("status") == "did not grow" or not (D / f"grow_{name}_f").exists():
                continue
            ink = run(["src/surface/first_ink.py", "run", "--name", f"grow_{name}_f", "--ct", a.ct,
                       "--fill-holes", "--harmonize", "--blur", "1.0", "--tag", "harm"])
            lines = [l for l in ink.splitlines() if l.startswith("normal") or "valid surface" in l or "Error" in l]
            r["ink"] = lines; results[name] = r
            print(f"{name}: {r.get('mm2')} mm^2, {r.get('section_cols_with_hop_pct')}% columns with hop", flush=True)
            for l in lines:
                print("   ", l, flush=True)
            continue
        grown = 0
        for dz, dx in ((0, 0), (0, 150), (0, -150), (120, 0), (-120, 0), (0, 300), (0, -300)):
            z, x = a.z + dz, a.x + dx
            yy = ridge_near(block, z, y, x, win=10 if (dz or dx) else 3)
            out = run(["src/surface/grow3d.py", "--name", name, "--seed", str(z), str(yy), str(x), "--max-points", "60000",
                       "--across", "2.0"] + (["--region", a.region] if a.region else []))
            m = re.search(r"grown: (\d+) points = ([\d.]+) mm", out)
            grown = int(m.group(1)) if m else 0
            print(f"{name}: seed z{z} y{yy} x{x} -> {grown} points", flush=True)
            if grown >= MIN_POINTS:
                break
        if grown < MIN_POINTS:
            results[name] = {"grown": grown, "status": "did not grow"}
            continue
        hop = re.search(r"hop check: .*?(\d+\.\d+)% of neighbour pairs", out)
        run(["src/surface/grow_to_trace.py", "--name", name])
        sec = run(["src/surface/section_check.py", "--name", f"grow_{name}"])
        secm = re.search(r"(\d+)% of columns contain", sec)
        print(f"{name}: section check - {secm.group(1) if secm else '?'}% of columns with a hop", flush=True)
        run(["src/surface/fill_grid_holes.py", "--name", name])
        lines = []
        if a.phase == "all":
            ink = run(["src/surface/first_ink.py", "run", "--name", f"grow_{name}_f", "--ct", a.ct,
                       "--fill-holes", "--harmonize", "--blur", "1.0", "--tag", "harm"])
            lines = [l for l in ink.splitlines() if l.startswith("normal") or "valid surface" in l]
        results[name] = {"grown": grown, "mm2": float(m.group(2)), "hop_pct": float(hop.group(1)) if hop else None,
                         "section_cols_with_hop_pct": int(secm.group(1)) if secm else None,
                         "seed": [z, yy, x], "ink": lines}
        print(f"{name}: {m.group(2)} mm^2; " + "; ".join(lines[:1]), flush=True)
        for l in lines[1:]:
            print("   ", l, flush=True)
        out_json.write_text(json.dumps(results, indent=2))
    out_json.write_text(json.dumps(results, indent=2))
    print("saved", out_json)


if __name__ == "__main__":
    main()
