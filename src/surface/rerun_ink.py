"""Re-run a chosen ink model on many grown PHerc0846A surfaces (no harmonisation) and collect results.

Usage:  uv run python src/surface/rerun_ink.py --run multiscroll --tag msd --variant d
        (--run: a short name from src/ink/registry.py, or a run folder under runs/)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "scout" / "PHerc0846A"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

# (surface, CT download, quality grade from the per-step hop rate)
SURFACES = [
    ("grow_g3_f", "l1_filled", "verified"),
    ("grow_s581_f", "l1_filled", "trusted"), ("grow_s652_f", "l1_filled", "trusted"),
    ("grow_s699_f", "l1_filled", "trusted"), ("grow_s869_f", "l1_filled", "trusted"),
    ("grow_s924_f", "l1_filled", "trusted"),
    ("grow_top_s474_f", "top_band", "trusted"), ("grow_top_s669_f", "top_band", "trusted"),
    ("grow_top_s788_f", "top_band", "trusted"), ("grow_top_s361_f", "top_band", "caution"),
    ("grow_bottom_s130_f", "bottom_band", "trusted"), ("grow_bottom_s255_f", "bottom_band", "trusted"),
    ("grow_bottom_s372_f", "bottom_band", "trusted"), ("grow_bottom_s648_f", "bottom_band", "caution"),
    ("grow_bottom_s798_f", "bottom_band", "caution"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default="ms")
    ap.add_argument("--variant", default="f", help="surface variant suffix: f = pinhole-filled, d = densified")
    a = ap.parse_args()
    results = {}
    for name, ct, grade in SURFACES:
        name = name[:-2] + "_" + a.variant if a.variant != "f" else name
        if not (D / name).exists():
            print(f"{name}: missing, skipped", flush=True); continue
        r = subprocess.run([PY, "-u", "src/surface/first_ink.py", "run", "--name", name, "--ct", ct, "--fill-holes",
                            "--runs", a.run, "--tag", a.tag], cwd=ROOT, capture_output=True, text=True)
        lines = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("normal") or "valid surface" in l or "Error" in l]
        results[name] = {"grade": grade, "lines": lines}
        print(f"{name} ({grade}):", flush=True)
        for l in lines:
            print("   ", l, flush=True)
        (D / f"rerun_ink_{a.tag}.json").write_text(json.dumps(results, indent=2))
    print("saved", D / f"rerun_ink_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
