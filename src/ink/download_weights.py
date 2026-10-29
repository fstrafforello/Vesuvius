"""Download the released ink-model weights into runs/, where the analysis scripts look for them.

Each model goes to runs/<run folder from registry.py>/model.pt and is checked against the SHA-256 in
the release's models.json.

Weights were trained on Vesuvius Challenge data and are released under CC BY-NC 4.0 (non-commercial),
not under the MIT license of the code. The Scroll 4 / Scroll 1 training data carries the additional
EduceLab-Scrolls terms (see README).

Usage:  uv run python src/ink/download_weights.py                 # all models
        uv run python src/ink/download_weights.py multiscroll     # one model
        uv run python src/ink/download_weights.py --from-dir dist/weights-v1   # from local files
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import requests

from registry import MODELS, RUNS

RELEASE = "https://github.com/fstrafforello/Vesuvius/releases/download/weights-v1"


def fetch(name, dest, from_dir=None):
    if from_dir:
        shutil.copyfile(Path(from_dir) / name, dest)
        return
    with requests.get(f"{RELEASE}/{name}", stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="short names (default: all released models)")
    ap.add_argument("--from-dir", default=None, help="copy from a local folder instead of downloading")
    ap.add_argument("--runs-dir", default=str(RUNS))
    a = ap.parse_args()
    runs = Path(a.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    card_path = runs / "models.json"
    fetch("models.json", card_path, a.from_dir)
    card = json.loads(card_path.read_text())
    for n in a.models or list(card):
        info = card[n]
        folder = runs / MODELS.get(n, info["run_folder"])
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / "model.pt"
        if dest.exists() and sha256(dest) == info["sha256"]:
            print(f"{n}: already present", flush=True)
            continue
        print(f"{n}: downloading {info['file']} ({info['size_mb']} MB)", flush=True)
        tmp = folder / "model.pt.part"
        fetch(info["file"], tmp, a.from_dir)
        if sha256(tmp) != info["sha256"]:
            tmp.unlink()
            raise SystemExit(f"{n}: checksum mismatch, download discarded")
        tmp.replace(dest)
        auc, f05 = info["val_metrics"]["auc"], info["val_metrics"]["f05"]
        print(f"{n}: ok -> {dest}  (validation on {info['held_out']}: AUC {auc:.3f}, F0.5 {f05:.3f})", flush=True)


if __name__ == "__main__":
    main()
