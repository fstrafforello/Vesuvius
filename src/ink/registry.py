"""Trained ink models, referred to by short name in the analysis scripts.

Weights are not stored in the repository (about 100 MB each, trained on CC BY-NC data). Download the
released ones with src/ink/download_weights.py, or train your own with src/ink/train_l1.py (commands in
docs/WRITEUP.md) and edit the folder names below to match your runs.
"""

from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "runs"

MODELS = {
    "quick3d": "ink_l1_3d_20261008_113212",            # small 3D front end + 2D U-Net, Scroll 4 only
    "full3d": "ink_l1_full3d_20261009_092602",         # 3D ResNet-18 encoder, Scroll 4 only
    "robust": "ink_l1_3d_20261011_182058",             # per-patch robust normalisation, Scroll 4 only
    "multiscroll": "ink_l1_3d_20261013_115628",        # Scroll 4 + PHerc0139 + Scroll 1, PHerc0841 held out
    "multiscroll_holes": "ink_l1_3d_20261013_152327",  # as above + grey-filled hole augmentation
}

SCROLL4_PAIR = ["quick3d", "full3d"]   # the two Scroll 4 models whose average is our best Scroll 4 result


def run_dir(name):
    """Run folder for a model short name (or a run folder name given directly)."""
    return RUNS / MODELS.get(name, name)
