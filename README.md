# psn-vesuvius

Independent tools for the [Vesuvius Challenge](https://scrollprize.org): tracing papyrus surfaces in an
unsegmented eligible scroll, rendering them, and testing how well ink models trained on one scroll
transfer to another.

Everything here was built and run on a single laptop GPU (24 GB), without the community segmentation
or ink-detection models. The full account, with numbers and figures, is in
**[docs/WRITEUP.md](docs/WRITEUP.md)**.

## Results in brief

- **Renderer validated against official data.** Re-rendering a Scroll 4 segment from its mesh matches
  the published surface volume at 0.993 correlation (3.6 grey levels mean difference), and the ink
  model scores the same on both (F0.5 0.856 vs 0.851).
- **3D surface growing on an eligible scroll (PHerc0846A, 2.4 µm).** A grid grower with an
  across-sheet constraint, a densify pass that fills holes point by point against the data, and a
  vertical section check that catches sheet jumps the usual per-slice checks miss. About 113 mm² of
  usable surface interior on 15 sheets in three regions.
- **Ink models do not transfer between scrolls by default.** Scroll 4 models drop from AUC about 0.90
  (held-out Scroll 4 segment) to 0.66–0.80 on PHerc0841, a labelled scroll from the same scan series as
  PHerc0846A. Training on three scrolls raises that to 0.80–0.87.
- **Brightness harmonisation is a trap.** It removes false alarms on unfamiliar papyrus, but on real
  ink it also suppresses the signal (PHerc0841 AUC falls by 0.07–0.11).
- **Calibrated negative result on PHerc0846A.** On most surveyed interiors the multi-scroll model stays
  at its false-alarm rate on known ink-free papyrus. The one consistent response traces back to a crack
  in the surface. No letters were found.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (CUDA 13 wheels are
configured in `pyproject.toml`; change the index URL for other CUDA versions).

```bash
uv sync
```

On Windows, `env.ps1` keeps every tool, cache and the environment inside the repository folder:

```powershell
. .\env.ps1
uv sync
```

Scroll data is downloaded on demand from the public
[open-data bucket](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/index.html) into `data/`
(git-ignored). Model runs go to `runs/` (git-ignored). Model names used by the analysis scripts are
defined in [src/ink/registry.py](src/ink/registry.py).

### Pretrained weights

Three trained ink models are attached to the
[`weights-v1` release](https://github.com/fstrafforello/Vesuvius/releases/tag/weights-v1). To download
them into `runs/` with checksum verification:

```bash
uv run python src/ink/download_weights.py
```

| Name | Architecture | Trained on | Validated on | AUC | F0.5 |
|---|---|---|---|---|---|
| `multiscroll` | small 3D front end + 2D U-Net | Scroll 4, PHerc0139, Scroll 1 | PHerc0841 (unseen scroll) | 0.846 | 0.701 |
| `quick3d` | small 3D front end + 2D U-Net | Scroll 4 | Scroll 4 segment w031 | 0.898 | 0.787 |
| `full3d` | 3D ResNet-18 encoder | Scroll 4 | Scroll 4 segment w031 | 0.920 | 0.785 |

All take 32 surface-volume layers around the surface at half resolution (level 1, about 4.8 µm px).
For a new scroll, use `multiscroll`; the Scroll 4 models transfer poorly (see the write-up).
The weights were trained on Vesuvius Challenge data and are released under **CC BY-NC 4.0**, not the
MIT license of the code.

## Repository map

Main pipeline:

| Step | Scripts |
|---|---|
| Fetch training data | `src/fetch_ink_l1.py` (Scroll 4), `src/ink/fetch_labelled.py` (any labelled scroll) |
| Train ink models | `src/ink/train_l1.py` (pause/resume, multi-scroll, held-out scroll), `src/ink/models.py` |
| Cross-scroll evaluation | `src/ink/cross_scroll_test.py` |
| Scout an eligible scroll | `src/surface/scout.py`, `scout_zoom.py`, `fetch_trace_region.py`, `fetch_trace_l1.py` |
| Grow surfaces in 3D | `src/surface/grow3d.py`, `grid_densify.py`, `survey.py` |
| Check surfaces | `src/surface/section_check.py` (vertical hop check), `grow_to_trace.py` |
| Render + run ink models | `src/surface/render.py`, `check_render.py` (validation), `fetch_ct_band.py`, `first_ink.py`, `rerun_ink.py` |
| Read results | `src/surface/interior_rescore.py`, `inspect_candidate.py` |

Experiments kept so the write-up can be reproduced (superseded by the scripts above):
`src/surface/trace.py`, `trace_anchor.py`, `fill_surface.py`, `merge_traces.py`, `clean_core.py`,
`fill_grid_holes.py`, `grow_clean.py`, `sheet_field.py`, `flatten_preview.py`, `depth_profile.py`,
`diagnose_domain.py`, `harmonize_control.py`, `find_known_ink.py`, `src/ink/train_baseline.py`,
`train_multi.py`, `postprocess_eval.py`, `compare_preds.py`, `bench_model.py`, `src/fetch_starter.py`,
`src/fetch_ink_blocks.py`.

`notebooks/01_explore_segment.py` is a guided first look at a Scroll 4 segment (mesh, CT, flattened
surface, labels). Run it as a notebook with `uv run python src/py2nb.py notebooks/01_explore_segment.py --execute`.

## Data and licensing

Code and documentation: MIT ([LICENSE](LICENSE)).

Scroll data is **not** included and is not covered by the MIT license. Vesuvius Challenge data is
released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); Scrolls 1–4 and
Fragments 1–6 carry additional EduceLab-Scrolls terms (see [scrollprize.org/data](https://scrollprize.org/data)).
The figures in `docs/figures` show PHerc0846A data (Vesuvius Challenge, CC BY-NC 4.0) and are shared
under the same license.

If you use data from Scrolls 1–4, cite: Parsons, S., Parker, C. S., Chapman, C., Hayashida, M., & Seales,
W. B. (2023). *EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT.*
arXiv:2304.02084.
