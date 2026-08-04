# DATASCI 281 — Group 2 Final Project

**Maneuver Prediction from Ego-Vehicle Camera Images**
UC Berkeley MIDS · Spring 2026 · Instructor: Vasha Dutell
Team: Dawson, Matt, Nithya, Victor

---

## Project Overview

We predict the ego-vehicle's upcoming maneuver from ego-vehicle panoramic camera
frames using **classical computer-vision features** (no end-to-end deep nets for the
classifier). Labels are derived from future-trajectory geometry in the Waymo Open
Dataset End-to-End (WOD-E2E, CVPR 2026).

**Classification task (final):** 3-class sequence-level maneuver prediction —
`straight`, `left-turn`, `right-turn`.
**Dataset (final):** the WOD-E2E training (2,037) and validation (479) splits are pooled —
Waymo's test split cannot be labeled (future trajectories withheld) — giving **1,966
three-class sequences**, re-split stratified 70/15/15 (train 1,376 / val 295 / test 295)
with a fixed seed. The frozen fold assignment lives in `data/cnn_2d_seq_fold.csv`, and
every feature family joins it by sequence id.

> **Task history — 5-class → 3-class.** The project began as a 5-class task
> (`straight`, `left-turn`, `right-turn`, `lane-change-left`, `lane-change-right`).
> The two lane-change classes (and the 11 rare `stationary` sequences) proved to be
> weakly separable with frame-level classical features and were dropped in favor of a
> focused 3-class problem. The earlier 5-class exploration and an abandoned
> daytime-only-filtering direction are preserved under [`archive/`](archive/README.md).

---

## Results

The final benchmark (`notebooks/benchmark_pooled_3class.ipynb`) evaluates every feature
family on the pooled 1,966-sequence set with the frozen 70/15/15 fold: fit on train,
select on validation macro-F1, report once on the held-out test fold. Balanced SVM (RBF)
and Random Forest throughout.

| Pipeline (test fold) | Accuracy | Macro-F1 |
|---|---|---|
| Majority-class baseline | 0.461 | 0.210 |
| HOG / SVM | 0.576 | 0.526 |
| 2D CNN embedding / SVM | 0.542 | 0.517 |
| BDD100K ResNet-50 embedding / RF | 0.512 | 0.441 |
| **Trend + Flow / RF (best classical)** | **0.671** | **0.607** |
| **Trend + Flow + frozen MoViNet / SVM (final)** | **0.746** | **0.693** |

**Final result: Trend+Flow stacked with frozen MoViNet-A0 (Kinetics-600) video
embeddings, SVM — 74.6% accuracy / 0.693 macro-F1** (per-class recall 0.94 / 0.67 / 0.46
for straight / right / left). The best purely classical model is **Trend+Flow with a
Random Forest (0.671 / 0.607)** — 22 hand-crafted temporal dimensions that beat every
single-frame family, including learned embeddings. Trend+Flow captures how road geometry
and optical-flow quantities drift across the context window; a driving-domain
*single-frame* embedding (BDD100K ResNet-50) does not help, confirming that motion, not
appearance, carries the maneuver signal. Selection details, ablations (CNN, BDD), and a
documented validation-selection-noise example are in the benchmark notebook's read-out;
`notebooks/efficiency_and_tuning.ipynb` adds hyperparameter search and timing.

> Earlier iteration results (e.g. Trend+Flow/RF 0.648/0.587, HOG/SVM 0.601/0.547,
> baseline 0.464) were computed on the pre-pooling 1,604-sequence 75/25 split and remain
> in `framediff_v2_3class.ipynb` / `feature_pipeline_v3_3class.ipynb` as history — they
> are not comparable to the pooled numbers above.

---

## Label Design

Maneuver labels are derived from the geometry of each sequence's future trajectory,
following Waymo's official trajectory-shape classifier (`ClassifyTrack`). Full
methodology, thresholds, and citations are in [docs/labeling.md](docs/labeling.md).

Full 5-class label distribution (before the 3-class restriction):

| Maneuver | Count | % |
|---|---|---|
| Straight | 745 | 36.6% |
| Right-turn | 467 | 22.9% |
| Left-turn | 392 | 19.2% |
| Lane-change-right | 222 | 10.9% |
| Lane-change-left | 200 | 9.8% |
| Stationary | 11 | 0.5% |

**Why not Waymo's native intent labels?** Waymo's `intent` field (`GO_STRAIGHT`,
`GO_LEFT`, `GO_RIGHT`) contains routing commands rather than executed maneuvers. See
`notebooks/label_design.ipynb` for the empirical justification: a cross-tabulation of
intent against derived maneuvers shows each intent smears across multiple maneuvers, and
lane changes have no dedicated intent value at all. Intent is used only as a comparison
baseline; it is not part of our label logic.

---

## Repo Structure

```
281-s2-group2-final/
├── configs/
│   ├── waymo_e2e_pano.yaml              # StandardE2E preprocessing config
│   └── waymo_e2e_keep_calibration.yaml  # calibration-extraction config
├── data/                                # raw/processed data gitignored — see below
│   ├── train_manifest.json              # sequence → target-frame map + labels (committed)
│   └── processed/                       # per-frame .npz + feature .npy cache (gitignored)
├── docs/
│   └── labeling.md                      # maneuver-labeling methodology + citations
├── notebooks/                           # CURRENT 3-class pipeline
│   ├── waymo_feature_extraction.ipynb   # ← BASE: HOG / HSV / YOLO / road-geometry features
│   ├── feature_pipeline_v3_3class.ipynb # ← 3-class assembly + engineered features (old-split benchmark)
│   ├── framediff_v2_3class.ipynb        # ← temporal Trend+Flow features (winning classical set)
│   ├── benchmark_pooled_3class.ipynb    # ← FINAL benchmark on the pooled 70/15/15 fold + figures
│   ├── efficiency_and_tuning.ipynb      # hyperparameter search + efficiency/accuracy tradeoff
│   ├── CNN_2D_resplit.ipynb             # builds the pooled 1,966-seq stratified fold
│   ├── CNN_2D / CNN_3D / CNN_3D_v2.ipynb# learned features (2D, prior-only 3D, frozen MoViNet)
│   ├── label_design.ipynb               # label justification (intent vs. maneuver)
│   ├── dimensionality_reduction_analysis.ipynb  # PCA variance (mirrors framediff chart)
│   └── train_inventory.ipynb            # dataset structure / .npz schema
├── scripts/
│   ├── build_train_manifest.py          # generates data/train_manifest.json
│   ├── extract_calibration.py           # camera-calibration extraction
│   └── extract_waymo_e2e_calibration.py
├── archive/                             # previous attempts, retained + documented
│   └── README.md                        # what each was, why it was set aside, best result
├── slides/                              # class update deck (class_update_slides_v3.pptx)
├── environment.yml
└── README.md
```

---

## Pipeline: how the notebooks fit together

1. **`waymo_feature_extraction.ipynb`** — the base feature extractor. Produces the
   canonical per-sequence arrays (`hog.npy`, `hsv.npy`, `yolo.npy`, `road.npy`,
   `labels.npy`, `seq_ids.npy`). Its per-frame feature functions are the **single source
   of truth** copied into the notebooks below.
2. **`feature_pipeline_v3_3class.ipynb`** — the primary pipeline. Audits/aligns the base
   arrays, filters to the 3-class task, computes engineered features (`road_v2`, vehicle
   occupancy, IPM/BEV drivable-corridor, scene-cue + traffic-law encodings), and runs the
   static-feature benchmark. Saves the chosen arrays with a `_3class` suffix.
3. **`framediff_v2_3class.ipynb`** — adds temporal features across the context window:
   trend (slopes of road/VP/detection quantities) plus dense Farneback **optical flow**
   (speed and heading proxies). Produces the **Trend+Flow** set that wins the final
   benchmark, aligned to the 3-class `seq_ids` for direct concatenation.

`dimensionality_reduction_analysis.ipynb` and `train_inventory.ipynb` are supporting
analysis and are not part of the runtime pipeline: the former reproduces the cumulative
explained-variance chart from `framediff_v2_3class.ipynb` — a PCA reducibility check on each
3-class feature family — and the latter documents the dataset structure and `.npz` schema.

### Notebooks at a glance

| Notebook | Role | Key inputs | Key outputs |
|---|---|---|---|
| `waymo_feature_extraction.ipynb` | Base feature extractor | `.npz` frame cache, `train_manifest.json` | `hog` / `hsv` / `yolo` / `road` / `labels` / `seq_ids` `.npy` |
| `feature_pipeline_v3_3class.ipynb` | 3-class assembly, engineered features, benchmark | base arrays (+ optional embeddings) | `*_3class.npy` engineered features, `X_final_3class.npy` |
| `framediff_v2_3class.ipynb` | Temporal Trend+Flow features (winning set) | base arrays, `.npz` frames | `framediff_v2_{trend,flow,3c_aligned}.npy` |
| `label_design.ipynb` | Label justification (Waymo intent vs. maneuver) | manifest, Waymo `intent` | cross-tabulation (analysis) |
| `dimensionality_reduction_analysis.ipynb` | PCA variance per feature family (mirrors framediff's chart) | 3-class feature arrays | variance-curve `.png` + summary `.csv` |
| `train_inventory.ipynb` | Dataset structure / `.npz` schema EDA | `.npz` cache | (analysis) |

Notebooks read and write feature arrays under `data/processed/waymo_e2e/features/`, which is
gitignored — regenerate them by running the pipeline, or pull them from the team Google Drive.

---

## How the Data Is Structured

This is the most important thing to understand before setting up.

StandardE2E writes **one `.npz` file per frame**, not per sequence. After processing the
training split you will have roughly **415,000 `.npz` files** named
`{sequence_hash}_{frame_number}.npz`. This is expected and correct.

The **manifest** (`data/train_manifest.json`, committed to the repo) ties everything
together. It maps each of the **2,037 sequences** to a single target frame used for
classification, along with its maneuver label and context frames:

```json
"003b62820d0e9345eb025de35b046999": {
  "label": "straight",
  "target_fname": "003b62820d0e9345eb025de35b046999_9.npz",
  "context_fnames": ["...", "..."],
  "n_context": 12,
  "best_fname": "003b62820d0e9345eb025de35b046999_9.npz"
}
```

The feature-extraction notebook loads only the `target_fname` for each sequence (plus
`context_fnames` for the temporal features), so you never load all 415k files.

---

## Environment Setup

### Prerequisites

- macOS (tested on M3 Pro) or Linux. Windows works via Anaconda (drop `caffeinate` from
  commands; it is macOS-only).
- Anaconda or Miniconda
- Google account registered for [Waymo Open Dataset access](https://waymo.com/open)
- `gcloud` CLI installed: `brew install --cask google-cloud-sdk`

### 1. Create the conda environment

```bash
CONDA_SOLVER=classic conda env create -f environment.yml
conda activate 281-s2-group2
```

> **Note:** `CONDA_SOLVER=classic` is required — the default libmamba solver fails on
> some dependency combinations.

### 2. Install StandardE2E (order matters)

```bash
pip install standard-e2e --no-deps
pip install av2
pip install "numpy<2"
```

> Install in this exact order. `numpy<2` must be pinned last to avoid StandardE2E
> breakage. The `waymo-open-dataset` proto library has no ARM64 Mac wheel; we access data
> through StandardE2E instead.

### 3. Set required environment variable

Set this in your shell **before** launching Python or Jupyter (add it to `~/.zshrc` or
`~/.bash_profile` to make it permanent):

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

> This must be a shell environment variable set before Python starts. Setting it inside a
> notebook cell or via `os.environ` does not reliably work, because the OpenMP libraries
> load at import time.

### 4. Authenticate with Google Cloud

```bash
gcloud auth login                         # use your Waymo-registered Google account
gcloud auth application-default login
```

---

## Downloading and Processing the Raw Data

You only need this if you want to modify the feature pipeline. The raw data is ~13 GB and
is not shared; process it from GCS. (The extracted feature `.npy` arrays are shared
separately via the team Google Drive — see below — so most reruns can skip this.)

**Step 1 — Get Waymo dataset access.** Register at [waymo.com/open](https://waymo.com/open)
with your Google account. Access is granted quickly after providing an email.

**Step 2 — Process the training split from GCS.** This takes ~2 hours. On macOS, prefix
with `caffeinate -id` to prevent sleep (omit on Windows/Linux):

```bash
caffeinate -id python -m standard_e2e.caching.process_source_dataset waymo_e2e \
  --input_path=gs://waymo_open_dataset_end_to_end_camera_v_1_0_0 \
  --output_path=./data/processed \
  --split=training \
  --num_workers=1 \
  --config_file=configs/waymo_e2e_pano.yaml
```

> The correct split value is `training` (not `train`).

**Step 3 — Verify.** You should end up with ~415,000 per-frame `.npz` files in
`data/processed/waymo_e2e/training/`:

```bash
ls data/processed/waymo_e2e/training/ | wc -l      # macOS / Linux
```

The `train_manifest.json` is already committed, so you do **not** need to regenerate it.
(If you ever do — e.g. after changing labeling logic — run
`KMP_DUPLICATE_LIB_OK=TRUE python scripts/build_train_manifest.py` from the repo root.)

---

## Running the Pipeline

With the `281-s2-group2` kernel selected, run the notebooks in order:

1. `notebooks/waymo_feature_extraction.ipynb` — extracts base HOG / HSV / YOLO /
   road-geometry arrays (~2 min). Skippable if you already have the feature `.npy` files
   from the team Google Drive.
2. `notebooks/framediff_v2_3class.ipynb` — extracts temporal Trend+Flow features. Farneback
   optical flow is the bottleneck (~15–20 min; run under `caffeinate -i`).
3. `notebooks/feature_pipeline_v3_3class.ipynb` — assembles the 3-class feature sets
   (its internal benchmark is the old 1,604/75-25 split, kept as history).
4. `notebooks/benchmark_pooled_3class.ipynb` — the **final benchmark** on the pooled
   70/15/15 fold: full family comparison, final-model selection, MoViNet stack, ablations
   (CNN, BDD), and the presentation/paper figures (saved under `outputs/`).

**Feature families produced:**

| Feature | Description |
|---|---|
| HOG | Histogram of Oriented Gradients via skimage |
| YOLO + Position | YOLOv8s detections (driving-relevant classes), normalized bbox position + relative size |
| Road Geometry (v1/v2) | Hough lines, vanishing point (pairwise intersection voting), road centroid via color segmentation |
| Trend | Per-quantity linear slope across the context window (road centroid, VP, detection balance) |
| Optical Flow | Dense Farneback flow in the road ROI — speed and left/right heading asymmetry proxies |

**Key finding — road geometry and flow encode pre-maneuver motion:** the road centroid
and vanishing-point x-coordinates drift with heading, and net horizontal optical flow
reverses sign between left- and right-turns. Both are computable purely from pixels with
no depth or map information.

---

## Data Notes

- **Image shape:** `(142, 384, 3)` — panoramic stitch of all 8 cameras, already
  preprocessed by StandardE2E's `PanoImageAdapter`.
- **No hood artifact:** StandardE2E crops it; no additional preprocessing needed.
- **Sequence-level labels:** one label per sequence, derived from the most complex future
  maneuver in that clip. Individual frames do not carry labels. See
  [docs/labeling.md](docs/labeling.md).
- **HSV was dropped:** PCA showed HSV captured day/night lighting variation rather than
  maneuver-relevant variance.
- **Day and night both kept:** an earlier daytime-only-filtering direction (see
  `archive/`) was abandoned; the final pipeline uses all lighting conditions.

---

## Resources

- [WOD-E2E Dataset Page](https://waymo.com/open)
- [StandardE2E GitHub](https://github.com/stepankonev/StandardE2E)
- [StandardE2E Docs](https://standarde2e.readthedocs.io/en/latest/)
- [WOD-E2E CVPR 2026 Paper (Xu et al.)](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_WOD-E2E_Waymo_Open_Dataset_for_End-to-End_Driving_in_Challenging_Long-tail_CVPR_2026_paper.pdf) — source for label defensibility
- [WOMD CVPR 2021 Paper (Ettinger et al.)](https://arxiv.org/abs/2104.10133) — source of the ClassifyTrack maneuver taxonomy and thresholds