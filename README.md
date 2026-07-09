# DATASCI 281 — Group 2 Final Project

**Maneuver Prediction from Ego-Vehicle Camera Images**  
UC Berkeley MIDS · Spring 2026 · Instructor: Vasha Dutell  
Team: Dawson, Matt, Nithya, Victor

---

## Project Overview

We predict the ego-vehicle's upcoming maneuver (straight, left-turn, right-turn, lane-change) from a single panoramic camera frame using classical computer vision features. Labels are derived from future trajectory geometry in the Waymo Open Dataset End-to-End (WOD-E2E, CVPR 2026).

**Classification task:** 4-class sequence-level maneuver prediction  
**Feature pipeline:** HOG · YOLO+Position · Road Geometry  
**Dataset:** 1,875 labeled training sequences from WOD-E2E

---

## Label Design

We define maneuver labels by applying a geometric classifier (`classify_future_maneuver`) to the 5-second future trajectory of each sequence, then selecting the most complex maneuver observed:

| Maneuver | Criterion | Count | % |
|---|---|---|---|
| Straight | Heading change < 8° | 971 | 51.8% |
| Right-turn | Heading change > 20°, rightward | 425 | 22.7% |
| Left-turn | Heading change > 20°, leftward | 323 | 17.2% |
| Lane-change-right | Heading change 8–20°, rightward | 86 | 4.6% |
| Lane-change-left | Heading change 8–20°, leftward | 70 | 3.7% |

**Why not Waymo's native intent labels?** Waymo's `intent` field contains routing commands (`GO_STRAIGHT`, `GO_LEFT`, `GO_RIGHT`) that explicitly exclude micro-maneuvers like lane changes. Our geometry-based labels are finer-grained and validated against Waymo's own reported driving behavior distribution in the CVPR 2026 paper (Xu et al., WOD-E2E).

---

## Repo Structure

```
281-s2-group2-final/
├── configs/
│   └── waymo_e2e_pano.yaml        # StandardE2E preprocessing config
├── data/                          # gitignored — see download instructions below
│   └── processed/
│       └── waymo_e2e/
│           └── training/          # .npz files, one per sequence
├── notebooks/
│   ├── feature_extraction.ipynb   # ← PRIMARY: feature pipeline + label generation
│   └── eda_1.ipynb                # Dataset structure exploration (val split)
├── scripts/
│   ├── check_fmaneuvers.py        # Maneuver label distribution checker
│   └── check_fmaneuvers_split.py  # Per-split maneuver counts
├── environment.yml
└── README.md
```

---

## Environment Setup

### Prerequisites

- macOS (tested on M3 Pro) or Linux
- Anaconda or Miniconda
- Google account registered for [Waymo Open Dataset access](https://waymo.com/open)
- `gcloud` CLI installed: `brew install --cask google-cloud-sdk`

### 1. Create the conda environment

```bash
CONDA_SOLVER=classic conda env create -f environment.yml
conda activate 281-s2-group2
```

> **Note:** `CONDA_SOLVER=classic` is required — the default libmamba solver fails on some dependency combinations.

### 2. Install StandardE2E (order matters)

```bash
pip install standard-e2e --no-deps
pip install av2
pip install "numpy<2"
```

> Install in this exact order. `numpy<2` must be pinned last to avoid StandardE2E breakage. The `waymo-open-dataset` proto library has no ARM64 Mac wheel; we access data through StandardE2E instead.

### 3. Set required environment variable

Add this to your shell profile (`~/.zshrc` or `~/.bash_profile`), or set it before launching Jupyter:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

> This must be set as a shell environment variable **before** Python starts. Setting it inside a notebook cell does not work.

### 4. Authenticate with Google Cloud

```bash
gcloud auth login                         # use your Waymo-registered Google account
gcloud auth application-default login
```

---

## Getting the Data

### Downloading and processing the raw data (requires Waymo access)

If you need to modify the feature pipeline, you'll need the raw `.npz` files. These are not shared (13 GB) — you need to process them from GCS.

**Step 1 — Get Waymo dataset access.** Register at [waymo.com/open](https://waymo.com/open) with your Google account. Access is granted quickly after providing an email.

**Step 2 — Process the training split from GCS.** This takes ~2 hours and ~50 GB of temporary storage. Use `caffeinate` to prevent sleep:

```bash
caffeinate -id python -m standard_e2e.caching.process_source_dataset waymo_e2e \
  --input_path=gs://waymo_open_dataset_end_to_end_camera_v_1_0_0 \
  --output_path=./data/processed \
  --split=training \
  --num_workers=1 \
  --config_file=configs/waymo_e2e_pano.yaml
```

> The correct split value is `training` (not `train`).

**Step 3 — Verify.** You should end up with 1,875 `.npz` files:

```
data/processed/waymo_e2e/training/     # 1,875 .npz files
```

Each `.npz` contains one sequence: a `(142, 384, 3)` panoramic image plus past/future trajectory waypoints at 4 Hz. Then run `notebooks/feature_extraction.ipynb` to regenerate the feature arrays.

---

## Running the Feature Extraction

Open `notebooks/feature_extraction.ipynb` with the `281-s2-group2` kernel selected in VS Code.

The notebook runs end-to-end and produces:

| Feature | Dims | Description |
|---|---|---|
| HOG | 5,796 | Histogram of Oriented Gradients via skimage |
| YOLO+Position | 32 | YOLOv8s detections (8 driving classes), normalized bbox position + relative size |
| Road Geometry | 8 | Hough lines, vanishing point (pairwise intersection voting), road centroid via color segmentation |

Full extraction over 1,875 sequences takes approximately **2 minutes**.

**Key finding — road geometry encodes pre-maneuver positioning:** The road centroid x-coordinate (`centroid_x`) shifts rightward for left-turns and leftward for right-turns, reflecting the vehicle's lane positioning before executing the maneuver. This is computable purely from pixel values with no depth or map information.

---

## Data Notes

- **Image shape:** `(142, 384, 3)` — panoramic stitch of all 8 cameras, already preprocessed by StandardE2E's `PanoImageAdapter`
- **No hood artifact:** StandardE2E crops it; no additional preprocessing needed
- **Sequence-level labels:** One label per sequence, derived from the most complex future maneuver in that 5-second clip. Individual frames do not carry labels.
- **HSV was dropped:** PCA showed HSV captured day/night lighting variation rather than maneuver-relevant variance

---

## Resources

- [WOD-E2E Dataset Page](https://waymo.com/open)
- [StandardE2E GitHub](https://github.com/stepankonev/StandardE2E)
- [StandardE2E Docs](https://standarde2e.readthedocs.io/en/latest/)
- [WOD-E2E CVPR 2026 Paper](https://arxiv.org/abs/...) — Xu et al., the source for label defensibility
