# DATASCI 281 — Group 2 Final Project

**Maneuver Prediction from Ego-Vehicle Camera Images**  
UC Berkeley MIDS · Spring 2026 · Instructor: Vasha Dutell  
Team: Dawson, Matt, Nithya, Victor

---

## Project Overview

We predict the ego-vehicle's upcoming maneuver from a single panoramic camera frame using classical computer vision features. Labels are derived from future trajectory geometry in the Waymo Open Dataset End-to-End (WOD-E2E, CVPR 2026).

**Classification task:** 5-class sequence-level maneuver prediction (straight, left-turn, right-turn, lane-change-left, lane-change-right)  
**Feature pipeline:** HOG · YOLO+Position · Road Geometry  
**Dataset:** 1,875 labeled training sequences from WOD-E2E

---

## Label Design

We define maneuver labels by applying a geometric classifier (`classify_future_maneuver` in `scripts/build_train_manifest.py`) to the future trajectory of each sequence, then selecting the most complex maneuver observed:

| Maneuver | Criterion | Count | % |
|---|---|---|---|
| Straight | Heading change < 8° | 971 | 51.8% |
| Right-turn | Heading change > 20°, rightward | 425 | 22.7% |
| Left-turn | Heading change > 20°, leftward | 323 | 17.2% |
| Lane-change-right | Heading change 8–20°, rightward | 86 | 4.6% |
| Lane-change-left | Heading change 8–20°, leftward | 70 | 3.7% |

**Why not Waymo's native intent labels?** Waymo's `intent` field contains routing commands (`GO_STRAIGHT`, `GO_LEFT`, `GO_RIGHT`) rather than maneuvers. See `notebooks/label_design.ipynb` for the empirical justification: a cross-tabulation of intent against our derived maneuvers shows each intent smears across multiple maneuvers, and lane changes have no dedicated intent value at all. This motivates our finer-grained, geometry-derived labels. Intent is used only as a comparison baseline; it is not part of our label logic.

---

## Repo Structure

```
281-s2-group2-final/
├── configs/
│   └── waymo_e2e_pano.yaml         # StandardE2E preprocessing config
├── data/                           # raw data gitignored — see below
│   └── train_manifest.json         # sequence → target-frame map + labels (committed)
├── notebooks/
│   ├── waymo_feature_extraction.ipynb   # ← PRIMARY: feature pipeline
│   ├── label_design.ipynb          # label justification (intent vs. maneuver)
│   ├── train_inventory.ipynb       # dataset structure / .npz schema
│   ├── filter_day_night.ipynb      # day/night filtering
│   ├── initialExploration_Matt.ipynb
│   └── feature_exploration_update.ipynb
├── scripts/
│   └── build_train_manifest.py     # generates data/train_manifest.json
├── slides/                         # class update decks
├── environment.yml
└── README.md
```

---

## How the Data Is Structured

This is the most important thing to understand before setting up.

StandardE2E writes **one `.npz` file per frame**, not per sequence. After processing the training split you will have roughly **415,000 `.npz` files** named `{sequence_hash}_{frame_number}.npz`. This is expected and correct.

The **manifest** (`data/train_manifest.json`, committed to the repo) is what ties everything together. It maps each of the **1,875 sequences** to a single target frame used for classification, along with its maneuver label and context frames:

```json
"003b62820d0e9345eb025de35b046999": {
  "label": "straight",
  "target_fname": "003b62820d0e9345eb025de35b046999_9.npz",
  "context_fnames": [...],
  "best_fname": "003b62820d0e9345eb025de35b046999_9.npz"
}
```

The feature extraction notebook loads only the `target_fname` for each sequence, so you never load all 415k files — just the 1,875 target frames.

---

## Environment Setup

### Prerequisites

- macOS (tested on M3 Pro) or Linux. Windows works via Anaconda (drop `caffeinate` from commands; it is macOS-only).
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

## Downloading and Processing the Raw Data

You only need this if you want to modify the feature pipeline. The raw data is ~13 GB and is not shared; process it from GCS.

**Step 1 — Get Waymo dataset access.** Register at [waymo.com/open](https://waymo.com/open) with your Google account. Access is granted quickly after providing an email.

**Step 2 — Process the training split from GCS.** This takes ~2 hours. On macOS, prefix with `caffeinate -id` to prevent sleep (omit on Windows/Linux):

```bash
caffeinate -id python -m standard_e2e.caching.process_source_dataset waymo_e2e \
  --input_path=gs://waymo_open_dataset_end_to_end_camera_v_1_0_0 \
  --output_path=./data/processed \
  --split=training \
  --num_workers=1 \
  --config_file=configs/waymo_e2e_pano.yaml
```

> The correct split value is `training` (not `train`).

**Step 3 — Verify.** You should end up with ~415,000 per-frame `.npz` files in `data/processed/waymo_e2e/training/`. To count them:

```bash
ls data/processed/waymo_e2e/training/ | wc -l      # macOS / Linux
dir data\processed\waymo_e2e\training\ | more       # Windows
```

The `train_manifest.json` is already committed to the repo, so you do **not** need to regenerate it. (If you ever do need to rebuild it — e.g. after changing the labeling logic — run `python scripts/build_train_manifest.py` from the repo root.)

Then run `notebooks/waymo_feature_extraction.ipynb` to extract features.

---

## Running the Feature Extraction

Open `notebooks/waymo_feature_extraction.ipynb` with the `281-s2-group2` kernel selected in VS Code.

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
- **Sequence-level labels:** One label per sequence, derived from the most complex future maneuver in that clip. Individual frames do not carry labels.
- **HSV was dropped:** PCA showed HSV captured day/night lighting variation rather than maneuver-relevant variance

---

## Resources

- [WOD-E2E Dataset Page](https://waymo.com/open)
- [StandardE2E GitHub](https://github.com/stepankonev/StandardE2E)
- [StandardE2E Docs](https://standarde2e.readthedocs.io/en/latest/)
- [WOD-E2E CVPR 2026 Paper (Xu et al.)](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_WOD-E2E_Waymo_Open_Dataset_for_End-to-End_Driving_in_Challenging_Long-tail_CVPR_2026_paper.pdf) — source for label defensibility