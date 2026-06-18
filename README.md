# DATASCI 281 — Section 2, Group 2 Final Project

**Task:** Multiclass classification of long-tail driving scenarios from front-camera frames  
**Dataset:** Waymo Open Dataset – End-to-End Driving (WOD-E2E, CVPR 2026)

## Classes
Construction, Cut-ins, Cyclist, Foreign Object Debris, Intersections,
Multi-Lane Maneuvers, Others, Pedestrian, Single-Lane Maneuvers, Special Vehicles

## Setup

### 1. Get dataset access
Register at https://waymo.com/open and request access to the WOD-E2E dataset.
Once approved, authenticate with `gcloud`:

```bash
brew install --cask google-cloud-sdk
gcloud init
gcloud auth login
```

### 2. Clone this repo
```bash
git clone https://github.com/YOUR_GH_USERNAME/281-s2-group2-final.git
cd 281-s2-group2-final
```

### 3. Create the conda environment
```bash
conda env create -f environment.yml
conda activate 281-s2-group2
```

### 4. Download a val shard and the scenario labels
```bash
mkdir -p data
gsutil cp gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/val_sequence_name_to_scenario_cluster.json data/
gsutil cp gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/val_202504211843.tfrecord-00000-of-00093 data/
```

### 5. Launch the notebook
```bash
jupyter notebook notebooks/exploration.ipynb
```

## Repo Structure
```
├── data/                  # local data (gitignored — do not commit tfrecords)
├── notebooks/
│   └── exploration.ipynb  # starter notebook
├── scripts/               # extraction and processing scripts
├── environment.yml
└── README.md
```

## Data Notes
- Labels are assigned at the **sequence level**, not per-frame
- Each tfrecord contains ~2GB of data across 8 cameras per frame
- Use the `StandardE2E` library for proper panoramic image extraction
- `data/val_sequence_name_to_scenario_cluster.json` maps sequence IDs to scenario labels

## Resources
- [WOD-E2E Dataset Page](https://waymo.com/open)
- [StandardE2E GitHub](https://github.com/stepankonev/StandardE2E)
- [StandardE2E Docs](https://standarde2e.readthedocs.io/en/latest/)
