# DATASCI 281 — Section 2, Group 2 Final Project

**Task:** Multiclass classification of long-tail driving scenarios using panoramic camera frames  
**Dataset:** Waymo Open Dataset – End-to-End Driving (WOD-E2E, CVPR 2026)  
**Paper:** Xu et al., "WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios"

## Classes (10)
Construction, Cut-ins, Cyclist, Foreign Object Debris, Intersections,
Multi-Lane Maneuvers, Others, Pedestrian, Single-Lane Maneuvers, Special Vehicles

## Key Dataset Facts
- **479 labeled val sequences**, one label per sequence (segment-level, not per-frame)
- **8 cameras per frame**: FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT, REAR, REAR_LEFT, REAR_RIGHT
- **93 val shards** (~2.6 GB each) — sequences are distributed non-contiguously across all shards
- Labels are assigned by automated mining — discriminative content is sparse within each sequence
- Waymo's own baseline concatenates all 8 cameras into a single 768×768 panoramic image

## Setup

### 1. Get dataset access
Register at https://waymo.com/open and request access to the WOD-E2E dataset.
Once approved, authenticate:

```bash
brew install --cask google-cloud-sdk
gcloud auth login                       # use your Waymo-registered Google account
gcloud auth application-default login  # needed for gsutil
```

### 2. Clone this repo
```bash
git clone https://github.com/dawsofd/281-s2-group2-final.git
cd 281-s2-group2-final
```

### 3. Create the conda environment
```bash
CONDA_SOLVER=classic conda env create -f environment.yml
conda activate 281-s2-group2
pip install standard-e2e --no-deps
pip install "numpy<2"   # pin back after av2 may upgrade it
```

### 4. Download the labels JSON and one val shard (for EDA)
```bash
gsutil cp gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/val_sequence_name_to_scenario_cluster.json data/
gsutil cp gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/val_202504211843.tfrecord-00000-of-00093 .
```

The tfrecord (~2.6 GB) lives in the **repo root** (gitignored). The labels JSON is in `data/` and tracked in git.

### 5. Run the EDA notebook
Open `notebooks/eda_wod_e2e.ipynb` in VS Code and select the `281-s2-group2` kernel.

### 6. Full feature extraction (one-time, run by Dawson)
Streams all 93 val shards from GCS, stitches 8 cameras → 384px panorama, writes cache:
```bash
python -m standard_e2e.caching.process_source_dataset waymo_e2e \
  --input_path=gs://waymo_open_dataset_end_to_end_camera_v_1_0_0 \
  --output_path=./data/processed \
  --split=val \
  --num_workers=1 \
  --config_file=configs/waymo_e2e_pano.yaml
```
Output: `data/processed/waymo_e2e/val/` — one `.npz` per frame + parquet index.
CLIP/DINOv2 embeddings are then extracted and saved to `data/embeddings.parquet` (~1 MB, committed to repo).
**Teammates work from `data/embeddings.parquet` only — no raw data access needed.**

## Resources
- [WOD-E2E Dataset Page](https://waymo.com/open/data/e2e/)
- [WOD-E2E Paper (CVPR 2026)](https://arxiv.org/abs/2510.26125)
- [StandardE2E GitHub](https://github.com/stepankonev/StandardE2E)
- [StandardE2E Docs](https://standarde2e.readthedocs.io/en/latest/)
- [Waymo proto definition](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/end_to_end_driving_data.proto)