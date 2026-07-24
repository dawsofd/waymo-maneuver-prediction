# Archive — Previous Attempts

This folder preserves earlier feature-set and modeling work that is **no longer part of
the active pipeline** but is retained for provenance, grading, and in case any thread is
revisited. The current pipeline lives in [`../notebooks/`](../notebooks/); see the
top-level [README](../README.md) for the final 3-class results.

**Note on outputs:** the archived notebooks have had their embedded cell outputs stripped
to keep the repo small (they collectively held ~28 MB of rendered images). The code and
markdown are intact; re-run a notebook if you need its figures back.

---

## What was set aside, and why

### `initialExploration_Matt.ipynb`
The earliest exploration of the WOD-E2E data — inspecting the contents of a single `.npz`
file, discovering the `{hash}_{frame}` naming scheme, locating class-label data, and
building a first GIF of a sequence. Superseded by `train_inventory.ipynb` (kept) for
dataset structure and by the feature-extraction pipeline for everything downstream.
**Role:** orientation only; no features or results carried forward.

### `feature_exploration_update.ipynb`  (5-class project update)
The mid-project update deliverable, on the original **5-class** task. Contents:
per-class feature-visualization panels; t-SNE and UMAP projections alongside PCA; and a
first pass at learned embeddings — a dense CNN embedding from the YOLOv8s backbone, plus
**MobileNetV2** and **ViT** embeddings — benchmarked with SVM / RF.
**Why archived:** the task moved to 3-class, and the learned-embedding results did not beat
classical features (CNN ≈ 0.52, MobileNetV2 ≈ 0.54, ViT ≈ 0.53 accuracy, all below HOG's
~0.60). The genuinely useful pieces — the `road_v2` / vehicle-occupancy feature
definitions and the embedding arrays themselves — were **ported into**
`feature_pipeline_v3_3class.ipynb` (the embeddings are re-aligned to the 3-class `seq_ids`
and included as benchmark candidates), so nothing needed here is lost.

### `framediff.ipynb`  (v1 temporal feature)
The first frame-difference / temporal feature, built specifically to chase
**lane-change** separability on the 5-class task. Computed lateral-difference quantities
(`road_centroid_x`, `det_mean_x`, `det_lr_balance`) between context frames.
**Why archived:** directly superseded by `framediff_v2_3class.ipynb`, which retargets the
idea to 3-class, adds optical-flow magnitude/direction, handles `n_context == 1`
gracefully, and produces the **winning Trend+Flow set (RF, 0.648 acc / 0.587 macro-F1)**.

### `filter_day_night.ipynb`  (abandoned lighting-filter direction)
Classified each sequence as day or night from first-frame luminance (threshold 0.30) and
built a daytime-only, relabeled dataset. Part of an early direction that also considered
switching to pedestrian/cyclist classes.
**Why archived:** the final pipeline keeps **both day and night** sequences (no lighting
filter). The day/night classifier function it defined was reused briefly by
`Waymo_feature_extraction_v2.ipynb` (also archived) but is not used by the final pipeline.

### `Waymo_feature_extraction_v2.ipynb`  (daytime-only comparison)
A careful non-destructive comparison of full vs. daytime-only training: it subset the
existing feature arrays to daytime rows (saved with a `_v2` suffix so nothing was
overwritten) and evaluated whether daytime-only training improved per-class recall using
out-of-fold predictions from a balanced linear SVM.
**Why archived:** the analysis did not justify restricting to daytime, so the direction was
dropped. The `_v2` daytime-only arrays it produced (`hog_v2.npy`, etc.) are the abandoned
feature version; see the repo-hygiene note below.

### `segment_road_features.ipynb`  (road-segmentation development)
Development notebook for the road-segmentation / drivable-area feature work.
**Why archived:** the road-geometry logic was consolidated into the canonical
`segment_road` functions in `feature_pipeline_v3_3class.ipynb` (called out there as the
single source of truth to avoid copy-drift). Kept for reference on how the segmentation
was developed.

### `class_update_slides.pptx`
The earlier class-update slide deck. Superseded by `slides/class_update_slides_v3.pptx`.

---

## Repo-hygiene note — the `_v2` feature arrays

The abandoned daytime-only direction left a set of `*_v2.npy` feature arrays tracked in
git under `data/processed/waymo_e2e/features/` — most notably **`hog_v2.npy` (~63 MB)**,
which is the single largest tracked file in the repo and the main reason `.git` is large.
These arrays belong to this archived direction, not the final pipeline. They can be safely
untracked from git (they remain on disk / on the team Drive and are regenerable). See the
top-level cleanup notes / `git rm --cached` commands provided during cleanup.
