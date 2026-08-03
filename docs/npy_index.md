# Canonical `.npy` Index — DS281 Waymo Maneuver Prediction

Single reference for which feature arrays are current, what they contain, and how they
align. If you are building or analyzing features, start here.

## The one rule

**Everything joins on `data/cnn_2d_seq_fold.csv`.** It is the single source of truth for
row order, labels, and the train/val/test split — 1,966 pooled sequences.

| column | meaning |
|---|---|
| `cnn_row` | canonical row order (== `CNN_2D.npy` / `CNN_3D.npy` row order) |
| `seq_id` | join key for every feature family |
| `fold` | `train` (1376) / `val` (295) / `test` (295) — the split we actually score on |
| `class_label` | `straight` / `right-turn` / `left-turn` (use this, not the legacy `labels_*.npy`) |

Load features aligned to it with `scripts/bench_common.py` (3 lines) instead of touching
raw `seq_ids` files:

```python
import bench_common as bc
root, FD, FOLD_CSV = bc.resolve_paths()
FOLD, ORDER, y, masks = bc.load_fold(FOLD_CSV)
FAM = bc.assemble_families(FD, ORDER, fold=FOLD)   # every current family, aligned to the fold
```

## Why row counts differ (the four index spaces)

| rows | what it is | join key |
|---|---|---|
| **2037** | original feature-extraction pool | `seq_ids.npy` |
| **1604** | 3-class training subset | `seq_ids_3class.npy` (or a family-specific `*_seq_ids.npy`) |
| **362** | held-out arrays that top up the pool | `seq_ids_val.npy` |
| **1966** | pooled set = 1604 + 362 = the fold | `cnn_2d_seq_fold.csv` (`cnn_row` order) |

**Important:** the `*_val.npy` files are **not** the fold's validation set. They are just the
*second source file* needed so the two arrays together cover all 1,966 sequences.
`bench_common` merges each family's train array + `*_val` array by `seq_id`, then lays the
result out in fold order. The fold column — not the file name — defines train/val/test.

## Canonical feature families (CURRENT — used in the benchmark)

Each family = a train array + its index + a `*_val` top-up array. `bench_common` merges them.

| Family | Rubric type | Dims | Train array | Train index | Val top-up |
|---|---|---|---|---|---|
| **HOG** | simple | 5796 | `hog.npy` (2037) | `seq_ids.npy` | `hog_val.npy` (362) |
| **HSV histogram** | simple | 34 | `hsv.npy` (2037) | `seq_ids.npy` | `hsv_val.npy` |
| **YOLO + position** | complex | 32 | `yolo.npy` (2037) | `seq_ids.npy` | `yolo_val.npy` |
| **Road geometry v2** | complex | 10 | `road_v2_3class.npy` (1604) | `seq_ids_3class.npy` | `road_v2_val.npy` |
| **Vehicle occupancy** | complex | 7 | `vehicle_occupancy_3class.npy` (1604) | `seq_ids_3class.npy` | `vehicle_occupancy_val.npy` |
| **Trend + Flow** (temporal) | complex | 22 | `framediff_v2_3class.npy` (1604) | `framediff_v2_3class_seq_ids.npy` | `framediff_v2_val.npy` |

### Learned embeddings (already pooled, 1966 rows)

| Family | Dims | Array(s) | Alignment | Type |
|---|---|---|---|---|
| **MoViNet-A0 (frozen)** | 480 | `CNN_3D_v2/embedding_cache/movinet_a0_stream_{train,val,test}_embeddings.npy` (1376/295/295) | via fold `fold` split, `cnn_row` order within split | **frozen — leakage-free** ← headline learned feature |
| 2D-CNN penultimate | 64 | `CNN_2D.npy` (1966) | already in `cnn_row` order | trained head — *in-sample-train caveat* |
| 3D-CNN penultimate | 128 | `CNN_3D.npy` (1966) | already in `cnn_row` order | trained head — *in-sample-train caveat* |
| DINOv2 | 384 | `dino_v2_3class.npy` (1604) | `dino_v2_3class_seq_ids.npy` | frozen; **train-only, no val array** |

> **Frozen vs. trained-head:** frozen embeddings (MoViNet, DINOv2) never saw the labels and
> are clean to stack. The `CNN_2D/CNN_3D` penultimate arrays come from heads trained on the
> train fold, so treat them as ablations, not headline features.

## Trend + Flow internals

`framediff_v2_3class.npy` (22 dims) = `framediff_v2_trend_3c.npy` (10, trend) ⧺
`framediff_v2_flow_3c.npy` (12, optical flow). Column order is **trend (0:10) then flow (10:22)**.

## Index / label files

| File | Rows | Use |
|---|---|---|
| `data/cnn_2d_seq_fold.csv` | 1966 | **canonical** — order, labels, split |
| `seq_ids.npy` | 2037 | index for HOG / HSV / YOLO |
| `seq_ids_3class.npy` | 1604 | index for road v2 / vehicle occupancy |
| `framediff_v2_3class_seq_ids.npy` | 1604 | index for Trend+Flow |
| `seq_ids_val.npy` | 362 | index for every `*_val.npy` |
| `labels_CNN_2D.npy` / `labels_CNN_3D.npy` | 1966 | legacy; prefer fold `class_label` |

## Archived / experimental — do NOT use for current results

Kept for provenance only; none feed the current benchmark.

- **Older non-3-class `*_v2` set (1405 rows):** `hog_v2`, `hsv_v2`, `yolo_v2`, `road_v2`,
  `seq_ids_v2`, `labels_v2` — earlier day/night 5-class extraction.
- **Original non-3-class arrays (2037):** `road.npy`, `vehicle_occupancy.npy`,
  `cnn_embedding.npy` (512), `vit_embedding.npy` (768), `mobilenetv2_embedding.npy` (1280),
  `labels.npy` — superseded frozen backbones / pre-pivot labels.
- **Earlier temporal versions:** `framediff.npy`, `framediff_trend*`, `framediff_v2_3c_aligned.npy`,
  `framediff_v2_peak_flow*` — intermediates; `framediff_v2_3class.npy` is the final Trend+Flow.
- **Experimental geometry / scene features:** `corridor*`, `bev_corridor*`, `ipm_3class*`,
  `drivable_3class*`, `scene_cue_*`, `traffic_law_3class*`, `seg_*`, `curve_pavement.npy` —
  explored, not in the final feature set.
- **Stacked / superseded:** `X_final_3class.npy` (pre-built HOG matrix),
  `labels_final_3class.npy`, `seq_ids_final_3class.npy`.
