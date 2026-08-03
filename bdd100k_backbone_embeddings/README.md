# BDD100K-pretrained backbone embeddings (experiment)

**Status:** new, standalone experiment. Nothing in the existing repo (notebooks, scripts,
or `data/`) was modified to build this — it only *reads* `data/cnn_2d_seq_fold.csv` (the
team's canonical fold file — see "Canonical .npy Index") and the existing feature `.npy`
files, and *writes* two new arrays plus a benchmark CSV into
`data/processed/waymo_e2e/features/`.

**Data model note:** this was originally built against an older pattern (deriving the
3-class subset from `train_manifest.json` + `labels.npy`). It's since been reworked to match
the team's current canonical index — everything now joins on `data/cnn_2d_seq_fold.csv`
(`seq_id`, `source`, `class_label`, `target_frame_path`), and `labels.npy`/`road.npy` (now
archived/superseded per that index) are no longer used. See "Data model" below for specifics.

## Hypothesis

The DINOv2 embedding in `notebooks/feature_pipeline_v3_3class.ipynb` (macro-F1 ≈ 0.565) is a
generic ImageNet-pretrained backbone. [BDD100K](https://doc.bdd100k.com/) is 100K driving
images with scene/weather/time-of-day labels, and the
[SysCV/bdd100k-models](https://github.com/SysCV/bdd100k-models) model zoo publishes
classification backbones trained on it. A backbone trained on driving scenes specifically
(rather than generic ImageNet classes) might produce embeddings more useful for maneuver
classification.

## What this does

Same template as the DINOv2 cell in `feature_pipeline_v3_3class.ipynb`:

1. Loads a **ResNet-50** backbone trained on BDD100K **scene tagging** (tunnel / residential /
   parking lot / city street / gas station / highway), from the SysCV BDD100K model zoo.
2. Strips the classification head, keeping only the backbone + global-average-pool
   (2048-d penultimate feature).
3. Runs it over the same **CENTER_SEG = (128, 256)** crop of each sequence's target frame
   used everywhere else in the pipeline.
4. Saves `bdd100k_resnet50_scene_3class.npy` (+ companion `..._seq_ids.npy`) into
   `data/processed/waymo_e2e/features/`.
5. Benchmarks it (alone and combined with HOG / HSV / YOLO / Road_v2 / VehicleOcc /
   Trend+Flow / DINOv2 — whichever of those are actually present locally) with the same
   SVM/RF setup as the DINOv2 benchmark cell, and saves
   `benchmark_bdd100k_resnet50_scene_3class.csv` next to the other benchmark CSVs.

## Data model

Per the team's **Canonical .npy Index**, everything joins on `data/cnn_2d_seq_fold.csv`
(1,966 pooled sequences; columns `cnn_row`, `seq_id`, `fold`, `source`, `class_label`,
`target_frame_path`). This notebook only covers the **`source == 'training'`** rows — the
1,604-row 3-class training pool (exactly `seq_ids_3class.npy`) — the same scope DINOv2 has
in the index ("frozen; train-only, no val array"), since that's the only slice with per-frame
images available to run a fresh backbone over. `class_label` and `target_frame_path` come
straight from the fold file, not from `train_manifest.json`/`labels.npy` (both archived).

**HOG is required**; every other comparison family (HSV, YOLO, Road_v2, VehicleOcc,
Trend+Flow, DINOv2) is optional and silently skipped (with a printed note) if its `.npy`
isn't present locally yet — each is joined to the notebook's own `seq_ids_3c` by `seq_id`
via that family's own index file (e.g. `road_v2_3class.npy` + `seq_ids_3class.npy`), never
assumed to already be in the same row order.

## Source model

- Repo: https://github.com/SysCV/bdd100k-models (`tagging/` task)
- Config: [`tagging/configs/scene/resnet50_5x_224x224_scene_tag_bdd100k.py`](https://github.com/SysCV/bdd100k-models/blob/main/tagging/configs/scene/resnet50_5x_224x224_scene_tag_bdd100k.py)
  → standard `mmcls` `ResNet` backbone, `depth=50`, `style="pytorch"` (no deep-stem, no
  avg-down) + `GlobalAveragePooling` neck + `LinearClsHead(2048 → 7)`.
- Checkpoint: `https://dl.cv.ethz.ch/bdd100k/tagging/scene/models/resnet50_5x_224x224_scene_tag_bdd100k.pth`
  (val accuracy 77.66%, per the repo's model zoo table)
- **That host is currently down** (DNS NXDOMAIN against multiple public resolvers; matches
  the still-open [SysCV/bdd100k-models#28](https://github.com/SysCV/bdd100k-models/issues/28),
  filed 2026-03-25). The checkpoint in `checkpoints/` here was instead pulled from the
  Internet Archive's Wayback Machine capture of the same file from 2025-01-29, before the
  host died — see the notebook's download cell for the exact fallback URL.

**Why no `mmcls`/`mmcv` dependency:** the config confirms this is a plain, non-deep-stem
`style="pytorch"` ResNet-50, which `mmcls` deliberately keeps parameter-name-compatible
with `torchvision.models.resnet50`. So the notebook loads the checkpoint's `backbone.*`
weights directly into a `torchvision` ResNet-50 (strip the `backbone.` prefix, drop `fc`),
instead of pulling in `mmcv-full`/`mmcls` as new project dependencies — `torch`/`torchvision`
are already in `environment.yml`. The notebook prints the matched/missing key counts on load
so this assumption is verifiable at run time, not just asserted.

Preprocessing matches the model zoo's `test_pipeline` for this config: resize short side to
256, center-crop 224, normalize with `mean=[123.675,116.28,103.53]`,
`std=[58.395,57.12,57.375]` (standard ImageNet stats).

## Verification status

The `281-s2-group2` conda env now exists on this machine (via `environment.yml`, per the
main README's "Environment Setup"), and everything in this notebook **except the real
benchmark numbers** has been run for real, not just inferred:

- `dl.cv.ethz.ch` (the model zoo's checkpoint host) confirmed down repo-wide via DNS lookups
  against three independent resolvers (system, 8.8.8.8, 1.1.1.1 — all NXDOMAIN) and a
  matching, still-open upstream issue:
  [SysCV/bdd100k-models#28](https://github.com/SysCV/bdd100k-models/issues/28).
- Working fallback: the Internet Archive's Wayback Machine crawled this exact file on
  2025-01-29, before the host died. Downloaded (94,411,924 bytes, matching the original
  host's pre-outage `Content-Length`) into `checkpoints/` here, so running the notebook
  needs no download.
- **`torch.load` + `load_state_dict` run for real**, in the actual `281-s2-group2` env,
  against the downloaded checkpoint: `318` backbone tensors loaded, only `['fc.weight',
  'fc.bias']` reported missing, `0` unexpected keys, and a forward pass on a dummy
  `(1, 3, 224, 224)` input produced the expected `(1, 2048)` embedding. This confirms the
  no-`mmcls` loading approach (strip the `backbone.` prefix, load into
  `torchvision.models.resnet50`, drop `fc`) is correct for this exact checkpoint file, not
  just plausible from the config.
- **The whole notebook was executed end-to-end** (`jupyter nbconvert --execute`), twice,
  against synthetic stand-in project trees (9 fake sequences: 3 per class, random-noise
  images, random feature arrays matching the canonical schema) built and run outside this
  repo, purely to catch code bugs — every cell ran with zero errors both times: fold-file
  loading, the required-HOG check, the optional-family joins, checkpoint load, the
  extraction loop (produced a `(9, 2048)` embedding array), and the benchmark cell (ran
  SVM/RF, saved a CSV). One run omitted the fake `hog.npy` to confirm the required-family
  check raises a clear `FileNotFoundError` instead of silently producing an incomplete
  benchmark. The benchmark *numbers* from these runs are meaningless (random noise, random
  labels) — only the mechanics were being checked.

**What genuinely still needs the real dataset** — not available on this machine, and not
something I have credentials for:
1. `data/processed/waymo_e2e/training/` — the real Waymo per-frame `.npz` files. This is the
   one thing none of the pre-extracted `.npy` families can substitute for (they're numeric
   summaries of the pixels, not the pixels), and it's a separate, much larger fetch (~13GB
   via the Waymo GCS pipeline, per the main README) than the features folder. Cell 9 (the
   extraction loop) will fail with `FileNotFoundError` on the first frame until this exists.
2. `hog.npy` in `data/processed/waymo_e2e/features/` — everything else the canonical index
   lists as current was present as of this write-up (confirmed by directly listing the
   folder); HOG was the one gap, and cell 3 now hard-requires it rather than silently
   skipping it, since the plan is to pull it from the team Drive rather than treat it as
   permanently optional.

No official MD5 could be cross-checked for the checkpoint — the `.md5` file itself was never
archived by the Wayback Machine — so there's no cryptographic provenance guarantee beyond the
structural key-match above and the exact byte-count match against the pre-outage
`Content-Length`.

## Run

```
conda activate 281-s2-group2
jupyter notebook bdd100k_resnet50_scene_embedding.ipynb
```

(The `281-s2-group2` kernel is already registered with Jupyter on this machine via
`python -m ipykernel install --user --name 281-s2-group2`.)

Requires `data/cnn_2d_seq_fold.csv` (already present/git-tracked) and, under
`data/processed/waymo_e2e/`: `training/` (the raw `.npz` frames — still missing on this
machine) and `features/hog.npy` (still missing on this machine; everything else the
canonical index lists as current is already present in `features/`).
