#!/usr/bin/env python3
"""
Extract computer-vision features for the Waymo VALIDATION split (DS281 final project).

Run this FROM THE REPO ROOT (same environment used to run the notebooks, so that
`standard_e2e` is importable exactly as it is there). It reuses the EXACT feature
functions from the team's notebooks so validation features are computed identically
to the training features.

Sources (verbatim copies, see the report accompanying this script):
  - notebooks/feature_pipeline_v3_3class.ipynb  (Cell 9: canonical v1 base functions;
      Cell 11: road_v2 + vehicle_occupancy; Cell 24: yolo_lateral_aggregates /
      compute_frame_quantities / QUANTITY_KEYS)
  - notebooks/framediff_v2_3class.ipynb          (Cell 6: optical flow; Cell 10:
      TREND_KEYS / FLOW_KEYS ordering + framediff_v2 extraction logic)
  - notebooks/waymo_feature_extraction.ipynb     (original source of the base
      functions + the road-v1 8-dim assembly and .npz image-loading convention)

Outputs (all row-aligned to a single `seq_ids_val` ordering), written to
`data/processed/waymo_e2e/features/`:
  hog_val.npy, hsv_val.npy, yolo_val.npy, road_val.npy, road_v2_val.npy,
  vehicle_occupancy_val.npy, framediff_v2_val.npy, framediff_v2_valid_val.npy,
  labels_val.npy, seq_ids_val.npy
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import time
from collections import Counter, defaultdict

import numpy as np
import cv2

# --- standard_e2e import ---------------------------------------------------
# The notebooks run from `notebooks/` and simply do `from standard_e2e import Modality`.
# This script runs from the repo root; add a few candidate locations to sys.path so
# the module resolves regardless of where it physically lives in the repo, then import
# it exactly as the notebooks do.  (See report: standard_e2e location was NOT verified
# on disk — if the import fails, add its directory to PYTHONPATH.)
for _cand in ('.', 'notebooks', 'scripts', 'src'):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
from standard_e2e import Modality

from skimage.feature import hog as _skimage_hog
from ultralytics import YOLO


# ===========================================================================
# Paths (relative to repo root) — VALIDATION split
# ===========================================================================
MANIFEST_PATH = 'data/val_manifest.json'
FRAMES_DIR    = 'data/processed/waymo_e2e/val/'          # val .npz frames
FEATURES_DIR  = 'data/processed/waymo_e2e/features/'
os.makedirs(FEATURES_DIR, exist_ok=True)

# The copied notebook loaders (`load_frame`, extraction loops) reference the global
# TRAIN_DIR. To keep those functions byte-for-byte verbatim while loading VAL frames,
# TRAIN_DIR is bound to the val frames directory here.  (Flagged in the report.)
TRAIN_DIR = FRAMES_DIR

CLASSES = ['straight', 'right-turn', 'left-turn']   # 3-class task -> 362 val sequences


# ===========================================================================
# ---- VERBATIM: canonical v1 feature functions ----
# feature_pipeline_v3_3class.ipynb, Cell 9 ("## 3. Canonical v1 feature functions")
# ===========================================================================
HOG_PARAMS = {'orientations': 9, 'pixels_per_cell': (16, 16), 'cells_per_block': (2, 2), 'channel_axis': -1}

def extract_hog(img):
    features, hog_img = _skimage_hog(img, visualize=True, **HOG_PARAMS)
    return features, hog_img


HSV_BINS = (18, 8, 8)  # hue, saturation, value bins

def extract_hsv_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist_h = np.histogram(hsv[:, :, 0], bins=HSV_BINS[0], range=(0, 180), density=True)[0]
    hist_s = np.histogram(hsv[:, :, 1], bins=HSV_BINS[1], range=(0, 256), density=True)[0]
    hist_v = np.histogram(hsv[:, :, 2], bins=HSV_BINS[2], range=(0, 256), density=True)[0]
    return np.concatenate([hist_h, hist_s, hist_v])


DRIVING_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
    5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign',
}

def extract_yolo_features(img, model, driving_classes=DRIVING_CLASSES):
    '''[count, mean_x, mean_y, mean_size] per class, sorted by class id.'''
    H, W = img.shape[:2]
    results = model(img, verbose=False)
    class_detections = defaultdict(list)
    for box in results[0].boxes:
        cls_id = int(box.cls)
        if cls_id in driving_classes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            size = ((x2 - x1) * (y2 - y1)) / (W * H)
            class_detections[cls_id].append((cx, cy, size))
    features = []
    for cls_id in sorted(driving_classes.keys()):
        dets = class_detections[cls_id]
        if len(dets) == 0:
            features.extend([0, 0, 0, 0])
        else:
            features.extend([
                len(dets),
                np.mean([d[0] for d in dets]),
                np.mean([d[1] for d in dets]),
                np.mean([d[2] for d in dets]),
            ])
    return np.array(features, dtype=np.float32)


def detect_edges_and_lines(img, canny_low=30, canny_high=100, hough_threshold=15,
                            min_line_length=20, max_line_gap=15, road_region_frac=0.55):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    road_mask = np.zeros_like(gray); road_mask[int(H * road_region_frac):, :] = 1
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges_masked = edges * road_mask
    lines_raw = cv2.HoughLinesP(edges_masked, rho=1, theta=np.pi / 180, threshold=hough_threshold,
                                 minLineLength=min_line_length, maxLineGap=max_line_gap)
    lines = []
    if lines_raw is not None:
        for line in lines_raw:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if 20 < angle < 75 or 105 < angle < 160:
                lines.append((x1, y1, x2, y2))
    return edges, np.array(lines), road_mask


def estimate_vanishing_point(lines, img_shape):
    H, W = img_shape[:2]
    def line_intersection(l1, l2):
        x1, y1, x2, y2 = l1; x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    if len(lines) < 2:
        return (0.5, 0.5), len(lines), []
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = line_intersection(lines[i], lines[j])
            if pt is not None:
                x, y = pt
                if -W < x < 2 * W and -H < y < 2 * H:
                    intersections.append((x, y))
    if not intersections:
        return (0.5, 0.5), len(lines), []
    xs = np.array([p[0] for p in intersections]); ys = np.array([p[1] for p in intersections])
    x_bins = np.linspace(-W, 2 * W, 30); y_bins = np.linspace(-H, 2 * H, 30)
    hist, xedges, yedges = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
    peak_idx = np.unravel_index(hist.argmax(), hist.shape)
    vp_x = (xedges[peak_idx[0]] + xedges[peak_idx[0] + 1]) / 2
    vp_y = (yedges[peak_idx[1]] + yedges[peak_idx[1] + 1]) / 2
    return (vp_x / W, vp_y / H), len(lines), intersections


def get_robust_seed_color(hsv, H, W):
    seed_points = [
        (W // 2, int(H * 0.95)), (W // 2, int(H * 0.85)),
        (int(W * 0.35), int(H * 0.92)), (int(W * 0.65), int(H * 0.92)),
        (int(W * 0.35), int(H * 0.82)), (int(W * 0.65), int(H * 0.82)),
    ]
    seed_colors, valid_points = [], []
    for sx, sy in seed_points:
        patch = hsv[max(0, sy - 5):sy + 5, max(0, sx - 10):sx + 10]
        if patch.size > 0:
            seed_colors.append(np.mean(patch, axis=(0, 1))); valid_points.append((sx, sy))
    seed_colors = np.array(seed_colors)
    median_color = np.median(seed_colors, axis=0)
    best_idx = np.argmin(np.linalg.norm(seed_colors - median_color, axis=1))
    return seed_colors[best_idx].astype(np.uint8), valid_points[best_idx]


def segment_road(img, road_region_frac=0.55):
    '''Canonical version, returns all 5 keys. Used everywhere in this notebook,
    including inside road_v2, so no downstream function has to recompute
    road_width_bottom / road_width_mid from scratch.'''
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    seed_color, (seed_x, seed_y) = get_robust_seed_color(hsv, H, W)
    tolerance = np.array([20, 60, 60])
    lower = np.clip(seed_color.astype(int) - tolerance, 0, 255).astype(np.uint8)
    upper = np.clip(seed_color.astype(int) + tolerance, 0, 255).astype(np.uint8)
    color_mask = cv2.inRange(hsv, lower, upper)
    road_y_start = int(H * road_region_frac)
    color_mask[:road_y_start, :] = 0
    flood_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
    cv2.floodFill(color_mask.copy(), flood_mask, (seed_x, seed_y), 255,
                  loDiff=(10, 10, 10), upDiff=(10, 10, 10))
    road_mask = (flood_mask[1:-1, 1:-1] * 255).astype(np.uint8)
    road_pixels = np.where(road_mask > 0)
    if len(road_pixels[0]) < 10:
        return road_mask, {
            'road_area_frac': 0.0, 'road_centroid_x': 0.5,
            'road_width_bottom': 0.0, 'road_width_mid': 0.0, 'road_taper': 0.0,
        }
    total_pixels = H * W
    road_area_frac = len(road_pixels[0]) / total_pixels
    road_centroid_x = np.mean(road_pixels[1]) / W
    bottom_row = road_mask[int(H * 0.9), :]; mid_row = road_mask[int(H * 0.7), :]
    road_width_bottom = np.sum(bottom_row > 0) / W
    road_width_mid = np.sum(mid_row > 0) / W
    road_taper = road_width_bottom - road_width_mid
    features = {
        'road_area_frac': road_area_frac, 'road_centroid_x': road_centroid_x,
        'road_width_bottom': road_width_bottom, 'road_width_mid': road_width_mid,
        'road_taper': road_taper,
    }
    return road_mask, features


# ===========================================================================
# ---- VERBATIM: road_v2 and vehicle_occupancy ----
# feature_pipeline_v3_3class.ipynb, Cell 11 ("## 4. road_v2 and vehicle_occupancy")
# ===========================================================================
CENTER_SEG = (128, 256)  # front-center camera segment

def _line_intersection(l1, l2):
    x1, y1, x2, y2 = l1; x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

def _detect_lines_in_roi(gray_center, roi_top_row, canny_low=30, canny_high=100,
                          hough_threshold=12, min_line_length=15, max_line_gap=15):
    H, W = gray_center.shape
    mask = np.zeros_like(gray_center); mask[roi_top_row:, :] = 1
    edges = cv2.Canny(gray_center, canny_low, canny_high)
    edges_masked = edges * mask
    lines_raw = cv2.HoughLinesP(edges_masked, rho=1, theta=np.pi / 180, threshold=hough_threshold,
                                 minLineLength=min_line_length, maxLineGap=max_line_gap)
    lines = []
    if lines_raw is not None:
        for line in lines_raw:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if 20 < angle < 75 or 105 < angle < 160:
                lines.append((x1, y1, x2, y2))
    return np.array(lines)

def _estimate_vp(lines, W, H):
    fallback = (W / 2, H / 2)
    if len(lines) < 2:
        return fallback, []
    intersections, pair_idx = [], []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                x, y = pt
                if -W < x < 2 * W and -H < y < 2 * H:
                    intersections.append((x, y)); pair_idx.append((i, j))
    if not intersections:
        return fallback, []
    xs = np.array([p[0] for p in intersections]); ys = np.array([p[1] for p in intersections])
    hist, xe, ye = np.histogram2d(xs, ys, bins=[np.linspace(-W, 2*W, 30), np.linspace(-H, 2*H, 30)])
    pi = np.unravel_index(hist.argmax(), hist.shape)
    vp_x = (xe[pi[0]] + xe[pi[0]+1]) / 2; vp_y = (ye[pi[1]] + ye[pi[1]+1]) / 2
    return (vp_x, vp_y), list(zip(intersections, pair_idx))

def extract_road_v2(img, inlier_tol_frac=0.15):
    x0, x1 = CENTER_SEG
    center_img = img[:, x0:x1]
    gray = cv2.cvtColor(center_img, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape

    lines_1 = _detect_lines_in_roi(gray, roi_top_row=int(H * 0.55))
    (vp1_x, vp1_y), _ = _estimate_vp(lines_1, W, H)

    margin = int(0.15 * H)
    fixed_top_row = int(H * 0.55)
    vp_informed_top_row = int(np.clip(vp1_y - margin, 0, H - 5))
    roi_top_row_2 = min(fixed_top_row, vp_informed_top_row)
    lines_2 = _detect_lines_in_roi(gray, roi_top_row=roi_top_row_2)
    (vp2_x, vp2_y), intersections_with_idx = _estimate_vp(lines_2, W, H)

    tol = inlier_tol_frac * np.hypot(W, H)
    inlier_line_idx = set()
    for (ix, iy), (i, j) in intersections_with_idx:
        if np.hypot(ix - vp2_x, iy - vp2_y) < tol:
            inlier_line_idx.add(i); inlier_line_idx.add(j)
    n_lines = len(lines_2); n_inliers = len(inlier_line_idx)
    inlier_ratio = n_inliers / n_lines if n_lines > 0 else 0.0

    vp_norm = (vp2_x / W, vp2_y / H)
    return vp_norm, n_lines, n_inliers, inlier_ratio, lines_2, inlier_line_idx, roi_top_row_2

def build_road_v2_features(img):
    '''FIX applied: reuses road_width_bottom/road_width_mid from the canonical
    segment_road() instead of recomputing them (see cell 1 drift note).'''
    vp, n_lines, n_inliers, inlier_ratio, *_ = extract_road_v2(img)
    _, road_feats = segment_road(img)
    return np.array([
        vp[0], vp[1], n_lines, n_inliers, inlier_ratio,
        road_feats['road_area_frac'], road_feats['road_centroid_x'],
        road_feats['road_width_bottom'], road_feats['road_width_mid'],
        road_feats['road_taper'],
    ], dtype=np.float32)

ROAD_V2_DIMS = ['vp_x', 'vp_y', 'n_lines', 'n_inliers', 'inlier_ratio',
                'road_area_frac', 'road_centroid_x', 'road_width_bottom', 'road_width_mid', 'road_taper']


VEHICLE_CLASSES = {2: 'car', 5: 'bus', 7: 'truck'}
OCC_CENTER_SEG = (128, 256)

def extract_vehicle_occupancy(img, yolo_results=None):
    H, W = img.shape[:2]
    x0, x1 = OCC_CENTER_SEG
    seg_w = x1 - x0
    results = yolo_results if yolo_results is not None else yolo_model(img, verbose=False)
    left, right = [], []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        if cls_id not in VEHICLE_CLASSES:
            continue
        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
        cx = (bx1 + bx2) / 2
        if not (x0 <= cx < x1):
            continue
        rel_x = (cx - x0) / seg_w
        dx = rel_x - 0.5
        bottom_y = by2 / H
        if dx < -1/6:
            left.append(bottom_y)
        elif dx > 1/6:
            right.append(bottom_y)
    left_count, right_count = len(left), len(right)
    left_crowding = sum(left) if left else 0.0
    right_crowding = sum(right) if right else 0.0
    left_nearest = max(left) if left else 0.0
    right_nearest = max(right) if right else 0.0
    asymmetry = left_crowding - right_crowding
    return np.array([left_count, left_crowding, left_nearest,
                      right_count, right_crowding, right_nearest, asymmetry], dtype=np.float32)

VEHICLE_OCC_DIMS = ['left_count', 'left_crowding', 'left_nearest',
                     'right_count', 'right_crowding', 'right_nearest', 'asymmetry']


# ===========================================================================
# ---- VERBATIM: per-frame quantity + optical flow (framediff v2) ----
# QUANTITY_KEYS / yolo_lateral_aggregates / compute_frame_quantities:
#   feature_pipeline_v3_3class.ipynb Cell 24  (== framediff_v2_3class.ipynb Cell 8)
# compute_optical_flow + FLOW_KEYS: framediff_v2_3class.ipynb Cell 6
# TREND_KEYS / FLOW ordering + extraction logic: framediff_v2_3class.ipynb Cell 10
# ===========================================================================
QUANTITY_KEYS = ['road_centroid_x', 'road_area_frac', 'vp_x', 'det_mean_x', 'det_lr_balance']

def yolo_lateral_aggregates(img, model, driving_classes=DRIVING_CLASSES):
    H, W = img.shape[:2]
    results = model(img, verbose=False)
    xs = []
    for box in results[0].boxes:
        if int(box.cls) in driving_classes:
            x1, _, x2, _ = box.xyxy[0].tolist()
            xs.append(((x1 + x2) / 2) / W)
    if not xs:
        return {'det_mean_x': 0.5, 'det_lr_balance': 0.0}
    xs = np.array(xs)
    left = (xs < 0.5).sum(); right = (xs >= 0.5).sum()
    return {'det_mean_x': float(xs.mean()), 'det_lr_balance': float((left - right) / len(xs))}

def compute_frame_quantities(img, model):
    _, road = segment_road(img)
    _, lines, _ = detect_edges_and_lines(img)
    (vp_x, _vp_y), _n, _ = estimate_vanishing_point(lines, img.shape)
    det = yolo_lateral_aggregates(img, model)
    return {
        'road_centroid_x': road['road_centroid_x'], 'road_area_frac': road['road_area_frac'],
        'vp_x': vp_x, 'det_mean_x': det['det_mean_x'], 'det_lr_balance': det['det_lr_balance'],
    }


def compute_optical_flow(img_prev, img_curr, center_seg=CENTER_SEG,
                          road_region_frac=0.55):
    """
    Dense Farneback optical flow between two consecutive RGB frames.
    Restricted to front-center crop + road ROI (bottom 45%).

    Returns dict of 6 flow scalars.
    """
    x0, x1 = center_seg
    prev_crop = img_prev[:, x0:x1, :]
    curr_crop = img_curr[:, x0:x1, :]
    H, W = prev_crop.shape[:2]

    prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_crop, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )  # shape: (H, W, 2)

    road_y      = int(H * road_region_frac)
    flow_road   = flow[road_y:, :]           # bottom 45%
    flow_bottom = flow[int(H * 0.80):, :]    # bottom 20%

    mag_road   = np.sqrt(flow_road[..., 0]**2   + flow_road[..., 1]**2)
    mag_bottom = np.sqrt(flow_bottom[..., 0]**2 + flow_bottom[..., 1]**2)

    mid = W // 2
    mag_left  = mag_road[:, :mid].mean()
    mag_right = mag_road[:, mid:].mean()
    lr_balance = (mag_left - mag_right) / (mag_left + mag_right + 1e-6)

    return {
        'flow_mag_mean':   float(mag_road.mean()),
        'flow_mag_p75':    float(np.percentile(mag_road, 75)),
        'flow_lr_balance': float(lr_balance),
        'flow_x_mean':     float(flow_road[..., 0].mean()) / W,
        'flow_y_mean':     float(flow_road[..., 1].mean()) / H,
        'flow_mag_bottom': float(mag_bottom.mean()),
    }


FLOW_KEYS = [
    'flow_mag_mean', 'flow_mag_p75', 'flow_lr_balance',
    'flow_x_mean', 'flow_y_mean', 'flow_mag_bottom',
]

# TREND_KEYS / flow feature-key ordering (framediff_v2_3class.ipynb Cell 10)
TREND_KEYS          = ([f'slope_{k}' for k in QUANTITY_KEYS] +
                       [f'last_{k}'  for k in QUANTITY_KEYS])
MEAN_FLOW_KEYS      = [f'mean_{k}' for k in FLOW_KEYS]
LAST_FLOW_KEYS      = [f'last_{k}' for k in FLOW_KEYS]
ALL_FLOW_FEATURE_KEYS = MEAN_FLOW_KEYS + LAST_FLOW_KEYS
ALL_FEATURE_KEYS    = TREND_KEYS + ALL_FLOW_FEATURE_KEYS


def load_frame(fname):
    # VERBATIM loader (framediff_v2_3class.ipynb Cell 8); TRAIN_DIR is bound to the
    # val frames directory at the top of this script.
    data = np.load(os.path.join(TRAIN_DIR, fname), allow_pickle=True)
    return np.array(data['_modality_data'].item()[Modality.CAMERAS])


# ===========================================================================
# YOLO model — same weights + call convention as the notebooks
# ===========================================================================
# Use the repo's existing weights (notebooks/yolov8s.pt) rather than triggering an
# auto-download, so the val detections use the exact same weights as training.
_YOLO_WEIGHTS = 'notebooks/yolov8s.pt' if os.path.exists('notebooks/yolov8s.pt') else 'yolov8s.pt'
yolo_model = YOLO(_YOLO_WEIGHTS)


# ===========================================================================
# Main extraction
# ===========================================================================
def main():
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    # Filter to the 3 classes, preserving manifest order.
    entries = [(sid, e) for sid, e in manifest.items() if e['label'] in CLASSES]
    print(f'Loaded {len(manifest)} val sequences; {len(entries)} in classes {CLASSES}')

    hog_list, hsv_list, yolo_list, road_list = [], [], [], []
    road_v2_list, vocc_list = [], []
    framediff_list, valid_list = [], []
    labels_list, seq_ids_list = [], []

    t0 = time.time()
    print('NOTE: the per-context-frame quantity + optical-flow (framediff_v2) work is the '
          'slow part (segment_road + Hough + YOLO + dense Farneback per context frame). '
          'For 362 sequences expect roughly 15-25 minutes. Run under `caffeinate -i` on macOS.')

    for i, (sid, entry) in enumerate(entries):
        # seq_id = target .npz filename with trailing _<frame>.npz stripped
        target_fname = entry['target_fname']
        seq_id = os.path.splitext(target_fname)[0].rsplit('_', 1)[0]

        # ---- target-frame image ----
        img = load_frame(target_fname)

        # HOG
        hog_feat, _ = extract_hog(img)
        # HSV histogram
        hsv_feat = extract_hsv_histogram(img)
        # YOLO + position
        yolo_feat = extract_yolo_features(img, yolo_model)

        # road geometry v1 (8-dim) — assembled exactly as waymo_feature_extraction Cell 14
        edges, lines, _road_mask_bin = detect_edges_and_lines(img)
        vp, n_lines, _ = estimate_vanishing_point(lines, img.shape)
        _, road_features = segment_road(img)
        road_feat = np.array([
            vp[0],
            vp[1],
            n_lines,
            road_features['road_area_frac'],
            road_features['road_centroid_x'],
            road_features['road_width_bottom'],
            road_features['road_width_mid'],
            road_features['road_taper'],
        ], dtype=np.float32)

        # road_v2 (10-dim)
        road_v2_feat = build_road_v2_features(img)

        # vehicle occupancy (7-dim) — pass a single YOLO pass, as in Cell 14
        yolo_results = yolo_model(img, verbose=False)
        vocc_feat = extract_vehicle_occupancy(img, yolo_results=yolo_results)

        # ---- framediff_v2 (22-dim: 10 trend + 12 flow) ----
        # VERBATIM logic from framediff_v2_3class.ipynb Cell 10.
        fnames = entry['context_fnames']   # oldest -> newest; last == target frame
        n      = len(fnames)
        imgs   = [load_frame(fn) for fn in fnames]

        # Trend
        Qs = np.array([[compute_frame_quantities(im, yolo_model)[k]
                        for k in QUANTITY_KEYS]
                       for im in imgs])
        if n >= 2:
            t_idx = np.arange(n); t_c = t_idx - t_idx.mean()
            slopes = ((t_c[:, None] * (Qs - Qs.mean(0))).sum(0) / (t_c**2).sum())
        else:
            slopes = np.zeros(len(QUANTITY_KEYS))
        trend_feat = np.concatenate([slopes, Qs[-1]])

        # Flow
        if n >= 2:
            flow_pairs = np.array([
                [compute_optical_flow(imgs[j], imgs[j+1])[k] for k in FLOW_KEYS]
                for j in range(n - 1)
            ])
            mean_flow = flow_pairs.mean(axis=0)
            last_flow = flow_pairs[-1]
            valid = True
        else:
            mean_flow = last_flow = np.zeros(len(FLOW_KEYS))
            valid = False

        framediff_feat = np.concatenate([trend_feat, mean_flow, last_flow]).astype(np.float32)

        # ---- collect (all row-aligned to this single ordering) ----
        hog_list.append(hog_feat)
        hsv_list.append(hsv_feat)
        yolo_list.append(yolo_feat)
        road_list.append(road_feat)
        road_v2_list.append(road_v2_feat)
        vocc_list.append(vocc_feat)
        framediff_list.append(framediff_feat)
        valid_list.append(valid)
        labels_list.append(entry['label'])
        seq_ids_list.append(seq_id)

        if (i + 1) % 25 == 0 or (i + 1) == len(entries):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(entries) - (i + 1))
            print(f'  {i+1}/{len(entries)}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)')

    hog_val          = np.array(hog_list)
    hsv_val          = np.array(hsv_list)
    yolo_val         = np.array(yolo_list, dtype=np.float32)
    road_val         = np.array(road_list, dtype=np.float32)
    road_v2_val      = np.array(road_v2_list, dtype=np.float32)
    vehicle_occ_val  = np.array(vocc_list, dtype=np.float32)
    framediff_v2_val = np.array(framediff_list, dtype=np.float32)
    valid_val        = np.array(valid_list)
    labels_val       = np.array(labels_list)
    seq_ids_val      = np.array(seq_ids_list)

    np.save(os.path.join(FEATURES_DIR, 'hog_val.npy'), hog_val)
    np.save(os.path.join(FEATURES_DIR, 'hsv_val.npy'), hsv_val)
    np.save(os.path.join(FEATURES_DIR, 'yolo_val.npy'), yolo_val)
    np.save(os.path.join(FEATURES_DIR, 'road_val.npy'), road_val)
    np.save(os.path.join(FEATURES_DIR, 'road_v2_val.npy'), road_v2_val)
    np.save(os.path.join(FEATURES_DIR, 'vehicle_occupancy_val.npy'), vehicle_occ_val)
    np.save(os.path.join(FEATURES_DIR, 'framediff_v2_val.npy'), framediff_v2_val)
    np.save(os.path.join(FEATURES_DIR, 'framediff_v2_valid_val.npy'), valid_val)
    np.save(os.path.join(FEATURES_DIR, 'labels_val.npy'), labels_val)
    np.save(os.path.join(FEATURES_DIR, 'seq_ids_val.npy'), seq_ids_val)

    print(f'\nDone in {time.time()-t0:.0f}s. Saved to {FEATURES_DIR}')
    print('Shape summary (all row-aligned to seq_ids_val):')
    print(f'  hog_val:              {hog_val.shape}')
    print(f'  hsv_val:              {hsv_val.shape}')
    print(f'  yolo_val:             {yolo_val.shape}')
    print(f'  road_val:             {road_val.shape}')
    print(f'  road_v2_val:          {road_v2_val.shape}')
    print(f'  vehicle_occupancy_val:{vehicle_occ_val.shape}')
    print(f'  framediff_v2_val:     {framediff_v2_val.shape}')
    print(f'  framediff_v2_valid_val:{valid_val.shape}  '
          f'(valid={int(valid_val.sum())}/{len(valid_val)})')
    print(f'  labels_val:           {labels_val.shape}  {Counter(labels_val.tolist())}')
    print(f'  seq_ids_val:          {seq_ids_val.shape}')


if __name__ == '__main__':
    main()
