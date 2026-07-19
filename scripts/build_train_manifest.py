#!/usr/bin/env python3
"""
build_train_manifest.py

Builds a manifest of sequence-level maneuver labels and target frames
from the Waymo E2E training set.

For each sequence:
  - Classifies each frame's future trajectory into a maneuver type
  - Assigns the sequence label as the highest-ranked maneuver
  - Selects the target frame as the frame immediately before the best maneuver frame
  - Stores up to MAX_CONTEXT_FRAMES preceding frames for optional multi-frame modeling

Maneuver classification is a port of Waymo's own ClassifyTrack
(waymo_open_dataset/metrics/motion_metrics_utils.cc), with two adaptations:
  (1) heading_diff and max_speed are reconstructed from X/Y/TIMESTAMP because the
      future states in WOD-E2E do not populate HEADING or VELOCITY.
  (2) u-turns are collapsed into their turn direction, since this task has no
      u-turn class and true u-turns are absent from these forward-driving clips.

Output: data/train_manifest.json

Usage:
    python scripts/build_train_manifest.py
    python scripts/build_train_manifest.py --sample 200  # for testing
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict, Counter
from standard_e2e import Modality, TrajectoryComponent

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

TRAIN_DIR = 'data/processed/waymo_e2e/training/'
OUTPUT_PATH = 'data/train_manifest.json'
MAX_CONTEXT_FRAMES = 20

MANEUVER_RANK = {
    'left-turn': 5,
    'right-turn': 5,
    'lane-change-left': 4,
    'lane-change-right': 4,
    'straight': 3,
    'stationary': 1,
}

# --- Waymo ClassifyTrack thresholds (motion_metrics_utils.cc) ---
MAX_SPEED_FOR_STATIONARY = 2.0                  # m/s
MAX_DISPLACEMENT_FOR_STATIONARY = 3.0           # m
MAX_LATERAL_DISPLACEMENT_FOR_STRAIGHT = 2.5     # m
MAX_ABS_HEADING_DIFF_FOR_STRAIGHT = np.pi / 6.0  # 30 degrees (rad)

# Adaptation parameter (not from Waymo): minimum segment displacement (m)
# required before a motion direction is trusted as a heading. Guards against
# stationary jitter producing garbage headings.
MIN_DISP_FOR_HEADING = 0.5  # m


def _heading_from_ends(xs, ys):
    """Start/end heading via displacement-gated tangents. (None, None) if the
    trajectory never moves >= MIN_DISP_FOR_HEADING from either endpoint."""
    pts = np.column_stack([xs, ys])
    start_h = None
    for i in range(1, len(pts)):
        if np.hypot(*(pts[i] - pts[0])) >= MIN_DISP_FOR_HEADING:
            start_h = np.arctan2(pts[i][1] - pts[0][1], pts[i][0] - pts[0][0])
            break
    end_h = None
    for i in range(len(pts) - 2, -1, -1):
        if np.hypot(*(pts[-1] - pts[i])) >= MIN_DISP_FOR_HEADING:
            end_h = np.arctan2(pts[-1][1] - pts[i][1], pts[-1][0] - pts[i][0])
            break
    return start_h, end_h


def _max_speed(xs, ys, ts):
    """Max instantaneous speed (m/s) from consecutive X/Y and TIMESTAMP."""
    pts = np.column_stack([xs, ys])
    dt = np.diff(ts.flatten())
    seg = np.hypot(*(np.diff(pts, axis=0).T))
    dt = np.where(dt <= 0, np.nan, dt)
    v = seg / dt
    return np.nanmax(v) if len(v) else 0.0


def classify_future_maneuver(xs, ys, ts):
    """Port of Waymo ClassifyTrack, adapted for our 5-class scheme.

    Returns one of:
      'stationary','straight','lane-change-left','lane-change-right',
      'left-turn','right-turn'.

    Trajectory is ego-relative (origin at start, x forward, y left)."""
    xs = np.asarray(xs).flatten()
    ys = np.asarray(ys).flatten()
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    final_displacement = np.hypot(dx, dy)
    max_speed = _max_speed(xs, ys, ts)

    # stationary
    if (max_speed < MAX_SPEED_FOR_STATIONARY and
            final_displacement < MAX_DISPLACEMENT_FOR_STATIONARY):
        return 'stationary'

    start_h, end_h = _heading_from_ends(xs, ys)
    if start_h is None or end_h is None:
        # not stationary by threshold, but no stable heading pair
        # (degenerate near-zero-motion path) -> treat as straight.
        return 'straight'
    heading_diff = np.arctan2(np.sin(end_h - start_h), np.cos(end_h - start_h))

    # straight-ish (small heading change)
    if abs(heading_diff) < MAX_ABS_HEADING_DIFF_FOR_STRAIGHT:
        if abs(dy) < MAX_LATERAL_DISPLACEMENT_FOR_STRAIGHT:
            return 'straight'
        return 'lane-change-right' if dy < 0 else 'lane-change-left'

    # turning (u-turns collapsed into turn direction by sign of lateral)
    return 'right-turn' if dy < 0 else 'left-turn'


def _load_future(fname, train_dir):
    """Load future X, Y, TIMESTAMP for a frame, or None if empty."""
    data = np.load(os.path.join(train_dir, fname), allow_pickle=True)
    modality = data['_modality_data'].item()
    future = modality[Modality.FUTURE_STATES]
    if future.isEmpty:
        return None
    xs = future.get(TrajectoryComponent.X).flatten()
    ys = future.get(TrajectoryComponent.Y).flatten()
    ts = future.get(TrajectoryComponent.TIMESTAMP)
    return xs, ys, ts


def build_manifest(train_dir, seq_ids, sequences):
    manifest = {}
    dropped = 0

    for i, seq_id in enumerate(seq_ids):
        if (i + 1) % 100 == 0:
            print(f'  Processing sequence {i+1}/{len(seq_ids)}...')

        seq_frames = sequences[seq_id]
        frame_indices = [f[0] for f in seq_frames]
        fname_map = {f[0]: f[1] for f in seq_frames}

        best_fname = None
        best_frame_idx = None
        best_maneuver = None
        best_rank = -1

        for frame_idx, fname in seq_frames:
            fut = _load_future(fname, train_dir)
            if fut is None:
                continue
            xs, ys, ts = fut
            maneuver = classify_future_maneuver(xs, ys, ts)
            rank = MANEUVER_RANK[maneuver]
            if rank > best_rank:
                best_rank = rank
                best_maneuver = maneuver
                best_fname = fname
                best_frame_idx = frame_idx

        if best_fname is None:
            dropped += 1
            continue

        pos = frame_indices.index(best_frame_idx)

        # Determine the target frame: the frame immediately before the best
        # maneuver frame, so there is always >= 1 context frame available for
        # image-difference features.
        if pos == 0:
            # Best frame is the first in the sequence. Shift the target forward
            # to the second frame so frame[0] can serve as context. Requires at
            # least 2 frames in the sequence.
            if len(frame_indices) > 1:
                target_frame_idx = frame_indices[1]
            else:
                dropped += 1
                continue
        else:
            target_frame_idx = frame_indices[pos - 1]

        target_fname = fname_map[target_frame_idx]

        # Build context window up to MAX_CONTEXT_FRAMES before target (inclusive).
        target_pos = frame_indices.index(target_frame_idx)
        context_start = max(0, target_pos - MAX_CONTEXT_FRAMES + 1)
        context_fnames = [
            fname_map[frame_indices[j]]
            for j in range(context_start, target_pos + 1)
        ]

        manifest[seq_id] = {
            'label': best_maneuver,
            'target_fname': target_fname,
            'context_fnames': context_fnames,
            'n_context': len(context_fnames),
            'best_fname': best_fname,
        }

    return manifest, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of sequences to sample (default: all)')
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR)
    parser.add_argument('--output', type=str, default=OUTPUT_PATH)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f'Scanning {args.train_dir}...')
    files = sorted([f for f in os.listdir(args.train_dir) if f.endswith('.npz')])
    print(f'Found {len(files):,} .npz files')

    # group by sequence
    sequences = defaultdict(list)
    for fname in files:
        seq_id = fname.rsplit('_', 1)[0]
        frame_idx = int(fname.rsplit('_', 1)[1].replace('.npz', ''))
        sequences[seq_id].append((frame_idx, fname))
    for seq_id in sequences:
        sequences[seq_id].sort(key=lambda x: x[0])

    seq_ids = list(sequences.keys())
    print(f'Found {len(seq_ids):,} sequences')

    if args.sample:
        import random
        random.seed(args.seed)
        seq_ids = random.sample(seq_ids, args.sample)
        print(f'Sampling {args.sample} sequences (seed={args.seed})')

    print('Building manifest...')
    manifest, dropped = build_manifest(args.train_dir, seq_ids, sequences)

    # report distribution
    label_counts = Counter(v['label'] for v in manifest.values())
    total = sum(label_counts.values())
    print(f'\nDone. Labeled: {total}, Dropped: {dropped}')
    print('\nSequence-level maneuver distribution:')
    for k, v in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v} ({v/total*100:.1f}%)')

    context_lengths = [v['n_context'] for v in manifest.values()]
    print(f'\nContext window sizes -- min: {min(context_lengths)}, '
          f'max: {max(context_lengths)}, mean: {np.mean(context_lengths):.1f}')

    # save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(manifest, f)
    print(f'\nManifest saved to {args.output}')


if __name__ == '__main__':
    main()