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

Output: data/processed/waymo_e2e/train_manifest.json

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


def classify_future_maneuver(xs, ys):
    total_dist = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    if total_dist < 2.0:
        return 'stationary'
    n = len(xs)
    mid = n // 2
    dx1, dy1 = xs[mid] - xs[0], ys[mid] - ys[0]
    dx2, dy2 = xs[-1] - xs[mid], ys[-1] - ys[mid]
    mag1 = np.sqrt(dx1**2 + dy1**2)
    mag2 = np.sqrt(dx2**2 + dy2**2)
    if mag1 < 0.1 or mag2 < 0.1:
        return 'stationary'
    cross = dx1 * dy2 - dy1 * dx2
    dot = dx1 * dx2 + dy1 * dy2
    angle = np.arctan2(cross / (mag1 * mag2), dot / (mag1 * mag2)) * 180 / np.pi
    if dot / (mag1 * mag2) < -0.3:
        return 'left-turn' if ys[-1] > 0 else 'right-turn'
    elif abs(angle) < 8:
        return 'straight'
    elif abs(angle) < 20:
        return 'lane-change-left' if angle > 0 else 'lane-change-right'
    else:
        return 'left-turn' if angle > 0 else 'right-turn'


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
            data = np.load(os.path.join(train_dir, fname), allow_pickle=True)
            modality = data['_modality_data'].item()
            future = modality[Modality.FUTURE_STATES]
            if future.isEmpty:
                continue
            xs = future.get(TrajectoryComponent.X).flatten()
            ys = future.get(TrajectoryComponent.Y).flatten()
            maneuver = classify_future_maneuver(xs, ys)
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

        if pos == 0:
            if best_maneuver == 'straight':
                target_fname = best_fname
            else:
                dropped += 1
                continue
        else:
            target_fname = fname_map[frame_indices[pos - 1]]

        # build context window up to MAX_CONTEXT_FRAMES before target (inclusive)
        target_pos = frame_indices.index(
            int(target_fname.rsplit('_', 1)[1].replace('.npz', ''))
        )
        context_start = max(0, target_pos - MAX_CONTEXT_FRAMES + 1)
        context_fnames = [
            fname_map[frame_indices[i]]
            for i in range(context_start, target_pos + 1)
        ]

        manifest[seq_id] = {
            'label': best_maneuver,
            'target_fname': target_fname,
            'context_fnames': context_fnames,
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

    context_lengths = [len(v['context_fnames']) for v in manifest.values()]
    print(f'\nContext window sizes -- min: {min(context_lengths)}, '
          f'max: {max(context_lengths)}, mean: {np.mean(context_lengths):.1f}')

    # save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(manifest, f)
    print(f'\nManifest saved to {args.output}')


if __name__ == '__main__':
    main()