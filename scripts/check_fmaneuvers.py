import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from collections import Counter, defaultdict
from standard_e2e import Modality, TrajectoryComponent

val_dir = 'data/processed/waymo_e2e/val'
files = sorted([f for f in os.listdir(val_dir) if f.endswith('.npz')])

def classify_future_maneuver(xs, ys):
    total_dist = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    if total_dist < 2.0:
        return 'stationary'
    n = len(xs)
    mid = n // 2
    dx1 = xs[mid] - xs[0]
    dy1 = ys[mid] - ys[0]
    dx2 = xs[-1] - xs[mid]
    dy2 = ys[-1] - ys[mid]
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
        return 'lane-change-left' if ys[-1] > 0 else 'lane-change-right'
    else:
        return 'left-turn' if ys[-1] > 0 else 'right-turn'

priority = ['left-turn', 'right-turn', 'lane-change-left', 'lane-change-right', 'straight', 'stationary']

segment_maneuvers = defaultdict(Counter)

for fname in files:
    data = np.load(os.path.join(val_dir, fname), allow_pickle=True)
    segment_id = str(data['segment_id'])
    modality = data['_modality_data'].item()
    future = modality[Modality.FUTURE_STATES]
    if future.isEmpty:
        continue
    xs = future.get(TrajectoryComponent.X).flatten()
    ys = future.get(TrajectoryComponent.Y).flatten()
    maneuver = classify_future_maneuver(xs, ys)
    segment_maneuvers[segment_id][maneuver] += 1

segment_labels = {}
for seg_id, counts in segment_maneuvers.items():
    for label in priority:
        if counts[label] > 0:
            segment_labels[seg_id] = label
            break

final_counts = Counter(segment_labels.values())
total = sum(final_counts.values())
print(f'Total sequences: {total}')
print('\nSequence-level maneuver distribution:')
for k, v in sorted(final_counts.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v} ({v/total*100:.1f}%)')