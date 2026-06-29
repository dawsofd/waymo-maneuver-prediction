import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import json
from collections import Counter, defaultdict
from standard_e2e import Modality, TrajectoryComponent

val_dir = 'data/processed/waymo_e2e/val'
files = sorted([f for f in os.listdir(val_dir) if f.endswith('.npz')])

# load scenario cluster labels
with open('data/val_sequence_name_to_scenario_cluster.json') as f:
    scenario_labels = json.load(f)

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
    sin_angle = cross / (mag1 * mag2)
    cos_angle = dot / (mag1 * mag2)
    angle = np.arctan2(sin_angle, cos_angle) * 180 / np.pi
    if cos_angle < -0.3:
        return 'left-turn' if angle > 0 else 'right-turn'
    elif abs(angle) < 8:
        return 'straight'
    elif abs(angle) < 20:
        return 'lane-change-left' if angle > 0 else 'lane-change-right'
    else:
        return 'left-turn' if angle > 0 else 'right-turn'

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

# assign sequence-level maneuver label
segment_labels = {}
for seg_id, counts in segment_maneuvers.items():
    for label in priority:
        if counts[label] > 0:
            segment_labels[seg_id] = label
            break

# cross-reference with scenario clusters
scenario_to_maneuvers = defaultdict(Counter)
for seg_id, maneuver in segment_labels.items():
    scenario = scenario_labels.get(seg_id, {}).get('scenario_cluster', 'unknown')
    scenario_to_maneuvers[scenario][maneuver] += 1

print('Maneuver distribution by scenario cluster:\n')
for scenario, counts in sorted(scenario_to_maneuvers.items()):
    total = sum(counts.values())
    print(f'{scenario} (n={total}):')
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v} ({v/total*100:.1f}%)')
    print()