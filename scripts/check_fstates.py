import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from collections import Counter
from standard_e2e import Modality, TrajectoryComponent

val_dir = 'data/processed/waymo_e2e/val'
files = [f for f in os.listdir(val_dir) if f.endswith('.npz')]

def classify_future_maneuver(xs, ys):
    """
    Classify future trajectory into Waymo maneuver buckets:
    straight, left-turn, right-turn, lane-change-left, lane-change-right, stationary
    """
    total_dist = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    
    if total_dist < 2.0:
        return 'stationary'
    
    # total lateral displacement (y-axis = left in vehicle frame)
    lateral = ys[-1] - ys[0]
    # total forward displacement
    forward = xs[-1] - xs[0]
    
    # heading change approximation
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
    
    # lateral ratio for distinguishing lane change vs turn
    lateral_ratio = abs(lateral) / total_dist
    
    if cos_angle < -0.3:  # u-turn
        return 'left-turn' if angle > 0 else 'right-turn'
    elif abs(angle) < 8:
        return 'straight'
    elif abs(angle) < 20:
        # small angle - lane change rather than full turn
        return 'lane-change-left' if angle > 0 else 'lane-change-right'
    else:
        return 'left-turn' if angle > 0 else 'right-turn'

maneuver_counts = Counter()
empty_count = 0

for fname in files:
    data = np.load(os.path.join(val_dir, fname), allow_pickle=True)
    modality = data['_modality_data'].item()
    future = modality[Modality.FUTURE_STATES]
    
    if future.isEmpty:
        empty_count += 1
        continue
    
    xs = future.get(TrajectoryComponent.X).flatten()
    ys = future.get(TrajectoryComponent.Y).flatten()
    
    maneuver = classify_future_maneuver(xs, ys)
    maneuver_counts[maneuver] += 1

total = sum(maneuver_counts.values())
print(f'Empty future states: {empty_count}')
print(f'\nFuture maneuver distribution (n={total}):')
for k, v in sorted(maneuver_counts.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v} ({v/total*100:.1f}%)')