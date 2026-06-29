import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from collections import Counter
from standard_e2e import Modality, TrajectoryComponent

val_dir = 'data/processed/waymo_e2e/val'
files = [f for f in os.listdir(val_dir) if f.endswith('.npz')]

def classify_maneuver(xs, ys, headings=None):
    """
    Classify trajectory into Waymo motion prediction buckets:
    straight, straight-left, straight-right, left, right, 
    left-u-turn, right-u-turn, stationary
    """
    # total displacement
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    total_dist = np.sqrt(dx**2 + dy**2)
    
    if total_dist < 1.0:  # less than 1 meter total movement
        return 'stationary'
    
    # heading change: approximate from trajectory shape
    # forward direction is along the trajectory
    # lateral displacement relative to forward motion
    
    # use first and last segments to estimate turn
    n = len(xs)
    mid = n // 2
    
    # direction of first half
    dx1 = xs[mid] - xs[0]
    dy1 = ys[mid] - ys[0]
    
    # direction of second half  
    dx2 = xs[-1] - xs[mid]
    dy2 = ys[-1] - ys[mid]
    
    # cross product to determine turn direction
    cross = dx1 * dy2 - dy1 * dx2
    # dot product to determine if going forward or u-turn
    dot = dx1 * dx2 + dy1 * dy2
    
    # magnitude of turn
    mag1 = np.sqrt(dx1**2 + dy1**2)
    mag2 = np.sqrt(dx2**2 + dy2**2)
    
    if mag1 < 0.1 or mag2 < 0.1:
        return 'stationary'
    
    sin_angle = cross / (mag1 * mag2)
    cos_angle = dot / (mag1 * mag2)
    angle = np.arctan2(sin_angle, cos_angle) * 180 / np.pi
    
    # classify based on angle
    if cos_angle < -0.5:  # u-turn (>120 degrees)
        return 'left-u-turn' if cross > 0 else 'right-u-turn'
    elif abs(angle) < 10:
        return 'straight'
    elif abs(angle) < 25:
        return 'straight-left' if angle > 0 else 'straight-right'
    else:
        return 'left' if angle > 0 else 'right'

maneuver_counts = Counter()

for fname in files:
    data = np.load(os.path.join(val_dir, fname), allow_pickle=True)
    modality = data['_modality_data'].item()
    past = modality[Modality.PAST_STATES]
    
    xs = past.get(TrajectoryComponent.X).flatten()
    ys = past.get(TrajectoryComponent.Y).flatten()
    
    maneuver = classify_maneuver(xs, ys)
    maneuver_counts[maneuver] += 1

total = sum(maneuver_counts.values())
print('Maneuver distribution:')
for k, v in sorted(maneuver_counts.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v} ({v/total*100:.1f}%)')