import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from collections import Counter, defaultdict
from standard_e2e import Modality, TrajectoryComponent

val_dir = 'data/processed/waymo_e2e/val'
files = sorted([f for f in os.listdir(val_dir) if f.endswith('.npz')])

# group frames by segment, preserving order via frame_id
segments = defaultdict(list)
for fname in files:
    data = np.load(os.path.join(val_dir, fname), allow_pickle=True)
    segment_id = str(data['segment_id'])
    frame_id = int(data['frame_id'])
    modality = data['_modality_data'].item()
    past = modality[Modality.PAST_STATES]
    vx = past.get(TrajectoryComponent.VELOCITY_X)[-1][0]
    vy = past.get(TrajectoryComponent.VELOCITY_Y)[-1][0]
    speed = np.sqrt(vx**2 + vy**2)
    segments[segment_id].append((frame_id, speed))

def bucket(speed):
    if speed < 1.4:
        return 'stopped'
    elif speed < 11.0:
        return 'urban'
    else:
        return 'highway'

def smooth_buckets(speeds, window=10):
    """Apply rolling majority vote to smooth speed bucket labels."""
    buckets = [bucket(s) for s in speeds]
    smoothed = []
    for i in range(len(buckets)):
        window_start = max(0, i - window // 2)
        window_end = min(len(buckets), i + window // 2)
        window_buckets = buckets[window_start:window_end]
        smoothed.append(Counter(window_buckets).most_common(1)[0][0])
    return smoothed

MIN_LENGTH = 20  # minimum frames per sub-sequence

subsegment_counts = Counter()
total_subsegments = 0

for seg_id, frames in segments.items():
    frames.sort(key=lambda x: x[0])
    speeds = [f[1] for f in frames]
    smoothed = smooth_buckets(speeds)
    
    # find contiguous runs
    current_bucket = smoothed[0]
    current_len = 1
    
    for b in smoothed[1:]:
        if b == current_bucket:
            current_len += 1
        else:
            if current_len >= MIN_LENGTH:
                subsegment_counts[current_bucket] += 1
                total_subsegments += 1
            current_bucket = b
            current_len = 1
    
    if current_len >= MIN_LENGTH:
        subsegment_counts[current_bucket] += 1
        total_subsegments += 1

print(f'Total usable sub-sequences (min {MIN_LENGTH} frames): {total_subsegments}')
print('\nSub-sequence distribution:')
for k, v in sorted(subsegment_counts.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v} ({v/total_subsegments*100:.1f}%)')