import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from collections import Counter, defaultdict
from standard_e2e import Modality, TrajectoryComponent

val_dir = 'data/processed/waymo_e2e/val'
files = [f for f in os.listdir(val_dir) if f.endswith('.npz')]

# group files by segment_id
from collections import defaultdict
segments = defaultdict(list)
for fname in files:
    data = np.load(os.path.join(val_dir, fname), allow_pickle=True)
    segment_id = str(data['segment_id'])
    modality = data['_modality_data'].item()
    past = modality[Modality.PAST_STATES]
    vx = past.get(TrajectoryComponent.VELOCITY_X)[-1][0]
    vy = past.get(TrajectoryComponent.VELOCITY_Y)[-1][0]
    speed = np.sqrt(vx**2 + vy**2)
    segments[segment_id].append(speed)

# for each segment, check how many unique speed buckets appear
def bucket(speed):
    if speed < 1.4:
        return 'stopped'
    elif speed < 11.0:
        return 'urban'
    else:
        return 'highway'

pure_segments = 0
mixed_segments = 0
bucket_counts = Counter()

for seg_id, speeds in segments.items():
    buckets = set(bucket(s) for s in speeds)
    if len(buckets) == 1:
        pure_segments += 1
        bucket_counts[list(buckets)[0]] += 1
    else:
        mixed_segments += 1

total = pure_segments + mixed_segments
print(f'Pure segments (single speed bucket): {pure_segments} ({pure_segments/total*100:.1f}%)')
print(f'Mixed segments (multiple speed buckets): {mixed_segments} ({mixed_segments/total*100:.1f}%)')
print('\nPure segment breakdown:')
for k, v in sorted(bucket_counts.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v}')