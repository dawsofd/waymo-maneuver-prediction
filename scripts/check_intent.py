import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from collections import Counter
from standard_e2e import Modality, TrajectoryComponent

val_dir = 'data/processed/waymo_e2e/val'
files = [f for f in os.listdir(val_dir) if f.endswith('.npz')]

intent_counts = Counter()
speed_buckets = Counter()

for fname in files:
    data = np.load(os.path.join(val_dir, fname), allow_pickle=True)
    modality = data['_modality_data'].item()

    # intent
    intent = modality[Modality.INTENT]
    intent_counts[intent.name] += 1

    # speed from most recent past state
    past = modality[Modality.PAST_STATES]
    vx = past.get(TrajectoryComponent.VELOCITY_X)[-1][0]
    vy = past.get(TrajectoryComponent.VELOCITY_Y)[-1][0]
    speed = np.sqrt(vx**2 + vy**2)

    if speed < 1.4:
        speed_buckets['stopped (<1.4 m/s)'] += 1
    elif speed < 11.0:
        speed_buckets['urban (1.4-11 m/s)'] += 1
    else:
        speed_buckets['highway (>11 m/s)'] += 1

print('Intent distribution:')
total = sum(intent_counts.values())
for k, v in sorted(intent_counts.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v} ({v/total*100:.1f}%)')

print('\nSpeed bucket distribution:')
total = sum(speed_buckets.values())
for k, v in sorted(speed_buckets.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v} ({v/total*100:.1f}%)')