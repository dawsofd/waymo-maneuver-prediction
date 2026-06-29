import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from collections import Counter
from standard_e2e import Modality, TrajectoryComponent

val_dir = 'data/processed/waymo_e2e/val'
files = [f for f in os.listdir(val_dir) if f.endswith('.npz')]

accel_buckets = Counter()

for fname in files:
    data = np.load(os.path.join(val_dir, fname), allow_pickle=True)
    modality = data['_modality_data'].item()
    past = modality[Modality.PAST_STATES]
    ax = past.get(TrajectoryComponent.ACCELERATION_X)[-1][0]
    ay = past.get(TrajectoryComponent.ACCELERATION_Y)[-1][0]
    accel = np.sqrt(ax**2 + ay**2)
    
    if accel < 0.5:
        accel_buckets['cruising'] += 1
    elif ax > 0:
        accel_buckets['accelerating'] += 1
    else:
        accel_buckets['braking'] += 1

total = sum(accel_buckets.values())
print('Acceleration state distribution:')
for k, v in sorted(accel_buckets.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v} ({v/total*100:.1f}%)')