import json
import os
from collections import Counter

label_base = 'data/bdd100k/labels/100k'

for split in ['train', 'val']:
    label_dir = os.path.join(label_base, split)
    weather_counts = Counter()
    scene_counts = Counter()
    timeofday_counts = Counter()
    
    for fname in os.listdir(label_dir):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(label_dir, fname)) as f:
            d = json.load(f)
        attrs = d.get('attributes', {})
        weather_counts[attrs.get('weather', 'undefined')] += 1
        scene_counts[attrs.get('scene', 'undefined')] += 1
        timeofday_counts[attrs.get('timeofday', 'undefined')] += 1
    
    total = sum(scene_counts.values())
    print(f'\n=== {split} (n={total}) ===')
    
    print('\nScene:')
    for k, v in sorted(scene_counts.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v} ({v/total*100:.1f}%)')
    
    print('\nTime of Day:')
    for k, v in sorted(timeofday_counts.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v} ({v/total*100:.1f}%)')
    
    print('\nWeather:')
    for k, v in sorted(weather_counts.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v} ({v/total*100:.1f}%)')