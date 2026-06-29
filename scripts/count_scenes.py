import json
import os
from collections import Counter

def count_scenes(labels_dir):
    counts = Counter()
    for fname in os.listdir(labels_dir):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(labels_dir, fname)) as f:
            d = json.load(f)
        scene = d.get('attributes', {}).get('scene', 'undefined')
        counts[scene] += 1
    return counts

for split in ['train', 'val']:
    path = f'data/bdd100k/labels/100k/{split}'
    print(f'\n{split}:')
    counts = count_scenes(path)
    total = sum(counts.values())
    for scene, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f'  {scene}: {cnt} ({cnt/total*100:.1f}%)')
    print(f'  Total: {total}')