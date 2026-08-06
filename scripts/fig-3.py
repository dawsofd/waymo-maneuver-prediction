# --- Figure: trajectory-label geometry (|heading change| vs |lateral displacement|) ---
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, 'scripts')
from build_train_manifest import _heading_from_ends, _load_future, TRAIN_DIR

CACHE = 'outputs/trajectory_label_geometry.csv'
manifest = json.load(open('data/train_manifest.json'))

if os.path.exists(CACHE):
    df = pd.read_csv(CACHE)
else:
    rows = []
    for i, (sid, entry) in enumerate(manifest.items()):
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(manifest)}')
        fut = _load_future(entry['best_fname'], TRAIN_DIR)
        if fut is None:
            continue
        xs, ys, ts = fut
        start_h, end_h = _heading_from_ends(xs, ys)
        if start_h is None or end_h is None:
            continue
        heading_deg = abs(np.degrees(np.arctan2(np.sin(end_h - start_h),
                                                np.cos(end_h - start_h))))
        lat = abs(ys.flatten()[-1] - ys.flatten()[0])
        rows.append({'seq_id': sid, 'label': entry['label'],
                     'abs_heading_deg': heading_deg, 'abs_lateral_m': lat})
    df = pd.DataFrame(rows)
    os.makedirs('outputs', exist_ok=True)
    df.to_csv(CACHE, index=False)

COLORS = {'straight': '#1f77b4', 'left-turn': '#2ca02c', 'right-turn': '#ff7f0e',
          'lane-change-left': '#9467bd', 'lane-change-right': '#d62728'}
fig, ax = plt.subplots(figsize=(6.5, 4.2))   # paper size; keep small or it dominates the doc
for label, color in COLORS.items():
    sub = df[df['label'] == label]
    ax.scatter(sub['abs_heading_deg'], sub['abs_lateral_m'], s=6, alpha=0.45,
               color=color, label=f'{label} (n={len(sub)})', linewidths=0)
ax.axvline(30, ls='--', c='gray', lw=1)
ax.axhline(2.5, ls='--', c='gray', lw=1)
ax.text(31.5, ax.get_ylim()[1] * 0.97, '30° turn threshold', fontsize=7, color='gray', va='top')
ax.text(ax.get_xlim()[1] * 0.99, 3.1, '2.5 m lane-change threshold', fontsize=7,
        color='gray', ha='right')
ax.set_xlabel('|net heading change| (degrees)', fontsize=9)
ax.set_ylabel('|net lateral displacement| (m)', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.savefig('outputs/fig_trajectory_label_geometry.png', dpi=200, bbox_inches='tight')
plt.show()