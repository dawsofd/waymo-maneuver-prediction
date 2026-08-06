"""Figure: test macro-F1 vs classifier inference cost (cached features).

Data sources: efficiency_and_tuning.ipynb (tuned SVM/RF rows, pooled test fold)
plus the Trend+Flow + MoViNet stack timed in benchmark_pooled_3class.ipynb cell 11.
Logistic-regression rows omitted (dominated everywhere). Inference cost is the
trained classifier on precomputed features; one-time feature extraction is
reported separately in the text.
"""
import matplotlib.pyplot as plt

#            label,                    test_F1, infer_ms, dx, dy, ha
POINTS = [
    ('Trend+Flow / RF',            0.607, 0.088,  1.13,  0.006, 'left'),
    ('Trend+Flow / SVM',           0.544, 0.036,  1.13,  0.000, 'left'),
    ('Trend+Flow+CNN / RF',        0.587, 0.085,  1.13,  0.000, 'left'),
    ('Trend+Flow+CNN / SVM',       0.575, 0.063,  0.88, -0.011, 'right'),
    ('HOG / SVM',                  0.526, 2.188,  0.88,  0.007, 'right'),
    ('HOG / RF',                   0.486, 0.089,  0.86,  0.009, 'right'),
    ('HOG (PCA-50) / SVM',         0.524, 0.133,  1.13,  0.003, 'left'),
    ('HOG+Trend+Flow / SVM',       0.526, 2.207,  0.88, -0.010, 'right'),
    ('HOG+Trend+Flow / RF',        0.495, 0.081,  0.88,  0.006, 'right'),
    ('All classical / SVM',        0.514, 2.297,  1.13,  0.000, 'left'),
    ('All classical / RF',         0.484, 0.103,  1.15, -0.006, 'left'),
    ('Trend+Flow + MoViNet / SVM', 0.693, 0.230,  1.13,  0.000, 'left'),
]
ACC_PICK = 'Trend+Flow + MoViNet / SVM'
EFF_PICK = 'Trend+Flow / RF'

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for label, f1, ms, dx, dy, ha in POINTS:
    ax.scatter(ms, f1, s=42, color='#1f77b4', zorder=3)
    ax.annotate(label, (ms * dx, f1 + dy), fontsize=6.8, ha=ha, va='center', zorder=4)

for pick, tag, tdy in [(ACC_PICK, 'accuracy pick', 0.018), (EFF_PICK, 'efficiency pick', 0.018)]:
    f1, ms = next((p[1], p[2]) for p in POINTS if p[0] == pick)
    ax.scatter(ms, f1, s=190, facecolors='none', edgecolors='#d95f02', linewidths=2.2, zorder=2)
    ax.annotate(tag, (ms, f1 + tdy), fontsize=8, ha='center', color='#d95f02',
                fontweight='bold', zorder=4)

ax.set_xscale('log')
ax.set_xlabel('classifier inference cost, ms per sequence (cached features)', fontsize=9)
ax.set_ylabel('test macro-F1', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, which='both', alpha=0.25, lw=0.5)
ax.set_ylim(0.455, 0.735)
ax.set_xlim(0.022, 9)
plt.tight_layout()
plt.savefig('fig_accuracy_vs_inference.png', dpi=200, bbox_inches='tight')
print('saved')
