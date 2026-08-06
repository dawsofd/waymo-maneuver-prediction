"""Figure (paper section 6): test macro-F1 vs training time and vs inference cost.

Left: model training time on cached features (log s). Right: per-sequence
classifier inference (log ms). Tuned SVM/RF rows from efficiency_and_tuning.ipynb;
Trend+Flow + MoViNet stack timed in benchmark_pooled_3class.ipynb under the same
protocol. LogReg rows omitted (dominated). Feature extraction is a one-time cached
cost reported in the text.
"""
import matplotlib.pyplot as plt

#        label,                        F1,   train_s, infer_ms
DATA = [
    ('Trend+Flow / RF',             0.607, 0.31, 0.088),
    ('Trend+Flow / SVM',            0.544, 0.06, 0.036),
    ('Trend+Flow+CNN / RF',         0.587, 0.34, 0.085),
    ('Trend+Flow+CNN / SVM',        0.575, 0.07, 0.063),
    ('HOG / SVM',                   0.526, 3.27, 2.188),
    ('HOG / RF',                    0.486, 1.36, 0.089),
    ('HOG (PCA-50) / SVM',          0.524, 0.07, 0.133),
    ('HOG+Trend+Flow / SVM',        0.526, 3.00, 2.207),
    ('HOG+Trend+Flow / RF',         0.495, 1.35, 0.081),
    ('All classical / SVM',         0.514, 2.90, 2.297),
    ('All classical / RF',          0.484, 1.53, 0.103),
    ('Trend+Flow + MoViNet / SVM',  0.693, 0.20, 0.230),
]
ACC, EFF = 'Trend+Flow + MoViNet / SVM', 'Trend+Flow / RF'

# per-panel label placement: {label: (xfactor, dy, ha)} ; None = unlabeled in that panel
L = {  # left panel (training time)
    'Trend+Flow / RF':            (1.16,  0.004, 'left'),
    'Trend+Flow / SVM':           (1.16,  0.000, 'left'),
    'Trend+Flow+CNN / RF':        (1.16, -0.010, 'left'),
    'Trend+Flow+CNN / SVM':       (1.16, -0.001, 'left'),
    'HOG / SVM':                  (0.86,  0.008, 'right'),
    'HOG / RF':                   (0.86,  0.000, 'right'),
    'HOG (PCA-50) / SVM':         (1.16, -0.012, 'left'),
    'HOG+Trend+Flow / SVM':       (0.86, -0.008, 'right'),
    'HOG+Trend+Flow / RF':        (1.16,  0.004, 'left'),
    'All classical / SVM':        (1.16, -0.002, 'left'),
    'All classical / RF':         (1.16, -0.006, 'left'),
    'Trend+Flow + MoViNet / SVM': (1.16,  0.000, 'left'),
}
R = {  # right panel (inference)
    'Trend+Flow / RF':            (1.16,  0.005, 'left'),
    'Trend+Flow / SVM':           (1.16,  0.000, 'left'),
    'Trend+Flow+CNN / RF':        (1.16, -0.003, 'left'),
    'Trend+Flow+CNN / SVM':       (0.86, -0.010, 'right'),
    'HOG / SVM':                  (0.86,  0.008, 'right'),
    'HOG / RF':                   (0.86,  0.009, 'right'),
    'HOG (PCA-50) / SVM':         (1.16,  0.003, 'left'),
    'HOG+Trend+Flow / SVM':       (0.86, -0.009, 'right'),
    'HOG+Trend+Flow / RF':        (0.86, -0.014, 'right'),
    'All classical / SVM':        (1.16,  0.000, 'left'),
    'All classical / RF':         (1.16, -0.006, 'left'),
    'Trend+Flow + MoViNet / SVM': (1.16,  0.000, 'left'),
}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)

def panel(ax, xi, placements, xlabel, xlim):
    for label, f1, ts, ms in DATA:
        x = ts if xi == 'train' else ms
        ax.scatter(x, f1, s=34, color='#1f77b4', zorder=3)
        if placements.get(label):
            xf, dy, ha = placements[label]
            ax.annotate(label, (x * xf, f1 + dy), fontsize=6.0, ha=ha, va='center', zorder=4)
    for pick, tag in [(ACC, 'accuracy pick'), (EFF, 'efficiency pick')]:
        f1, ts, ms = next((d[1], d[2], d[3]) for d in DATA if d[0] == pick)
        x = ts if xi == 'train' else ms
        ax.scatter(x, f1, s=160, facecolors='none', edgecolors='#d95f02', linewidths=2.0, zorder=2)
        ax.annotate(tag, (x, f1 + 0.017), fontsize=7.5, ha='center', color='#d95f02',
                    fontweight='bold', zorder=4)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, which='both', alpha=0.25, lw=0.5)
    ax.set_xlim(*xlim)

panel(axL, 'train', L, 'training time, s (cached features, log)', (0.03, 12))
panel(axR, 'infer', R, 'inference, ms per sequence (cached features, log)', (0.02, 9))
axL.set_ylabel('test macro-F1', fontsize=8.5)
axL.set_ylim(0.455, 0.735)
plt.tight_layout()
plt.savefig('fig_efficiency_two_panel.png', dpi=200, bbox_inches='tight')
print('saved')
