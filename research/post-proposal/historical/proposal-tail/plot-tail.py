"""Render the proposal's train-defined triplet tail; no experimental AP results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
d = json.loads((root / 'train-counts.json').read_text())
rows = sorted(d['rows'], key=lambda r: -r['train_boxes'])
z = np.array([r['z'] for r in rows])
assert len(rows) == 86 and sum(z < 0) == 47
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14})
fig, ax = plt.subplots(figsize=(8, 8))
fig.subplots_adjust(left=.14, right=.96, bottom=.20, top=.81)
navy, red = '#17365D', '#B43232'
x = np.arange(1, 87)
ax.axvspan(39.5, 86.5, color=red, alpha=.07)
ax.bar(x, z, width=.85, color=np.where(z < 0, red, navy), zorder=3)
ax.axhline(0, color='#242424', linewidth=1.5, zorder=4)
ax.set(xlim=(.3, 86.7), ylim=(-3.15, 3.1),
       xlabel='Triplet classes, most to least frequent',
       ylabel='z-score of log₁₀ training-box count')
ax.set_xticks([1, 20, 39, 60, 86])
ax.set_yticks([-3, -2, -1, 0, 1, 2])
ax.grid(axis='y', alpha=.18, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.text(2, 2.9, '39 classes at or above\nthe geometric mean',
        color=navy, fontsize=13, va='top', weight='bold')
ax.text(45, 2.9, '47 tail classes\nz < 0',
        color=red, fontsize=15, va='top', weight='bold')
ax.annotate('z = 0: geometric mean\n6,434 training boxes',
            xy=(68, 0), xytext=(49, .65), fontsize=13,
            arrowprops={'arrowstyle':'->', 'color':'#242424'},
            bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.9})
fig.text(.5, .95, 'Defining the triplet tail', ha='center',
         fontsize=23, weight='bold', color=navy)
fig.text(.5, .895, '86 ROAD-Waymo classes • training split only',
         ha='center', fontsize=15)
fig.text(.5, .10, r'$z_c = (\log_{10} n_c - \mu)\,/\,\sigma$',
         ha='center', fontsize=20)
fig.text(.5, .055, 'μ = 3.8085   σ = 0.7256   Geometric mean = 10^μ',
         ha='center', fontsize=12)
for ext in ['png', 'svg', 'pdf']:
    fig.savefig(root / f'triplet-tail-zscore.{ext}', dpi=220, facecolor='white')
