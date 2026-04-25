"""
Composite hypertension figure — Part 2.
  A  Cell level   L6_hypertension_cell.png
  B  Patient level L6_hypertension_pt.png
Output: outputs/figures/part2_hypertension_composite.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
OUT  = PROJ / 'outputs/figures'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':     'Arial',
    'font.sans-serif': ['Arial'],
    'text.usetex':     False,
})

PANELS = [
    ('A', PROJ / 'outputs/final_thesis_part2/hypertension/L6_hypertension_cell.png'),
    ('B', PROJ / 'outputs/final_thesis_part2/hypertension/L6_hypertension_pt.png'),
]

imgs = [np.array(Image.open(p).convert('RGB')) for _, p in PANELS]
H, W = imgs[0].shape[:2]
aspect = H / W

cell_w = 6.0
cell_h = cell_w * aspect

fig, axes = plt.subplots(1, 2, figsize=(cell_w * 2, cell_h),
                         constrained_layout=False)
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.04)

for ax, (label, _), img in zip(axes, PANELS, imgs):
    ax.imshow(img, aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left', color='black')

out_path = OUT / 'part2_hypertension_composite.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.04)
plt.close(fig)
print(f'Saved → {out_path}')
