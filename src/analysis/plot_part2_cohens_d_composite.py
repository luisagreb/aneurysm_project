"""
Composite Cohen's d figure — Part 2 (TAV-ATAA vs BAV-ATAA).
Stitches 4 existing PNGs into a 2×2 grid:
  A  Cell level   L1 Disease        B  Cell level   L2 Age-corrected
  C  Patient level L1 Disease       D  Patient level L2 Age-corrected
Output: outputs/figures/part2_cohens_d_composite.png
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
    ('A', PROJ / 'outputs/final_thesis_part2/cell/L1_cohens_d.png',                  'Cell level — Disease'),
    ('B', PROJ / 'outputs/final_thesis_part2/cell/L2_age_corrected_cohens_d.png',    'Cell level — Age-corrected'),
    ('C', PROJ / 'outputs/final_thesis_part2/patient/L1_cohens_d.png',               'Patient level — Disease'),
    ('D', PROJ / 'outputs/final_thesis_part2/patient/L2_age_corrected_cohens_d.png', 'Patient level — Age-corrected'),
]

imgs = [np.array(Image.open(p).convert('RGB')) for _, p, _ in PANELS]
H, W = imgs[0].shape[:2]
aspect = H / W

cell_w = 6.0
cell_h = cell_w * aspect

fig, axes = plt.subplots(2, 2, figsize=(cell_w * 2, cell_h * 2 + 0.3),
                         constrained_layout=False)
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01,
                    hspace=0.06, wspace=0.04)

for idx, (ax, (label, _, title)) in enumerate(zip(axes.flat, PANELS)):
    ax.imshow(imgs[idx], aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left',
            color='black')

out_path = OUT / 'part2_cohens_d_composite.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.04)
plt.close(fig)
print(f'Saved → {out_path}')
