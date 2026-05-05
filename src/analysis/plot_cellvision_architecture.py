"""
CellVision architecture diagram — clean redesign.
Output: outputs/figures/cellvision_architecture.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
OUT  = PROJ / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
C_BG      = '#F0EDF8'
C_USER    = '#1558A0'
C_FLASK   = '#5B21B6'
C_NNUNET  = '#B91C1C'
C_FEAT    = '#166534'
C_ML      = '#92400E'
C_OUT     = '#1E3A5F'
WHITE     = '#FFFFFF'
GREY      = '#555555'

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# ── helpers ───────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, color, title, subtitle_lines, step=None):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle='round,pad=0.15',
                           facecolor=color, edgecolor=WHITE,
                           linewidth=2.5, alpha=0.95, zorder=2)
    ax.add_patch(patch)
    if step:
        ax.text(x + 0.22, y + h - 0.22, step,
                ha='left', va='top', fontsize=11, fontweight='bold',
                color='white', alpha=0.7, zorder=3)
    ax.text(x + w/2, y + h - 0.28, title,
            ha='center', va='top', fontsize=15, fontweight='bold',
            color=WHITE, zorder=3)
    for i, line in enumerate(subtitle_lines):
        ax.text(x + w/2, y + h - 0.72 - i * 0.38, line,
                ha='center', va='top', fontsize=11.5,
                color=WHITE, alpha=0.88, zorder=3)

def arr(ax, x1, y1, x2, y2, label='', color=GREY, lw=2.2, dbl=False):
    style = '<->' if dbl else '->'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=18),
                zorder=4)
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        # offset label to the right of vertical arrows, above horizontal
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) > abs(dx):       # vertical
            ax.text(mx + 0.18, my, label, ha='left', va='center',
                    fontsize=10.5, color=color, fontstyle='italic', zorder=5)
        else:                        # horizontal
            ax.text(mx, my + 0.18, label, ha='center', va='bottom',
                    fontsize=10.5, color=color, fontstyle='italic', zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1  — Browser (left)  |  Flask (centre)  |  nnU-Net (right)
# ══════════════════════════════════════════════════════════════════════════════
rbox(ax, 0.3, 6.6, 3.8, 2.8, C_USER,
     'Browser',
     ['HTML / CSS / JavaScript',
      'Three.js — 3D meshes',
      'Async fetch (REST)'],
     step='1')

rbox(ax, 6.1, 6.6, 3.8, 2.8, C_FLASK,
     'Flask Server',
     ['app.py — 8 REST routes',
      'Holds image state in RAM',
      'Orchestrates all steps'],
     step='2')

rbox(ax, 11.9, 6.6, 3.8, 2.8, C_NNUNET,
     'nnU-Net V2',
     ['3 segmentation models',
      'Actin · Mito · Nucleus',
      'GPU server via SSH tunnel'],
     step='3')

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2  — Feature extraction (centre-left)  |  ML Classifier (centre-right)
# ══════════════════════════════════════════════════════════════════════════════
rbox(ax, 4.2, 3.3, 3.8, 2.8, C_FEAT,
     'Feature Extraction',
     ['analyze_structures.py',
      '30 morphological features',
      'scipy · skimage · nibabel'],
     step='4')

rbox(ax, 8.2, 3.3, 3.8, 2.8, C_ML,
     'ML Classifier',
     ['LR · RF · SVM',
      'LOPO-CV trained',
      'classifier.joblib'],
     step='5')

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3  — Outputs
# ══════════════════════════════════════════════════════════════════════════════
rbox(ax, 2.0, 0.3, 12.0, 2.6, C_OUT,
     'Outputs returned to user',
     [''],
     step=None)

out_items = [
    ('Progress bar', 'live % while segmenting'),
    ('3D meshes',    'interactive, rotatable (Three.js)'),
    ('Slice viewer', '2D z-stack per channel'),
    ('Prediction',   '"Healthy" / "Diseased" + confidence'),
    ('Feature table','30 values shown in browser'),
    ('CSV download', 'features for one cell'),
    ('.nii ZIP',     'segmentation masks (3 files)'),
]
col_w = 12.0 / len(out_items)
for i, (label, detail) in enumerate(out_items):
    cx = 2.0 + col_w * i + col_w / 2
    ax.text(cx, 2.45, label, ha='center', va='top',
            fontsize=11.5, fontweight='bold', color=WHITE, zorder=3)
    ax.text(cx, 2.05, detail, ha='center', va='top',
            fontsize=9.5, color=WHITE, alpha=0.80, zorder=3)
    if i > 0:
        ax.plot([2.0 + col_w * i, 2.0 + col_w * i], [0.4, 2.85],
                color=WHITE, lw=0.6, alpha=0.3, zorder=3)

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS
# ══════════════════════════════════════════════════════════════════════════════

# Browser ↔ Flask
arr(ax, 4.1, 8.0, 6.1, 8.0, 'HTTP / fetch', color=C_USER, lw=2.2, dbl=True)

# Flask → nnU-Net
arr(ax, 9.9, 8.0, 11.9, 8.0, 'subprocess + .nii', color=C_NNUNET, lw=2.2, dbl=True)

# Flask → Feature extraction (diagonal down-left)
arr(ax, 7.0, 6.6, 6.1, 6.1, 'masks\n(numpy)', color=C_FEAT, lw=2.2)

# Feature extraction → ML Classifier
arr(ax, 8.0, 4.7, 8.2, 4.7, '30 features', color=C_ML, lw=2.2)

# Feature extraction → Outputs
arr(ax, 6.1, 3.3, 6.1, 2.9, '', color=WHITE, lw=1.8)

# ML Classifier → Outputs
arr(ax, 10.1, 3.3, 10.1, 2.9, '', color=WHITE, lw=1.8)

# Flask → Outputs (3D meshes, slices)
arr(ax, 8.0, 6.6, 8.0, 2.9, '', color=WHITE, lw=1.8)

# ══════════════════════════════════════════════════════════════════════════════
# SSH note
# ══════════════════════════════════════════════════════════════════════════════
ax.text(13.8, 6.48, 'ssh -L 8765:localhost:8765\nluisa@172.23.149.82',
        ha='center', va='top', fontsize=9, color=C_NNUNET, fontstyle='italic',
        bbox=dict(facecolor=WHITE, edgecolor=C_NNUNET,
                  boxstyle='round,pad=0.25', alpha=0.85), zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
ax.text(8.0, 9.82, 'CellVision — System Architecture',
        ha='center', va='top', fontsize=19, fontweight='bold', color='#1A1A2E')

out_path = OUT / 'cellvision_architecture.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2,
            facecolor=C_BG)
plt.close(fig)
print(f'Saved → {out_path}')
