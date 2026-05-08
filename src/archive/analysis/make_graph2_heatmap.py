"""
Regenerate graph2_pairwise_heatmap.png with:
  - Organelle group labels (Nucleus / Actin / Mitochondria) as vertical
    bracket annotations on the LEFT of the heatmap
  - Colorbar pushed further right so it never overlaps feature labels

Reads: classification_results/three_group_analysis/summary_table.csv
Writes: classification_results/three_group_analysis/graph2_pairwise_heatmap.png

Usage:
  cd /home/luisa/aneurysm_project
  python src/analysis/make_graph2_heatmap.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path('classification_results/three_group_analysis')
summary_df = pd.read_csv(OUT_DIR / 'summary_table.csv')

# ── Feature order: Nucleus top, Actin middle, Mito bottom ─────────────────────
FEAT_ORDER = [
    # Nucleus (rows 0-5)
    'Nucleus_Volume', 'Nucleus_Sphericity', 'Nucleus_Circularity',
    'Nucleus_Elongation', 'Nucleus_Flatness', 'Nucleus_Solidity',
    # Actin (rows 6-11)
    'Actin_Volume', 'Actin_Skeleton_Length', 'Actin_Solidity',
    'Actin_Fractional_Anisotropy', 'Actin_Major_Axis', 'Actin_Minor_Axis',
    # Mito (rows 12-20)
    'Mito_Volume', 'Mito_Sphericity', 'Mito_Fragment_Count',
    'Mito_Mean_Fragment_Volume', 'Mito_Branch_Count',
    'Mito_Total_Network_Length', 'Mito_Mean_Tortuosity',
    'Mito_Junction_Count', 'Mito_Cyclomatic_Number',
]

# Organelle bracket definitions: (label, first_row, last_row_inclusive, color)
ORGANELLE_GROUPS = [
    ('Nucleus',       0,  5,  '#4A148C'),
    ('Actin',         6, 11,  '#1A237E'),
    ('Mitochondria', 12, 20,  '#1B5E20'),
]

PAIR_COLS = [
    ('d_TAVNA_vsTAVATAA',   'FDR_TAVNA_vsTAVATAA',   'TAV-NA\nvs\nTAV-ATAA'),
    ('d_TAVNA_vsBAVATAA',   'FDR_TAVNA_vsBAVATAA',   'TAV-NA\nvs\nBAV-ATAA'),
    ('d_TAVATAA_vsBAVATAA', 'FDR_TAVATAA_vsBAVATAA', 'TAV-ATAA\nvs\nBAV-ATAA'),
]

# ── Build matrices ─────────────────────────────────────────────────────────────
sf = summary_df.set_index('Feature')
d_vals   = sf[[p[0] for p in PAIR_COLS]].loc[FEAT_ORDER].values.astype(float)
fdr_vals = sf[[p[1] for p in PAIR_COLS]].loc[FEAT_ORDER].values.astype(float)
col_labels = [p[2] for p in PAIR_COLS]


def stars(p):
    if np.isnan(p): return ''
    if p < 0.001:   return '***'
    if p < 0.01:    return '**'
    if p < 0.05:    return '*'
    return ''


def shorten(feat):
    return (feat
            .replace('Nucleus_', 'N: ')
            .replace('Actin_', 'A: ')
            .replace('Mito_', 'M: ')
            .replace('_', ' '))


# ── Figure layout (explicit axes positions) ────────────────────────────────────
# figure coords: x in [0,1] left-to-right, y in [0,1] bottom-to-top
#   bracket labels : x  0.01 – 0.05
#   bracket lines  : x  0.05 – 0.07
#   feature labels : handled by matplotlib ytick labels
#   heatmap axes   : left=0.22, bottom=0.08, width=0.58, height=0.86
#   colorbar axes  : left=0.84, bottom=0.08, width=0.03, height=0.86

N_ROWS = len(FEAT_ORDER)
HM_LEFT   = 0.22
HM_BOTTOM = 0.08
HM_WIDTH  = 0.55
HM_HEIGHT = 0.86
CB_LEFT   = 0.85   # colorbar well to the right

fig = plt.figure(figsize=(10, 10))

ax = fig.add_axes([HM_LEFT, HM_BOTTOM, HM_WIDTH, HM_HEIGHT])

vmax = max(1.0, float(np.nanmax(np.abs(d_vals))))
im = ax.imshow(d_vals, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
               aspect='auto', interpolation='nearest')

# Stars
for r in range(N_ROWS):
    for c in range(3):
        s = stars(fdr_vals[r, c])
        if s:
            ax.text(c, r, s, ha='center', va='center',
                    fontsize=9, color='black', fontweight='bold')

# x-axis
ax.set_xticks(range(3))
ax.set_xticklabels(col_labels, fontsize=9.5)
ax.xaxis.set_tick_params(length=0)

# y-axis feature labels
ax.set_yticks(range(N_ROWS))
ax.set_yticklabels([shorten(f) for f in FEAT_ORDER], fontsize=8.5)
ax.yaxis.set_tick_params(length=0)

# Separator lines between organelle groups
ax.axhline(5.5,  color='white', lw=2.5)
ax.axhline(11.5, color='white', lw=2.5)

ax.set_title("Pairwise Cohen's d\n(* FDR < 0.05   ** < 0.01   *** < 0.001)",
             fontsize=11, pad=10)

# Colorbar — pushed right with explicit axes
cax = fig.add_axes([CB_LEFT, HM_BOTTOM, 0.025, HM_HEIGHT])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Cohen's d  (G2 − G1)", fontsize=9)
cbar.ax.axhline(0, color='black', lw=0.8, ls='--')
cbar.ax.tick_params(labelsize=8)

# ── Vertical organelle bracket labels (left of heatmap) ───────────────────────
# Row 0 is at the TOP of the axes (imshow origin='upper').
# In figure coords, row i centre = HM_BOTTOM + HM_HEIGHT*(1 - (i+0.5)/N_ROWS)

BRACKET_LINE_X = 0.065   # right edge of bracket line (figure coords)
BRACKET_LABEL_X = 0.04   # centre of vertical text

for label, r0, r1, color in ORGANELLE_GROUPS:
    # figure-y coords for group
    fig_y_top    = HM_BOTTOM + HM_HEIGHT * (1.0 - r0 / N_ROWS)
    fig_y_bottom = HM_BOTTOM + HM_HEIGHT * (1.0 - (r1 + 1) / N_ROWS)
    fig_y_center = (fig_y_top + fig_y_bottom) / 2

    # Vertical bracket line
    fig.add_artist(
        plt.Line2D(
            [BRACKET_LINE_X, BRACKET_LINE_X],
            [fig_y_bottom + 0.005, fig_y_top - 0.005],
            transform=fig.transFigure,
            color=color, lw=2.5, solid_capstyle='round',
        )
    )
    # Short horizontal tick at top and bottom of bracket
    for fig_y_tick in [fig_y_top - 0.005, fig_y_bottom + 0.005]:
        fig.add_artist(
            plt.Line2D(
                [BRACKET_LINE_X, BRACKET_LINE_X + 0.012],
                [fig_y_tick, fig_y_tick],
                transform=fig.transFigure,
                color=color, lw=1.5,
            )
        )

    # Vertical label text
    fig.text(
        BRACKET_LABEL_X, fig_y_center,
        label,
        ha='center', va='center',
        rotation=90,
        fontsize=10, fontweight='bold',
        color=color,
        transform=fig.transFigure,
    )

plt.savefig(OUT_DIR / 'graph2_pairwise_heatmap.png', dpi=180, bbox_inches='tight')
plt.close()
print("Saved graph2_pairwise_heatmap.png")
