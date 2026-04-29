"""
Composite Cohen's d figure — Part 1 (TAV-NA vs TAV-ATAA).
2 × 2 grid:
  A  Cell level   L1 Disease        B  Cell level   L2 Age-corrected
  C  Patient level L1 Disease       D  Patient level L2 Age-corrected

Features sorted by cell-level L1 |Cohen's d| (A sets the order; B, C, D follow).
Output: outputs/figures/part1_cohens_d_composite.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
OUT  = PROJ / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# ── colours ───────────────────────────────────────────────────────────────────
ORGANELLE_COLORS = {
    'Actin':        '#2ECC71',
    'Mitochondria': '#E74C3C',
    'Nucleus':      '#1558A0',
}

# ── load all four CSVs ────────────────────────────────────────────────────────
cell_l1  = pd.read_csv(PROJ / 'outputs/final_thesis_part1/cell/L1_pairwise.csv')
cell_l2  = pd.read_csv(PROJ / 'outputs/final_thesis_part1/cell/L2_age_corrected.csv')
pt_l1    = pd.read_csv(PROJ / 'outputs/final_thesis_part1/patient/L1_pairwise.csv')
pt_l2    = pd.read_csv(PROJ / 'outputs/final_thesis_part1/patient/L2_age_corrected.csv')

# ── feature order: sorted by cell L1 Cohen's d ascending (most negative first)
feature_order = (cell_l1.sort_values('Cohen_d', ascending=True)['Feature'].tolist())

def clean_label(feat):
    parts = feat.split('_', 1)
    return parts[1].replace('_', ' ') if len(parts) > 1 else feat.replace('_', ' ')

def sig_stars(q):
    if q < 0.001: return '***'
    if q < 0.01:  return '**'
    if q < 0.05:  return '*'
    return ''

# ── paper rc ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'Arial',
    'font.sans-serif':   ['Arial'],
    'font.size':         11,
    'font.weight':       'bold',
    'axes.labelsize':    12,
    'axes.labelweight':  'bold',
    'axes.titlesize':    11,
    'axes.titleweight':  'bold',
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'savefig.dpi':       300,
})

# ── figure ────────────────────────────────────────────────────────────────────
n = len(feature_order)
fig, axes = plt.subplots(
    2, 2,
    figsize=(13, n * 0.30 + 2.0),
    sharey=True,          # shared y-axis keeps features aligned across columns
)
fig.subplots_adjust(left=0.22, right=0.97, top=0.94, bottom=0.05,
                    hspace=0.28, wspace=0.22)

PANELS = [
    (axes[0, 0], cell_l1,  'A',  'Cell level — Disease',         'Cell\n(exploratory, BH-FDR)'),
    (axes[0, 1], cell_l2,  'B',  'Cell level — Age-corrected',   'Cell\n(age-residualised, BH-FDR)'),
    (axes[1, 0], pt_l1,    'C',  'Patient level — Disease',      'Patient\n(BH-FDR)'),
    (axes[1, 1], pt_l2,    'D',  'Patient level — Age-corrected','Patient\n(age-residualised, BH-FDR)'),
]

y_pos   = np.arange(n)
ylabels = [clean_label(f) for f in feature_order]

for ax, df, panel_label, title, footer_txt in PANELS:

    # align df to feature_order
    df = df.set_index('Feature').reindex(feature_order).reset_index()

    colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in df['Organelle']]
    d_vals = df['Cohen_d'].fillna(0).values

    bars = ax.barh(y_pos, d_vals, color=colors,
                   edgecolor='white', linewidth=0,
                   height=0.72, alpha=0.85)

    ax.axvline(0, color='#333333', lw=0.8)

    # significance stars
    xlim  = ax.get_xlim()
    pad   = (xlim[1] - xlim[0]) * 0.04
    star_xs = []
    for i, row in df.iterrows():
        if row.get('Significant', False):
            x  = row['Cohen_d']
            sx = x + (pad if x >= 0 else -pad)
            star_xs.append(sx)
            ax.text(sx, i, sig_stars(row['BH_q']),
                    va='center',
                    ha='left' if x >= 0 else 'right',
                    fontsize=8, color='#222222')

    # expand xlim so stars are never clipped
    if star_xs:
        extra     = (xlim[1] - xlim[0]) * 0.10
        new_left  = min(xlim[0], min(star_xs)) - extra
        new_right = max(xlim[1], max(star_xs)) + extra
        ax.set_xlim(new_left, new_right)

    # grid lines
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='#dddddd', linewidth=0.5, linestyle='--')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ylabels, fontsize=10, fontweight='bold')
    ax.set_xlabel("Cohen's $d$  (TAV-ATAA − TAV-NA)", fontsize=12, fontweight='bold')

    # panel label + title
    ax.text(-0.22, 1.02, panel_label,
            transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='bottom', ha='left')
    ax.set_title(title, fontsize=10, pad=4, loc='left', color='#333333', fontweight='bold')


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── shared legend ─────────────────────────────────────────────────────────────
patches = [mpatches.Patch(facecolor=ORGANELLE_COLORS[o], label=o, alpha=0.85)
           for o in ['Actin', 'Mitochondria', 'Nucleus']]
fig.legend(handles=patches,
           loc='upper center', ncol=3,
           frameon=False, fontsize=9,
           bbox_to_anchor=(0.60, 0.995))

# ── footnote ──────────────────────────────────────────────────────────────────
fig.text(0.60, 0.002,
         '* $q$<0.05   ** $q$<0.01   *** $q$<0.001',
         ha='center', fontsize=7.5, color='#666666', style='italic')

out_path = OUT / 'part1_cohens_d_composite.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f'Saved → {out_path}')
