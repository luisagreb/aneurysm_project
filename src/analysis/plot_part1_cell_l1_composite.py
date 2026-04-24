"""
Composite figure: Part 1 cell-level L1 significant features (TAV-NA vs TAV-ATAA).
12 panels (3 cols × 4 rows), only NoCollagen cells, box + strip style.
Output: outputs/figures/part1_cell_l1_composite.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import mannwhitneyu

PROJ = Path(__file__).resolve().parents[2]
OUT  = PROJ / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# ── colours ───────────────────────────────────────────────────────────────────
COLOR = {'Healthy': '#4C9BE8', 'TAA': '#E74C3C'}
LABEL = {'Healthy': 'TAV-NA', 'TAA': 'TAV-ATAA'}
ORGANELLE_COLOR = {
    'Actin':        '#2ECC71',
    'Mitochondria': '#E74C3C',
    'Nucleus':      '#1558A0',
}

# ── 12 significant cell-level features (organelle, raw_col, display_name, unit)
FEATURES = [
    # Actin
    ('Actin',        'Actin_Solidity_ratio',               'Solidity',                'ratio'),
    ('Actin',        'Actin_Extent_ratio',                 'Extent',                  'ratio'),
    ('Actin',        'Actin_Skeleton_Length_µm',           'Skeleton length',         'µm'),
    # Mitochondria
    ('Mitochondria', 'Mito_Min_Fragment_Sphericity_ratio', 'Min. fragment sphericity', 'ratio'),
    ('Mitochondria', 'Mito_Sphericity_ratio',              'Network sphericity',       'ratio'),
    ('Mitochondria', 'Mito_Branch_Count_n',                'Branch count',             'n'),
    ('Mitochondria', 'Mito_Junction_Count_n',              'Junction count',           'n'),
    ('Mitochondria', 'Mito_Total_Network_Length_µm',       'Total network length',     'µm'),
    ('Mitochondria', 'Mito_Fragment_Count_n',              'Fragment count',           'n'),
    ('Mitochondria', 'Mito_Cyclomatic_Number_n',           'Cyclomatic number',        'n'),
    ('Mitochondria', 'Mito_Surface_Area_µm²',              'Surface area',             'µm²'),
    # Nucleus
    ('Nucleus',      'Nucleus_Sphericity_ratio',           'Sphericity',               'ratio'),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    s = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(b) - np.mean(a)) / s if s > 0 else 0.0

def sig_stars(q):
    if q < 0.001: return '***'
    if q < 0.01:  return '**'
    if q < 0.05:  return '*'
    return 'ns'

# q-values from L1_pairwise.csv (cell level, BH-FDR)
Q_VALUES = {
    'Actin_Solidity_ratio':               4.89e-6,
    'Actin_Extent_ratio':                 4.89e-6,
    'Actin_Skeleton_Length_µm':           1.44e-4,
    'Mito_Min_Fragment_Sphericity_ratio': 9.03e-6,
    'Mito_Sphericity_ratio':              7.87e-5,
    'Mito_Branch_Count_n':                1.94e-3,
    'Mito_Junction_Count_n':              2.43e-3,
    'Mito_Total_Network_Length_µm':       8.32e-4,
    'Mito_Fragment_Count_n':              2.05e-3,
    'Mito_Cyclomatic_Number_n':           1.59e-2,
    'Mito_Surface_Area_µm²':              5.10e-3,
    'Nucleus_Sphericity_ratio':           2.93e-2,
}

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(PROJ / 'outputs' / 'Advanced_Features_Raw_Final.csv')
df = df[df['Collagen_Status'] == 'NoCollagen']
df = df[df['Disease'].isin(['Healthy', 'TAA'])]

# ── paper rc ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'Arial',
    'font.sans-serif':   ['Arial'],
    'font.size':         10,
    'axes.labelsize':    10,
    'axes.titlesize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
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
NCOLS, NROWS = 3, 4
fig, axes = plt.subplots(NROWS, NCOLS,
                         figsize=(7.2, 9.6),
                         constrained_layout=False)
fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.06,
                    hspace=0.72, wspace=0.42)

rng = np.random.default_rng(42)

for idx, (organelle, col, name, unit) in enumerate(FEATURES):
    row, c = divmod(idx, NCOLS)
    ax = axes[row, c]

    org_color = ORGANELLE_COLOR[organelle]

    vals = {}
    for grp in ['Healthy', 'TAA']:
        v = df.loc[df['Disease'] == grp, col].dropna().values
        vals[grp] = v

    # ── box plot ──────────────────────────────────────────────────────────────
    positions = [0, 1]
    bp = ax.boxplot(
        [vals['Healthy'], vals['TAA']],
        positions=positions,
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker='', linestyle='none'),
        boxprops=dict(linewidth=0.8),
    )
    for patch, grp in zip(bp['boxes'], ['Healthy', 'TAA']):
        patch.set_facecolor(COLOR[grp])
        patch.set_alpha(0.35)

    # ── strip plot ────────────────────────────────────────────────────────────
    for xi, grp in zip(positions, ['Healthy', 'TAA']):
        v = vals[grp]
        jitter = rng.uniform(-0.18, 0.18, size=len(v))
        ax.scatter(xi + jitter, v,
                   color=COLOR[grp], s=10, alpha=0.65,
                   linewidths=0, zorder=3)

    # ── significance bracket ──────────────────────────────────────────────────
    q     = Q_VALUES[col]
    d     = cohens_d(vals['Healthy'], vals['TAA'])
    stars = sig_stars(q)

    ymax  = max(np.percentile(vals['Healthy'], 97),
                np.percentile(vals['TAA'],     97))
    ymin  = min(np.percentile(vals['Healthy'], 3),
                np.percentile(vals['TAA'],     3))
    yspan = ymax - ymin
    bar_y = ymax + 0.10 * yspan
    tick  = 0.03 * yspan

    ax.plot([0, 0, 1, 1], [bar_y - tick, bar_y, bar_y, bar_y - tick],
            lw=0.9, color='#333333')
    ax.text(0.5, bar_y + 0.01 * yspan, stars,
            ha='center', va='bottom', fontsize=10, color='#333333')

    # ── axes labels ───────────────────────────────────────────────────────────
    y_label = name if unit == 'ratio' else f'{name} ({unit})'
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['TAV-NA', 'TAV-ATAA'], fontsize=9)
    ax.set_xlim(-0.55, 1.55)

    # ── panel title ───────────────────────────────────────────────────────────
    ax.set_title(name, fontsize=11, fontweight='bold', pad=4,
                 color=org_color)

    # ── d annotation ─────────────────────────────────────────────────────────
    ax.text(0.97, 0.97, f'$d$ = {d:+.2f}',
            transform=ax.transAxes, fontsize=8,
            ha='right', va='top', color='#555555')

# ── organelle legend (top centre) ────────────────────────────────────────────
patches = [mpatches.Patch(facecolor=ORGANELLE_COLOR[o], label=o, alpha=0.85)
           for o in ['Actin', 'Mitochondria', 'Nucleus']]
fig.legend(handles=patches, loc='upper center', ncol=3,
           frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 0.975))

# ── group legend (top right) ──────────────────────────────────────────────────
grp_patches = [mpatches.Patch(facecolor=COLOR[g], alpha=0.55, label=LABEL[g])
               for g in ['Healthy', 'TAA']]
fig.legend(handles=grp_patches, loc='upper right',
           frameon=False, fontsize=9,
           bbox_to_anchor=(0.99, 0.975))

# ── footnote ─────────────────────────────────────────────────────────────────
fig.text(0.5, 0.005,
         '* $q$<0.05   ** $q$<0.01   *** $q$<0.001   '
         '(BH-FDR corrected, cell level, no-collagen condition)',
         ha='center', fontsize=7.5, color='#666666', style='italic')

out_path = OUT / 'part1_cell_l1_composite.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f'Saved → {out_path}')
