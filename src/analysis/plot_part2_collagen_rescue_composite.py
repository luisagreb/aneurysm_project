"""
Composite collagen rescue figure — Part 2 (TAV-ATAA vs BAV-ATAA).
Shows L1-significant features grouped by organelle (Actin / Mitochondria / Nucleus).
Each panel: box+strip for 4 conditions:
  TAV-ATAA NoCollagen | TAV-ATAA +Collagen | BAV-ATAA NoCollagen | BAV-ATAA +Collagen
Output: outputs/figures/part2_collagen_rescue_composite.png
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc
from pathlib import Path
from scipy.stats import wilcoxon

def norm_id(s):
    m = re.search(r'(0[123][A-Za-z]+)-?0*(\d+)', str(s), re.IGNORECASE)
    return f"{m.group(1).lower()}-{m.group(2)}" if m else None

PROJ = Path(__file__).resolve().parents[2]
OUT  = PROJ / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# ── colours ───────────────────────────────────────────────────────────────────
COLOR_TAA = '#E74C3C'
COLOR_BAV = '#F39C12'
ORGANELLE_COLORS = {
    'Actin':        '#2ECC71',
    'Mitochondria': '#E74C3C',
    'Nucleus':      '#1558A0',
}

def lighten(hex_color, factor=0.45):
    rgb = mc.to_rgb(hex_color)
    return tuple(min(1.0, v + (1 - v) * factor) for v in rgb)

COLOR_TAA_COL = lighten(COLOR_TAA)
COLOR_BAV_COL = lighten(COLOR_BAV)

GROUP_LABELS = {
    'TAA': 'TAV-ATAA',
    'BAV': 'BAV-ATAA',
}

# ── features (organelle-grouped, L1 patient-level significant) ────────────────
FEATURES = [
    # Actin
    ('Actin',        'Actin_Extent_ratio',                  'Extent',               'ratio'),
    ('Actin',        'Actin_Skeleton_Length_µm',            'Skeleton length',       'µm'),
    ('Actin',        'Actin_Solidity_ratio',                'Solidity',              'ratio'),
    # Mitochondria
    ('Mitochondria', 'Mito_Branch_Count_n',                 'Branch count',          'n'),
    ('Mitochondria', 'Mito_Cyclomatic_Number_n',            'Cyclomatic number',     'n'),
    ('Mitochondria', 'Mito_Junction_Count_n',               'Junction count',        'n'),
    ('Mitochondria', 'Mito_Max_Fragment_Sphericity_ratio',  'Max. frag. sphericity', 'ratio'),
    ('Mitochondria', 'Mito_Mean_Branch_Length_µm',          'Mean branch length',    'µm'),
    ('Mitochondria', 'Mito_Mean_Tortuosity_ratio',          'Mean tortuosity',       'ratio'),
    ('Mitochondria', 'Mito_Std_Fragment_Sphericity_ratio',  'SD frag. sphericity',   'ratio'),
    ('Mitochondria', 'Mito_Total_Network_Length_µm',        'Total network length',  'µm'),
    # Nucleus
    ('Nucleus',      'Nucleus_Elongation_ratio',            'Elongation',            'ratio'),
    ('Nucleus',      'Nucleus_Sphericity_ratio',            'Sphericity',            'ratio'),
    ('Nucleus',      'Nucleus_Volume_µm³',                  'Volume',                'µm³'),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# ── load data ─────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(PROJ / 'outputs' / 'Advanced_Features_Raw_Final.csv')
df_raw = df_raw[df_raw['Disease'].isin(['TAA', 'BAV'])].copy()
df_raw['Subject'] = df_raw['CellName'].apply(norm_id)

# patient-level aggregated means
pt_agg = (df_raw.groupby(['Subject', 'Disease', 'Collagen_Status'])
          [[f[1] for f in FEATURES]].mean().reset_index())

res_pt = pd.read_csv(PROJ / 'outputs/final_thesis_part2/patient/L1_pairwise.csv')
res_pt = res_pt.set_index('Feature')

# ── paper rc ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'Arial',
    'font.sans-serif':    ['Arial'],
    'font.size':          10,
    'axes.labelsize':     10,
    'axes.titlesize':     11,
    'xtick.labelsize':    8.5,
    'ytick.labelsize':    9,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'savefig.dpi':        300,
    'text.usetex':        False,
})

# ── figure layout: 3 cols × 5 rows ───────────────────────────────────────────
NCOLS, NROWS = 3, 5
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(11, 13),
                         constrained_layout=False)
fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.05,
                    hspace=0.75, wspace=0.42)

rng = np.random.default_rng(42)
positions  = [0, 1, 2.3, 3.3]
xtick_lbls = ['TAV\nNoCol', 'TAV\n+Col', 'BAV\nNoCol', 'BAV\n+Col']
box_colors = [COLOR_TAA, COLOR_TAA_COL, COLOR_BAV, COLOR_BAV_COL]
hatches    = [None, '///', None, '///']

for idx, (organelle, col, name, unit) in enumerate(FEATURES):
    row, c = divmod(idx, NCOLS)
    ax = axes[row, c]
    org_color = ORGANELLE_COLORS[organelle]

    # collect 4 groups at patient level
    v_taa_nc  = pt_agg[(pt_agg['Disease'] == 'TAA') & (pt_agg['Collagen_Status'] == 'NoCollagen')][col].dropna().values
    v_taa_col = pt_agg[(pt_agg['Disease'] == 'TAA') & (pt_agg['Collagen_Status'] == 'Collagen')][col].dropna().values
    v_bav_nc  = pt_agg[(pt_agg['Disease'] == 'BAV') & (pt_agg['Collagen_Status'] == 'NoCollagen')][col].dropna().values
    v_bav_col = pt_agg[(pt_agg['Disease'] == 'BAV') & (pt_agg['Collagen_Status'] == 'Collagen')][col].dropna().values
    groups = [v_taa_nc, v_taa_col, v_bav_nc, v_bav_col]

    # ── box plot ──────────────────────────────────────────────────────────────
    bp = ax.boxplot(
        groups, positions=positions, widths=0.5,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker='', linestyle='none'),
        boxprops=dict(linewidth=0.8),
    )
    for patch, fc, h in zip(bp['boxes'], box_colors, hatches):
        patch.set_facecolor(fc)
        patch.set_alpha(0.40)
        if h:
            patch.set_hatch(h)

    # ── strip plot ────────────────────────────────────────────────────────────
    for xi, v, fc in zip(positions, groups, box_colors):
        jitter = rng.uniform(-0.15, 0.15, size=len(v))
        ax.scatter(xi + jitter, v, color=fc, s=14, alpha=0.75,
                   linewidths=0, zorder=3)

    # ── significance brackets ─────────────────────────────────────────────────
    all_vals = np.concatenate([v for v in groups if len(v)])
    ymax  = np.percentile(all_vals, 97) if len(all_vals) else 1
    ymin  = np.percentile(all_vals, 3)  if len(all_vals) else 0
    yspan = max(ymax - ymin, 1e-9)
    bracket_y = ymax + 0.10 * yspan
    tick = 0.03 * yspan

    def draw_bracket(ax, x1, x2, y, txt):
        ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick],
                lw=0.8, color='#333333')
        ax.text((x1 + x2) / 2, y + 0.005 * yspan, txt,
                ha='center', va='bottom', fontsize=8.5, color='#333333')

    # disease effect bracket (TAV-ATAA NoCol vs BAV-ATAA NoCol)
    q_dis = res_pt.loc[col, 'BH_q'] if col in res_pt.index else 1.0
    draw_bracket(ax, positions[0], positions[2], bracket_y, sig_stars(q_dis))

    # TAV-ATAA rescue (NoCol vs +Col), uncorrected
    if len(v_taa_nc) >= 4 and len(v_taa_col) >= 4:
        # paired wilcoxon if we can match subjects
        nc_s  = pt_agg[(pt_agg['Disease'] == 'TAA') & (pt_agg['Collagen_Status'] == 'NoCollagen')].set_index('Subject')[col].dropna()
        col_s = pt_agg[(pt_agg['Disease'] == 'TAA') & (pt_agg['Collagen_Status'] == 'Collagen')].set_index('Subject')[col].dropna()
        common = nc_s.index.intersection(col_s.index)
        if len(common) >= 4:
            try:
                _, p_taa = wilcoxon(col_s.loc[common].values - nc_s.loc[common].values)
            except Exception:
                p_taa = np.nan
        else:
            p_taa = np.nan
        if not np.isnan(p_taa) and p_taa < 0.05:
            rescue_y = bracket_y + 0.16 * yspan
            draw_bracket(ax, positions[0], positions[1], rescue_y, sig_stars(p_taa))

    # BAV-ATAA rescue (NoCol vs +Col), uncorrected
    if len(v_bav_nc) >= 4 and len(v_bav_col) >= 4:
        nc_s  = pt_agg[(pt_agg['Disease'] == 'BAV') & (pt_agg['Collagen_Status'] == 'NoCollagen')].set_index('Subject')[col].dropna()
        col_s = pt_agg[(pt_agg['Disease'] == 'BAV') & (pt_agg['Collagen_Status'] == 'Collagen')].set_index('Subject')[col].dropna()
        common = nc_s.index.intersection(col_s.index)
        if len(common) >= 4:
            try:
                _, p_bav = wilcoxon(col_s.loc[common].values - nc_s.loc[common].values)
            except Exception:
                p_bav = np.nan
        else:
            p_bav = np.nan
        if not np.isnan(p_bav) and p_bav < 0.05:
            rescue_y2 = bracket_y + 0.16 * yspan
            draw_bracket(ax, positions[2], positions[3], rescue_y2, sig_stars(p_bav))

    # ── axes labels ───────────────────────────────────────────────────────────
    y_label = name if unit == 'ratio' else f'{name} ({unit})'
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(xtick_lbls, fontsize=8)
    ax.set_xlim(-0.6, 4.0)

    ax.set_title(name, fontsize=11, fontweight='bold', pad=4, color=org_color)

# ── hide unused panel (14 features in 3×5=15 grid) ───────────────────────────
axes[4, 2].set_visible(False)

# ── organelle legend ──────────────────────────────────────────────────────────
org_patches = [mpatches.Patch(facecolor=ORGANELLE_COLORS[o], label=o, alpha=0.85)
               for o in ['Actin', 'Mitochondria', 'Nucleus']]
fig.legend(handles=org_patches, loc='upper center', ncol=3,
           frameon=False, fontsize=9, bbox_to_anchor=(0.42, 0.975))

# ── condition legend ──────────────────────────────────────────────────────────
cond_patches = [
    mpatches.Patch(facecolor=COLOR_TAA,     alpha=0.45, label='TAV-ATAA  NoCollagen'),
    mpatches.Patch(facecolor=COLOR_TAA_COL, alpha=0.45, label='TAV-ATAA  +Collagen', hatch='///'),
    mpatches.Patch(facecolor=COLOR_BAV,     alpha=0.45, label='BAV-ATAA  NoCollagen'),
    mpatches.Patch(facecolor=COLOR_BAV_COL, alpha=0.45, label='BAV-ATAA  +Collagen', hatch='///'),
]
fig.legend(handles=cond_patches, loc='upper right', ncol=2,
           frameon=False, fontsize=8.5, bbox_to_anchor=(0.99, 0.975))

# ── footnote ──────────────────────────────────────────────────────────────────
fig.text(0.5, 0.005,
         '* $p$<0.05   ** $p$<0.01   *** $p$<0.001   '
         '(disease bracket: BH-FDR $q$; rescue brackets: Wilcoxon paired, uncorrected)',
         ha='center', fontsize=7.5, color='#666666', style='italic')

out_path = OUT / 'part2_collagen_rescue_composite.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f'Saved → {out_path}')
