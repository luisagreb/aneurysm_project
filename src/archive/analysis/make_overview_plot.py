"""
Two overview figures:

  1. overview_pairwise.png  — pairwise disease effect (TAV-NA vs TAV-ATAA vs BAV-ATAA)
                               3 panels, one per comparison, Cohen's d sorted bars.

  2. overview_collagen.png  — collagen effect within each group (paired Cohen's d_z)
                               3 panels (TAV-NA / TAV-ATAA / BAV-ATAA),
                               positive = Collagen > NoCollagen.

Style: horizontal bars, red = FDR<0.05, gray = ns, stars annotated.

Inputs: classification_results/three_group_analysis/layer2_pairwise_raw.csv
        classification_results/three_group_analysis/layer6_collagen/collagen_summary.csv
Output: classification_results/three_group_analysis/overview_pairwise.png
        classification_results/three_group_analysis/overview_collagen.png

Usage:
  cd /home/luisa/aneurysm_project
  python src/analysis/make_overview_plot.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PW_FILE   = 'classification_results/three_group_analysis/layer2_pairwise_raw.csv'
COL_FILE  = 'classification_results/three_group_analysis/layer6_collagen/collagen_summary.csv'
OUT_FILE  = Path('classification_results/three_group_analysis/overview_pairwise.png')
OUT_FILE2 = Path('classification_results/three_group_analysis/overview_collagen.png')

PAIR_LABELS = ['TAV-NA vs TAV-ATAA', 'TAV-NA vs BAV-ATAA', 'TAV-ATAA vs BAV-ATAA']
PAIR_TITLES = [
    'TAV-NA  vs  TAV-ATAA\nPositive = higher in TAV-ATAA',
    'TAV-NA  vs  BAV-ATAA\nPositive = higher in BAV-ATAA',
    'TAV-ATAA  vs  BAV-ATAA\nPositive = higher in BAV-ATAA',
]

COLOR_SIG = '#E53935'   # red — FDR < 0.05
COLOR_NS  = '#9E9E9E'   # gray — not significant

def stars(p):
    if np.isnan(p): return ''
    if p < 0.001:   return '***'
    if p < 0.01:    return '**'
    if p < 0.05:    return '*'
    return ''

# ── Load ──────────────────────────────────────────────────────────────────────
pw = pd.read_csv(PW_FILE)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=False)
fig.subplots_adjust(wspace=0.45, left=0.05, right=0.97, top=0.88, bottom=0.08)

for ax, label, title in zip(axes, PAIR_LABELS, PAIR_TITLES):
    sub = pw[pw['Comparison'] == label].copy()
    sub = sub.sort_values('Cohens_d')          # sort ascending → most negative at bottom

    colors = [COLOR_SIG if s else COLOR_NS for s in sub['Significant']]
    y      = np.arange(len(sub))

    ax.barh(y, sub['Cohens_d'], color=colors, height=0.65, edgecolor='none')

    # Star labels
    for i, (_, row) in enumerate(sub.iterrows()):
        s = stars(row['p_FDR'])
        if s:
            x_off = 0.02 if row['Cohens_d'] >= 0 else -0.02
            ha    = 'left' if row['Cohens_d'] >= 0 else 'right'
            ax.text(row['Cohens_d'] + x_off, i, s,
                    va='center', ha=ha, fontsize=9,
                    color='black', fontweight='bold')

    ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(sub['Feature'].str.replace('_', ' '), fontsize=9)
    ax.set_xlabel("Cohen's d", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='x', color='#eeeeee', lw=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate p-scale once per panel (bottom right)
    ax.text(0.99, 0.01, '* p<0.05  ** p<0.01  *** p<0.001\n(BH-FDR corrected)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.5, color='#555555')

# Legend (shared)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=COLOR_SIG, label='FDR < 0.05'),
    Patch(facecolor=COLOR_NS,  label='FDR ≥ 0.05'),
]
fig.legend(handles=legend_handles, loc='upper center', ncol=2,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.97))

fig.suptitle('Pairwise Disease Effect — TAV-NA vs TAV-ATAA vs BAV-ATAA\n'
             'Shapiro-Wilk → T-test or Mann-Whitney  |  NoCollagen cells',
             fontsize=12, fontweight='bold', y=1.02)

plt.savefig(OUT_FILE, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_FILE}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Collagen effect (paired Cohen's d_z, within each group)
# ══════════════════════════════════════════════════════════════════════════════

coll = pd.read_csv(COL_FILE)

# Columns: Feature, TAV_NA_dz, TAV_NA_FDR, TAV_NA_sig,
#                   TAV_ATAA_dz, TAV_ATAA_FDR, TAV_ATAA_sig,
#                   BAV_ATAA_dz, BAV_ATAA_FDR, BAV_ATAA_sig

GROUPS_COL = [
    ('TAV_NA',   'TAV-NA\n(N=10 paired specimens)'),
    ('TAV_ATAA', 'TAV-ATAA\n(N=10 paired specimens)'),
    ('BAV_ATAA', 'BAV-ATAA\n(N=4 paired specimens)'),
]

fig2, axes2 = plt.subplots(1, 3, figsize=(22, 14), sharey=False)
fig2.subplots_adjust(wspace=0.60, left=0.09, right=0.97, top=0.82, bottom=0.06)

for ax, (key, title) in zip(axes2, GROUPS_COL):
    dz_col  = f'{key}_dz'
    fdr_col = f'{key}_FDR'
    sig_col = f'{key}_sig'

    sub = coll[['Feature', dz_col, fdr_col, sig_col]].copy()
    sub = sub.rename(columns={dz_col: 'dz', fdr_col: 'FDR', sig_col: 'sig'})
    sub = sub.dropna(subset=['dz'])
    sub = sub.sort_values('dz')

    colors = [COLOR_SIG if s else COLOR_NS for s in sub['sig']]
    y      = np.arange(len(sub))

    ax.barh(y, sub['dz'], color=colors, height=0.65, edgecolor='none')

    for i, (_, row) in enumerate(sub.iterrows()):
        s = stars(row['FDR'])
        if s:
            x_off = 0.02 if row['dz'] >= 0 else -0.02
            ha    = 'left' if row['dz'] >= 0 else 'right'
            ax.text(row['dz'] + x_off, i, s,
                    va='center', ha=ha, fontsize=13,
                    color='black', fontweight='bold')

    ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(sub['Feature'].str.replace('_', ' '), fontsize=13)
    ax.set_xlabel("Cohen's d_z  (Collagen − NoCollagen, paired)", fontsize=13)
    ax.set_title(f"{title}\nPositive = higher with collagen",
                 fontsize=15, fontweight='bold', pad=14)
    ax.grid(axis='x', color='#eeeeee', lw=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.99, 0.01, '* p<0.05  ** p<0.01  *** p<0.001\n(BH-FDR corrected, paired test)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, color='#555555')

from matplotlib.patches import Patch as _Patch
fig2.legend(
    handles=[_Patch(facecolor=COLOR_SIG, label='FDR < 0.05'),
             _Patch(facecolor=COLOR_NS,  label='FDR ≥ 0.05')],
    loc='upper center', ncol=2, fontsize=13,
    framealpha=0.9, bbox_to_anchor=(0.5, 0.95),
)
fig2.suptitle('Collagen Effect on SMC Morphology — Within Each Group\n'
              'Paired analysis: same specimen, NoCollagen vs Collagen condition',
              fontsize=16, fontweight='bold', y=1.00)

plt.savefig(OUT_FILE2, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_FILE2}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Collagen effect treating cells as independent observations
# (cell-level Mann-Whitney, NOT paired specimen-level)
# NOTE: this inflates N (cells are not independent — they cluster by patient)
#       but is shown for comparison and to match prior analyses.
# ══════════════════════════════════════════════════════════════════════════════

from scipy.stats import mannwhitneyu, shapiro, ttest_ind
from statsmodels.stats.multitest import multipletests

OUT_FILE3 = Path('classification_results/three_group_analysis/overview_collagen_celllevel.png')

FEATURE_COLS = [
    'Actin_Volume', 'Actin_Skeleton_Length', 'Actin_Solidity',
    'Actin_Fractional_Anisotropy', 'Actin_Major_Axis', 'Actin_Minor_Axis',
    'Mito_Volume', 'Mito_Sphericity', 'Mito_Fragment_Count',
    'Mito_Mean_Fragment_Volume', 'Mito_Branch_Count',
    'Mito_Total_Network_Length', 'Mito_Mean_Tortuosity',
    'Mito_Junction_Count', 'Mito_Cyclomatic_Number',
    'Nucleus_Volume', 'Nucleus_Sphericity', 'Nucleus_Circularity',
    'Nucleus_Elongation', 'Nucleus_Flatness', 'Nucleus_Solidity',
]

df_raw = pd.read_csv('outputs/Advanced_Features_3groups.csv')
df_raw = df_raw[df_raw['Collagen_Status'].isin(['Collagen', 'NoCollagen'])].copy()

def adaptive_test_cells(d1, d2):
    """Shapiro-Wilk → Welch T-test or Mann-Whitney (cell level)."""
    def is_norm(x):
        if len(x) < 3 or len(x) > 5000: return False
        return shapiro(x)[1] > 0.05
    if is_norm(d1) and is_norm(d2):
        return ttest_ind(d1, d2, equal_var=False)[1], 'T-test'
    return mannwhitneyu(d1, d2, alternative='two-sided')[1], 'Mann-Whitney'

GROUPS_LIST = ['TAV-NA', 'TAV-ATAA', 'BAV-ATAA']
GROUPS_COL3 = [
    ('TAV-NA',   'TAV-NA'),
    ('TAV-ATAA', 'TAV-ATAA'),
    ('BAV-ATAA', 'BAV-ATAA'),
]

# Run tests per group
results_by_group = {}
for grp, grp_label in GROUPS_COL3:
    sub = df_raw[df_raw['Group'] == grp]
    rows = []
    for feat in FEATURE_COLS:
        d_nc  = sub[sub['Collagen_Status'] == 'NoCollagen'][feat].dropna().values
        d_col = sub[sub['Collagen_Status'] == 'Collagen'][feat].dropna().values
        if len(d_nc) < 3 or len(d_col) < 3:
            rows.append({'Feature': feat, 'Cohens_d': np.nan,
                         'p_raw': np.nan, 'N_nc': len(d_nc), 'N_col': len(d_col)})
            continue
        p, test = adaptive_test_cells(d_nc, d_col)
        # Cohen's d: positive = Collagen > NoCollagen
        pooled_sd = np.sqrt((d_nc.std(ddof=1)**2 + d_col.std(ddof=1)**2) / 2)
        d = (d_col.mean() - d_nc.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        rows.append({'Feature': feat, 'Cohens_d': d, 'p_raw': p,
                     'N_nc': len(d_nc), 'N_col': len(d_col), 'Test': test})
    res = pd.DataFrame(rows)
    _, fdr, _, _ = multipletests(res['p_raw'].fillna(1), method='fdr_bh')
    res['p_FDR'] = fdr
    res['sig']   = fdr < 0.05
    results_by_group[grp] = res

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 8), sharey=False)
fig3.subplots_adjust(wspace=0.45, left=0.05, right=0.97, top=0.88, bottom=0.08)

for ax, (grp, grp_label) in zip(axes3, GROUPS_COL3):
    res = results_by_group[grp]
    res = res.dropna(subset=['Cohens_d']).sort_values('Cohens_d')

    n_nc  = int(res['N_nc'].iloc[0])
    n_col = int(res['N_col'].iloc[0])

    colors = [COLOR_SIG if s else COLOR_NS for s in res['sig']]
    y      = np.arange(len(res))

    ax.barh(y, res['Cohens_d'], color=colors, height=0.65, edgecolor='none')

    for i, (_, row) in enumerate(res.iterrows()):
        s = stars(row['p_FDR'])
        if s:
            x_off = 0.02 if row['Cohens_d'] >= 0 else -0.02
            ha    = 'left' if row['Cohens_d'] >= 0 else 'right'
            ax.text(row['Cohens_d'] + x_off, i, s,
                    va='center', ha=ha, fontsize=9,
                    color='black', fontweight='bold')

    ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(res['Feature'].str.replace('_', ' '), fontsize=9)
    ax.set_xlabel("Cohen's d  (Collagen − NoCollagen)", fontsize=10)
    ax.set_title(f"{grp_label}\nNoCollagen N={n_nc} cells  |  Collagen N={n_col} cells\n"
                 f"Positive = higher with collagen",
                 fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='x', color='#eeeeee', lw=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.99, 0.01,
            '* p<0.05  ** p<0.01  *** p<0.001\n(BH-FDR corrected, cell-level)\n'
            '⚠ cells treated as independent',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.5, color='#777777')

from matplotlib.patches import Patch as _Patch2
fig3.legend(
    handles=[_Patch2(facecolor=COLOR_SIG, label='FDR < 0.05'),
             _Patch2(facecolor=COLOR_NS,  label='FDR ≥ 0.05')],
    loc='upper center', ncol=2, fontsize=10,
    framealpha=0.9, bbox_to_anchor=(0.5, 0.97),
)
fig3.suptitle('Collagen Effect on SMC Morphology — Cell-Level Analysis\n'
              'Shapiro-Wilk → T-test or Mann-Whitney  |  Cells treated as independent observations',
              fontsize=12, fontweight='bold', y=1.02)

plt.savefig(OUT_FILE3, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_FILE3}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — TAV-NA NoCollagen as reference vs TAA −Col and TAA +Col
#
# Two panels side by side: TAV-ATAA | BAV-ATAA
# Each panel has TWO bars per feature:
#   Light bar  = Cohen's d  (TAV-NA NoCol  vs  TAA NoCol)   — disease baseline
#   Dark bar   = Cohen's d  (TAV-NA NoCol  vs  TAA +Col)    — collagen-treated
# The reference group (TAV-NA NoCol) is always the left/first group → d=0 is "healthy".
# Positive d = TAA group higher than healthy NoCol.
# Cells treated as independent (cell-level Mann-Whitney / T-test).
# ══════════════════════════════════════════════════════════════════════════════

OUT_FILE4 = Path('classification_results/three_group_analysis/overview_healthy_vs_taa_collagen.png')

# Build group-condition subsets
healthy_nc = df_raw[(df_raw['Group'] == 'TAV-NA') &
                    (df_raw['Collagen_Status'] == 'NoCollagen')]

subsets = {
    'TAV-ATAA': {
        'NoCol': df_raw[(df_raw['Group'] == 'TAV-ATAA') &
                        (df_raw['Collagen_Status'] == 'NoCollagen')],
        'Col':   df_raw[(df_raw['Group'] == 'TAV-ATAA') &
                        (df_raw['Collagen_Status'] == 'Collagen')],
    },
    'BAV-ATAA': {
        'NoCol': df_raw[(df_raw['Group'] == 'BAV-ATAA') &
                        (df_raw['Collagen_Status'] == 'NoCollagen')],
        'Col':   df_raw[(df_raw['Group'] == 'BAV-ATAA') &
                        (df_raw['Collagen_Status'] == 'Collagen')],
    },
}

def compute_vs_healthy(ref_df, taa_df):
    """Cohen's d and FDR for each feature: taa_df vs ref_df (cell-level)."""
    rows = []
    for feat in FEATURE_COLS:
        d1 = ref_df[feat].dropna().values
        d2 = taa_df[feat].dropna().values
        if len(d1) < 3 or len(d2) < 3:
            rows.append({'Feature': feat, 'Cohens_d': np.nan,
                         'p_raw': np.nan, 'N1': len(d1), 'N2': len(d2)})
            continue
        p, test = adaptive_test_cells(d1, d2)
        pooled = np.sqrt((d1.std(ddof=1)**2 + d2.std(ddof=1)**2) / 2)
        d = (d2.mean() - d1.mean()) / pooled if pooled > 0 else np.nan
        rows.append({'Feature': feat, 'Cohens_d': d, 'p_raw': p,
                     'N1': len(d1), 'N2': len(d2)})
    res = pd.DataFrame(rows)
    _, fdr, _, _ = multipletests(res['p_raw'].fillna(1), method='fdr_bh')
    res['p_FDR'] = fdr
    res['sig']   = fdr < 0.05
    return res.set_index('Feature')

# Compute all 4 comparisons
results = {}
for grp in ['TAV-ATAA', 'BAV-ATAA']:
    results[(grp, 'NoCol')] = compute_vs_healthy(healthy_nc, subsets[grp]['NoCol'])
    results[(grp, 'Col')]   = compute_vs_healthy(healthy_nc, subsets[grp]['Col'])

# Sort features by mean |d| across all 4 comparisons for a stable order
mean_abs = pd.Series(0.0, index=FEATURE_COLS)
for key, res in results.items():
    mean_abs += res['Cohens_d'].abs().reindex(FEATURE_COLS).fillna(0)
feat_order = mean_abs.sort_values().index.tolist()   # ascending → most different at top after invert

# Colors: light = NoCol condition, dark = +Col condition
COLORS = {
    'TAV-ATAA': {'NoCol': '#FFAB9F', 'Col': '#D73B2F'},   # light/dark red
    'BAV-ATAA': {'NoCol': '#FFD180', 'Col': '#E65100'},   # light/dark orange
}
BAR_HEIGHT  = 0.32
BAR_OFFSETS = {'NoCol': -BAR_HEIGHT * 0.55, 'Col': BAR_HEIGHT * 0.55}

n_feat4 = len(feat_order)
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
fig4.subplots_adjust(wspace=0.08, left=0.18, right=0.97, top=0.88, bottom=0.10)

n_ref = len(healthy_nc)

for ax, grp in zip(axes4, ['TAV-ATAA', 'BAV-ATAA']):
    y = np.arange(n_feat4)

    for cond, offset in BAR_OFFSETS.items():
        res  = results[(grp, cond)]
        d_vals   = [res.loc[f, 'Cohens_d'] for f in feat_order]
        sig_vals = [res.loc[f, 'sig']      for f in feat_order]
        fdr_vals = [res.loc[f, 'p_FDR']   for f in feat_order]
        color    = COLORS[grp][cond]

        for i, (d, sig, fdr) in enumerate(zip(d_vals, sig_vals, fdr_vals)):
            if np.isnan(d):
                continue
            alpha = 1.0 if sig else 0.45
            ax.barh(y[i] + offset, d, height=BAR_HEIGHT,
                    color=color, alpha=alpha, edgecolor='none')
            s = stars(fdr)
            if s:
                x_off = 0.02 if d >= 0 else -0.02
                ha    = 'left' if d >= 0 else 'right'
                ax.text(d + x_off, y[i] + offset, s,
                        va='center', ha=ha, fontsize=7.5,
                        color='#222222', fontweight='bold')

    n_nc  = len(subsets[grp]['NoCol'])
    n_col = len(subsets[grp]['Col'])
    ax.axvline(0, color='black', lw=1.0)
    ax.set_title(f"TAV-NA −Col  (N={n_ref})  vs  {grp}\n"
                 f"NoCollagen N={n_nc}  |  +Collagen N={n_col}\n"
                 f"Positive = higher than healthy NoCol",
                 fontsize=10.5, fontweight='bold', pad=10)
    ax.set_xlabel("Cohen's d  (TAA − TAV-NA NoCol)", fontsize=10)
    ax.grid(axis='x', color='#eeeeee', lw=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.99, 0.01,
            '* p<0.05  ** p<0.01  *** p<0.001\n(BH-FDR, cell-level, cells independent)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.5, color='#777777')

axes4[0].set_yticks(np.arange(n_feat4))
axes4[0].set_yticklabels([f.replace('_', ' ') for f in feat_order], fontsize=9)

# Legend
from matplotlib.patches import Patch as _P
legend_handles4 = [
    _P(facecolor=COLORS['TAV-ATAA']['NoCol'], alpha=1.0, label='TAV-ATAA  −Collagen (disease baseline)'),
    _P(facecolor=COLORS['TAV-ATAA']['Col'],   alpha=1.0, label='TAV-ATAA  +Collagen (collagen-treated)'),
    _P(facecolor=COLORS['BAV-ATAA']['NoCol'], alpha=1.0, label='BAV-ATAA  −Collagen (disease baseline)'),
    _P(facecolor=COLORS['BAV-ATAA']['Col'],   alpha=1.0, label='BAV-ATAA  +Collagen (collagen-treated)'),
    _P(facecolor='#aaaaaa', alpha=0.45, label='FDR ≥ 0.05 (faded)'),
]
fig4.legend(handles=legend_handles4, loc='lower center', ncol=5,
            fontsize=8.5, framealpha=0.95, bbox_to_anchor=(0.5, 0.01))

fig4.suptitle('SMC Morphology: TAV-NA −Collagen as Reference\n'
              'Disease effect (−Col) vs Collagen-treated effect (+Col)',
              fontsize=12, fontweight='bold', y=1.01)

plt.savefig(OUT_FILE4, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_FILE4}")

# ══════════════════════════════════════════════════════════════════════════════
import matplotlib.patches as mpatches

# Figure 5 — 2×3 grid, top 10 features per panel, two bars per feature
#
# Layout (matching reference style):
#   Row 1:
#     Panel 1 — Disease Effect (−Collagen): TAV-NA vs TAV-ATAA  +  TAV-NA vs BAV-ATAA
#     Panel 2 — Disease Effect (+Collagen): same groups in collagen condition
#     Panel 3 — TAV-ATAA vs BAV-ATAA: NoCollagen + Collagen
#   Row 2:
#     Panel 4 — Collagen effect in TAV-NA (single bar)
#     Panel 5 — Collagen effect in TAV-ATAA + BAV-ATAA (two bars)
#     Panel 6 — Cross: TAV-NA NoColl vs TAA +Coll  +  TAV-NA NoColl vs TAA −Coll
#
# For dual-bar panels: top sub-bar = comparison A, bottom = comparison B.
# Color encodes comparison identity (not significance).
# Alpha: 1.0 = FDR<0.05, 0.38 = ns.
# Stars annotated at bar tip.
# Top 10 selected by max(|d_A|, |d_B|) across both comparisons.
# ══════════════════════════════════════════════════════════════════════════════

OUT_FILE5 = Path('classification_results/three_group_analysis/overview_grid.png')

def compare_cell(grp1, cond1, grp2, cond2):
    """Cell-level comparison with FDR. Returns df indexed by Feature."""
    d1_df = df_raw[(df_raw['Group'] == grp1) & (df_raw['Collagen_Status'] == cond1)]
    d2_df = df_raw[(df_raw['Group'] == grp2) & (df_raw['Collagen_Status'] == cond2)]
    rows = []
    for feat in FEATURE_COLS:
        d1 = d1_df[feat].dropna().values
        d2 = d2_df[feat].dropna().values
        if len(d1) < 3 or len(d2) < 3:
            rows.append({'Feature': feat, 'Cohens_d': np.nan, 'p_raw': np.nan})
            continue
        p, _ = adaptive_test_cells(d1, d2)
        pooled = np.sqrt((d1.std(ddof=1)**2 + d2.std(ddof=1)**2) / 2)
        d = (d2.mean() - d1.mean()) / pooled if pooled > 0 else np.nan
        rows.append({'Feature': feat, 'Cohens_d': d, 'p_raw': p})
    res = pd.DataFrame(rows)
    _, fdr, _, _ = multipletests(res['p_raw'].fillna(1), method='fdr_bh')
    res['p_FDR'] = fdr
    res['sig']   = fdr < 0.05
    return res.set_index('Feature')

# Pre-compute all comparisons we need
C = {
    'dis_nc_tav':  compare_cell('TAV-NA', 'NoCollagen', 'TAV-ATAA', 'NoCollagen'),
    'dis_nc_bav':  compare_cell('TAV-NA', 'NoCollagen', 'BAV-ATAA', 'NoCollagen'),
    'dis_col_tav': compare_cell('TAV-NA', 'Collagen',   'TAV-ATAA', 'Collagen'),
    'dis_col_bav': compare_cell('TAV-NA', 'Collagen',   'BAV-ATAA', 'Collagen'),
    'bav_tav_nc':  compare_cell('TAV-ATAA', 'NoCollagen', 'BAV-ATAA', 'NoCollagen'),
    'bav_tav_col': compare_cell('TAV-ATAA', 'Collagen',   'BAV-ATAA', 'Collagen'),
    'coll_na':     compare_cell('TAV-NA',   'NoCollagen', 'TAV-NA',   'Collagen'),
    'coll_tav':    compare_cell('TAV-ATAA', 'NoCollagen', 'TAV-ATAA', 'Collagen'),
    'coll_bav':    compare_cell('BAV-ATAA', 'NoCollagen', 'BAV-ATAA', 'Collagen'),
    'cross_tav':   compare_cell('TAV-NA', 'NoCollagen', 'TAV-ATAA', 'Collagen'),
    'cross_bav':   compare_cell('TAV-NA', 'NoCollagen', 'BAV-ATAA', 'Collagen'),
}

TOP_N = 10

def top_features(res_a, res_b=None):
    """Return top N features by max |d| across both comparisons."""
    abs_a = res_a['Cohens_d'].abs().fillna(0)
    if res_b is not None:
        abs_b = res_b['Cohens_d'].abs().fillna(0)
        combined = abs_a.combine(abs_b, max)
    else:
        combined = abs_a
    return combined.nlargest(TOP_N).index.tolist()

# Define 6 panels
PANELS = [
    # (title, [(res_key, label, color), ...])
    dict(
        title='Disease Effect  (−Collagen)\nPositive = higher in TAA',
        comps=[
            ('dis_nc_tav', 'TAV-ATAA', '#E53935'),
            ('dis_nc_bav', 'BAV-ATAA', '#FF9800'),
        ],
    ),
    dict(
        title='Disease Effect  (+Collagen)\nPositive = higher in TAA',
        comps=[
            ('dis_col_tav', 'TAV-ATAA', '#E53935'),
            ('dis_col_bav', 'BAV-ATAA', '#FF9800'),
        ],
    ),
    dict(
        title='TAV-ATAA vs BAV-ATAA\nPositive = higher in BAV-ATAA',
        comps=[
            ('bav_tav_nc',  '−Collagen', '#546E7A'),
            ('bav_tav_col', '+Collagen', '#00897B'),
        ],
    ),
    dict(
        title='Collagen Effect in TAV-NA\nPositive = higher with collagen',
        comps=[
            ('coll_na', 'TAV-NA', '#1565C0'),
        ],
    ),
    dict(
        title='Collagen Effect in TAA\nPositive = higher with collagen',
        comps=[
            ('coll_tav', 'TAV-ATAA', '#E53935'),
            ('coll_bav', 'BAV-ATAA', '#FF9800'),
        ],
    ),
    dict(
        title='TAV-NA −Col  vs  TAA +Col\nPositive = higher in TAA +Col',
        comps=[
            ('cross_tav', 'TAV-ATAA', '#E53935'),
            ('cross_bav', 'BAV-ATAA', '#FF9800'),
        ],
    ),
]

# ── Draw ──────────────────────────────────────────────────────────────────────
fig5, axes5 = plt.subplots(2, 3, figsize=(22, 16))
fig5.subplots_adjust(hspace=0.55, wspace=0.65,
                     left=0.09, right=0.97, top=0.88, bottom=0.06)

BAR_H   = 0.32   # height of each sub-bar
OFFSETS = [0.18, -0.18]   # vertical offsets for two bars (single bar uses 0)

for ax, panel in zip(axes5.flat, PANELS):
    comps = panel['comps']
    n_bars = len(comps)

    # Get result objects
    res_list = [C[k] for k, _, _ in comps]

    # Select top N features
    feats = top_features(res_list[0], res_list[1] if n_bars > 1 else None)

    # Sort by the first comparison's d (ascending → most negative at bottom)
    feats = sorted(feats, key=lambda f: res_list[0].loc[f, 'Cohens_d']
                   if not np.isnan(res_list[0].loc[f, 'Cohens_d']) else 0)

    y = np.arange(len(feats))
    offsets = OFFSETS[:n_bars] if n_bars > 1 else [0.0]

    for (key, label, color), offset in zip(comps, offsets):
        res = C[key]
        for i, feat in enumerate(feats):
            d   = res.loc[feat, 'Cohens_d']
            sig = res.loc[feat, 'sig']
            fdr = res.loc[feat, 'p_FDR']
            if np.isnan(d):
                continue
            alpha = 1.0 if sig else 0.38
            ax.barh(y[i] + offset, d, height=BAR_H,
                    color=color, alpha=alpha, edgecolor='none')
            s = stars(fdr)
            if s:
                x_off = 0.02 if d >= 0 else -0.02
                ha    = 'left' if d >= 0 else 'right'
                ax.text(d + x_off, y[i] + offset, s,
                        va='center', ha=ha, fontsize=10,
                        color='#111111', fontweight='bold')

    ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace('_', ' ') for f in feats], fontsize=11)
    ax.set_xlabel("Cohen's d", fontsize=11)
    ax.set_title(panel['title'], fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='x', color='#eeeeee', lw=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.99, 0.01,
            '* p<0.05  ** p<0.01  *** p<0.001\n(BH-FDR, cell-level)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8.5, color='#888888')

    # Per-panel legend
    leg_handles = []
    for (key, label, color), offset in zip(comps, offsets):
        leg_handles.append(
            mpatches.Patch(facecolor=color, alpha=0.85, label=label))
    leg_handles.append(
        mpatches.Patch(facecolor='#aaaaaa', alpha=0.38, label='FDR ≥ 0.05'))
    ax.legend(handles=leg_handles, fontsize=9, loc='lower right',
              framealpha=0.9, borderpad=0.4)

fig5.suptitle(
    'Effect Sizes (Cohen\'s d) for All Pairwise Comparisons\nTop 10 Features per Comparison  |  Cell-level  |  BH-FDR corrected',
    fontsize=15, fontweight='bold', y=0.97)

plt.savefig(OUT_FILE5, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_FILE5}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Disease effect two-panel (No Collagen | +Collagen)
# Style: full feature list, sorted by NoCol effect, background shading,
#        colored bold labels for significant features
# ══════════════════════════════════════════════════════════════════════════════

OUT_FILE6 = Path('classification_results/three_group_analysis/disease_effect_twopanel_tav.png')
OUT_FILE6b = Path('classification_results/three_group_analysis/disease_effect_twopanel_bav.png')

COLOR_POS = '#E57373'   # red  — higher in TAA
COLOR_NEG = '#64B5F6'   # blue — lower in TAA

def make_twopanel(grp_taa, out_path):
    """Two-panel disease effect figure: NoCollagen | +Collagen."""
    res_nc = compare_cell('TAV-NA', 'NoCollagen', grp_taa, 'NoCollagen')
    res_c  = compare_cell('TAV-NA', 'Collagen',   grp_taa, 'Collagen')

    # Sort all features by NoCollagen Cohen's d (most negative → most positive)
    feats_sorted = res_nc['Cohens_d'].sort_values().index.tolist()
    n = len(feats_sorted)
    y = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12), sharey=True)
    fig.subplots_adjust(wspace=0.04, left=0.24, right=0.97, top=0.88, bottom=0.07)

    def draw_panel(ax, res, title):
        # Determine x range
        all_d = res['Cohens_d'].dropna().abs().max()
        xmax  = max(all_d * 1.25, 1.0)

        # Background shading for effect size thresholds
        ax.axvspan(-xmax, -0.8, alpha=0.07, color='#888888', zorder=0)
        ax.axvspan( 0.8,  xmax, alpha=0.07, color='#888888', zorder=0)
        ax.axvspan(-0.8, -0.5, alpha=0.04, color='#888888', zorder=0)
        ax.axvspan( 0.5,  0.8, alpha=0.04, color='#888888', zorder=0)

        for i, feat in enumerate(feats_sorted):
            d   = res.loc[feat, 'Cohens_d']
            sig = res.loc[feat, 'sig']
            fdr = res.loc[feat, 'p_FDR']
            if np.isnan(d):
                continue
            color = COLOR_POS if d >= 0 else COLOR_NEG
            alpha = 1.0 if sig else 0.38
            ax.barh(i, d, color=color, alpha=alpha, height=0.65, edgecolor='none', zorder=2)

            s = stars(fdr)
            if s:
                x_off = 0.05 if d >= 0 else -0.05
                ha    = 'left' if d >= 0 else 'right'
                ax.text(d + x_off, i, s, va='center', ha=ha,
                        fontsize=9, color='#111111', fontweight='bold', zorder=3)

        ax.axvline(0, color='black', lw=1.0, zorder=3)
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='x', color='#dddddd', lw=0.6, zorder=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.99, 0.01,
                '* p<0.05  ** p<0.01  *** p<0.001\n(BH-FDR, cell-level)',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8.5, color='#888888')

        # Effect size threshold labels (on first panel only)
        if ax == ax1:
            ax.text(-xmax + 0.03, n - 0.5, 'Large effect', fontsize=8,
                    color='#999999', va='top')
            ax.text(-0.65, n - 0.5, 'Medium\neffect', fontsize=7.5,
                    color='#aaaaaa', va='top', ha='center')

    draw_panel(ax1, res_nc, f'Disease Effect  (No Collagen)\nTAV-NA  vs  {grp_taa}')
    draw_panel(ax2, res_c,  f'Disease Effect  (+Collagen)\nTAV-NA  vs  {grp_taa}')

    # Y-axis labels on ax1 only (sharey), colored by significance in NoCollagen
    ax1.set_yticks(y)
    tick_labels = [f.replace('_', ' ') for f in feats_sorted]
    ax1.set_yticklabels(tick_labels, fontsize=10)

    for tick, feat in zip(ax1.get_yticklabels(), feats_sorted):
        sig = res_nc.loc[feat, 'sig']
        d   = res_nc.loc[feat, 'Cohens_d']
        if sig:
            tick.set_color(COLOR_POS if d >= 0 else COLOR_NEG)
            tick.set_fontweight('bold')
        else:
            tick.set_color('#555555')

    # Legend
    handles = [
        mpatches.Patch(facecolor=COLOR_POS, alpha=1.0,  label='Higher in TAA (positive d)'),
        mpatches.Patch(facecolor=COLOR_NEG, alpha=1.0,  label='Lower in TAA (negative d)'),
        mpatches.Patch(facecolor='#aaaaaa', alpha=0.38, label='FDR ≥ 0.05 (faded)'),
        mpatches.Patch(facecolor='#888888', alpha=0.07, label='Large effect  |d| > 0.8'),
        mpatches.Patch(facecolor='#888888', alpha=0.04, label='Medium effect  |d| > 0.5'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 0.96), framealpha=0.9)

    fig.suptitle(
        f'SMC Morphology: Disease Effect by Collagen Condition\n'
        f'TAV-NA vs {grp_taa}  |  Cell-level  |  BH-FDR corrected',
        fontsize=14, fontweight='bold', y=1.00)

    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

make_twopanel('TAV-ATAA', OUT_FILE6)
make_twopanel('BAV-ATAA', OUT_FILE6b)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — 4-group boxplots for all 21 features
# Groups: Healthy NoCol | Healthy +Col | TAV-ATAA NoCol | TAV-ATAA +Col
# (+ separate set for BAV-ATAA)
# ══════════════════════════════════════════════════════════════════════════════

from scipy.stats import mannwhitneyu
from itertools import combinations

BOXPLOT_OUT = Path('classification_results/three_group_analysis/boxplots_4group')
BOXPLOT_OUT.mkdir(exist_ok=True)
BOXPLOT_OUT_BAV = Path('classification_results/three_group_analysis/boxplots_4group_bav')
BOXPLOT_OUT_BAV.mkdir(exist_ok=True)

# Colors: Healthy=green shades, TAA=red shades
COLORS_4 = {
    'Healthy\nNo Coll':  '#66BB6A',
    'Healthy\n+Coll':    '#A5D6A7',
    'TAA\nNo Coll':      '#EF5350',
    'TAA\n+Coll':        '#FFAB91',
}

def make_4group_boxplots(taa_group, out_dir):
    healthy_nc  = df_raw[(df_raw['Group'] == 'TAV-NA')   & (df_raw['Collagen_Status'] == 'NoCollagen')]
    healthy_col = df_raw[(df_raw['Group'] == 'TAV-NA')   & (df_raw['Collagen_Status'] == 'Collagen')]
    taa_nc      = df_raw[(df_raw['Group'] == taa_group)  & (df_raw['Collagen_Status'] == 'NoCollagen')]
    taa_col     = df_raw[(df_raw['Group'] == taa_group)  & (df_raw['Collagen_Status'] == 'Collagen')]

    group_labels = ['Healthy\nNo Coll', 'Healthy\n+Coll', 'TAA\nNo Coll', 'TAA\n+Coll']
    group_data_map = {
        'Healthy\nNo Coll': healthy_nc,
        'Healthy\n+Coll':   healthy_col,
        'TAA\nNo Coll':     taa_nc,
        'TAA\n+Coll':       taa_col,
    }

    # Pairs to test (key disease + collagen comparisons)
    test_pairs = [
        (0, 2),  # Healthy NoCol vs TAA NoCol
        (0, 3),  # Healthy NoCol vs TAA +Col
        (1, 3),  # Healthy +Col vs TAA +Col
    ]

    for feat in FEATURE_COLS:
        fig, ax = plt.subplots(figsize=(8, 6))

        data_groups = [group_data_map[lbl][feat].dropna().values for lbl in group_labels]
        colors      = [COLORS_4[lbl] for lbl in group_labels]
        positions   = [1, 2, 3, 4]

        # Boxplots
        bp = ax.boxplot(data_groups, positions=positions, patch_artist=True,
                        widths=0.55, showfliers=True,
                        flierprops=dict(marker='o', markersize=3,
                                        markerfacecolor='#888888', alpha=0.5),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        boxprops=dict(linewidth=1.2))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        # Mean ± SD diamond markers
        for i, (vals, pos) in enumerate(zip(data_groups, positions)):
            if len(vals) == 0:
                continue
            mean_v = np.mean(vals)
            ax.plot(pos, mean_v, marker='D', color='black',
                    markersize=6, zorder=5, label='Mean ± SD' if i == 0 else '')
            sd_v = np.std(vals, ddof=1)
            ax.errorbar(pos, mean_v, yerr=sd_v, fmt='none',
                        color='black', capsize=4, linewidth=1.2, zorder=4)

        # Significance brackets
        y_max   = max(np.percentile(g, 97) for g in data_groups if len(g) > 0)
        y_range = y_max - min(np.percentile(g, 3) for g in data_groups if len(g) > 0)
        step    = y_range * 0.10
        y_top   = y_max + step * 0.5

        drawn = 0
        for (i, j) in test_pairs:
            d1, d2 = data_groups[i], data_groups[j]
            if len(d1) < 3 or len(d2) < 3:
                continue
            _, p = mannwhitneyu(d1, d2, alternative='two-sided')
            s = stars(p)
            if not s:
                continue
            y_br = y_top + step * drawn
            x1, x2 = positions[i], positions[j]
            ax.plot([x1, x1, x2, x2], [y_br, y_br + step*0.3,
                                         y_br + step*0.3, y_br],
                    color='black', lw=1.2)
            ax.text((x1 + x2) / 2, y_br + step * 0.32, s,
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
            drawn += 1

        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels, fontsize=11)
        ax.set_ylabel(feat.replace('_', ' '), fontsize=11)
        ax.set_title(feat.replace('_', ' '), fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#eeeeee', lw=0.7)

        ax.text(0.01, 0.01,
                f'TAV-NA (NC={len(healthy_nc)}, C={len(healthy_col)})  '
                f'{taa_group} (NC={len(taa_nc)}, C={len(taa_col)})\n'
                'Brackets: Mann-Whitney (uncorrected)',
                transform=ax.transAxes, fontsize=7.5, color='#888888', va='bottom')

        plt.tight_layout()
        plt.savefig(out_dir / f'{feat}.png', dpi=180, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(FEATURE_COLS)} boxplots → {out_dir}/")

make_4group_boxplots('TAV-ATAA', BOXPLOT_OUT)
make_4group_boxplots('BAV-ATAA', BOXPLOT_OUT_BAV)
