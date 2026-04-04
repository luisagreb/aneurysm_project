"""
Layer 4 — 3-condition bar plot (slide version, 8 features).

Two horizontal bars per feature:
  TAA −Col  (warm orange-pink, 70% opacity) — disease baseline
  TAA +Col  (dark burgundy, 100% opacity)   — treated condition

Healthy −Col = vertical reference line at zero (labeled).

Significance brackets right of bars:
  solid grey   = Healthy −Col vs TAA −Col  (disease effect)
  dashed red   = TAA −Col vs TAA +Col      (rescue effect)
  Both use BH-FDR across all 42 tests (21 features × 2 comparisons).

Usage:
  cd /home/luisa/aneurysm_project
  python src/analysis/make_layer4_3bar.py
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, shapiro
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Group definitions (same as complete_stat_analysis.py) ─────────────────────
NON_ANEURYSMAL = ['01asc-180','01asc-222','01asc-230',
                  '01c-96','01c-97','01c-113','01c-117','01c-202']
ANEURYSMAL     = ['03asc-24','03asc-43','03rt-45','03asc-46','03asc-47',
                  '03asc-51','03asc-54','03asc-55','03asc-56','03asc-57']

FEATURE_COLS = [
    'Actin_Volume','Actin_Skeleton_Length','Actin_Solidity',
    'Actin_Fractional_Anisotropy','Actin_Major_Axis','Actin_Minor_Axis',
    'Mito_Volume','Mito_Sphericity','Mito_Fragment_Count',
    'Mito_Mean_Fragment_Volume','Mito_Branch_Count',
    'Mito_Total_Network_Length','Mito_Mean_Tortuosity',
    'Mito_Junction_Count','Mito_Cyclomatic_Number',
    'Nucleus_Volume','Nucleus_Sphericity','Nucleus_Circularity',
    'Nucleus_Elongation','Nucleus_Flatness','Nucleus_Solidity',
]

def normalize(sid):
    return re.sub(r'-0*(\d+)$', r'-\1', sid.lower()) if sid else None

def extract_id(cell_name):
    m = re.search(r'(0[123][A-Za-z]+-\d+)', str(cell_name))
    if not m:
        m = re.search(r'(0[123][A-Za-z]+)(\d+)', str(cell_name))
        if not m: return None
        return (m.group(1) + '-' + m.group(2)).lower()
    return m.group(1).lower()

def adaptive_test(a, b):
    if len(a) < 3 or len(b) < 3:
        return np.nan
    _, pa = shapiro(a) if len(a) <= 5000 else (None, 0)
    _, pb = shapiro(b) if len(b) <= 5000 else (None, 0)
    if pa > 0.05 and pb > 0.05:
        _, p = ttest_ind(a, b, equal_var=False)
    else:
        _, p = mannwhitneyu(a, b, alternative='two-sided')
    return p

def shorten(feat):
    return (feat
            .replace('Actin_', 'Act ')
            .replace('Mito_',  'Mit ')
            .replace('Nucleus_', 'Nuc ')
            .replace('_', ' '))

# ── Load raw data ──────────────────────────────────────────────────────────────
na_norm = [normalize(x) for x in NON_ANEURYSMAL]
an_norm = [normalize(x) for x in ANEURYSMAL]

raw = pd.read_csv('outputs/Advanced_Features_Raw_200.csv')
raw['SpecimenID'] = raw['CellName'].apply(extract_id).apply(
    lambda x: normalize(x) if x else None)
raw = raw[raw['SpecimenID'].isin(na_norm + an_norm)].copy()
raw['SMC_Group'] = raw['SpecimenID'].apply(
    lambda x: 'Non-Aneurysmal' if x in na_norm else 'Aneurysmal')

# ── Slide-version: 8 curated features ────────────────────────────────────────
SLIDE_FEATURES = [
    'Mito_Branch_Count',
    'Mito_Total_Network_Length',
    'Mito_Junction_Count',
    'Mito_Cyclomatic_Number',
    'Mito_Fragment_Count',
    'Actin_Solidity',
    'Mito_Sphericity',
    'Actin_Fractional_Anisotropy',
]

feat_cols = [c for c in FEATURE_COLS if c in raw.columns]

# Three conditions
healthy_nc  = raw[(raw['SMC_Group'] == 'Non-Aneurysmal') & (raw['Collagen_Status'] == 'NoCollagen')]
taa_nc      = raw[(raw['SMC_Group'] == 'Aneurysmal')     & (raw['Collagen_Status'] == 'NoCollagen')]
taa_col     = raw[(raw['SMC_Group'] == 'Aneurysmal')     & (raw['Collagen_Status'] == 'Collagen')]

# ── Compute p-values for both comparisons ─────────────────────────────────────
p_disease, p_rescue = [], []

for feat in feat_cols:
    a = healthy_nc[feat].dropna().values
    b = taa_nc[feat].dropna().values
    c = taa_col[feat].dropna().values
    p_disease.append(adaptive_test(a, b))
    p_rescue.append(adaptive_test(b, c))

# FDR over all comparisons jointly (2 × 21 = 42 tests)
all_p   = p_disease + p_rescue
_, all_fdr, _, _ = multipletests(all_p, method='fdr_bh')
fdr_disease = all_fdr[:len(feat_cols)]
fdr_rescue  = all_fdr[len(feat_cols):]

def stars(p):
    if np.isnan(p): return ''
    if p < 0.001:   return '***'
    if p < 0.01:    return '**'
    if p < 0.05:    return '*'
    return ''

# ── Build plot table ───────────────────────────────────────────────────────────
rows = []
for i, feat in enumerate(feat_cols):
    a  = healthy_nc[feat].dropna().values
    b  = taa_nc[feat].dropna().values
    c  = taa_col[feat].dropna().values
    sd = a.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        continue
    rows.append({
        'Feature':       feat,
        'Label':         shorten(feat),
        'z_T':           (b.mean() - a.mean()) / sd,
        'z_C':           (c.mean() - a.mean()) / sd,
        'fdr_disease':   fdr_disease[i],
        'fdr_rescue':    fdr_rescue[i],
        'p_disease':     p_disease[i],
        'p_rescue':      p_rescue[i],
        'rescue_toward': abs(c.mean() - a.mean()) < abs(b.mean() - a.mean()),
    })

all_df  = pd.DataFrame(rows)

# ── Slide subset: keep only the 8 curated features, sorted by |z_T| descending
plot_df = all_df[all_df['Feature'].isin(SLIDE_FEATURES)].copy()
plot_df = plot_df.reindex(
    plot_df['z_T'].abs().sort_values(ascending=True).index   # ascending = largest at top
).reset_index(drop=True)

# ── Figure ────────────────────────────────────────────────────────────────────
N       = len(plot_df)
BAR_H   = 0.30          # height of each bar
GAP     = 0.06          # gap between TAA−Col and TAA+Col bars
PITCH   = 1.0           # centre-to-centre distance between features
Y_MID   = np.arange(N) * PITCH   # centre y per feature

Y_NC  = Y_MID + GAP / 2 + BAR_H / 2    # TAA −Col row
Y_COL = Y_MID - GAP / 2 - BAR_H / 2    # TAA +Col row

C_NC  = '#F4845F'   # warm orange-pink
C_COL = '#6B1019'   # dark burgundy

fig, ax = plt.subplots(figsize=(9, max(5, N * 0.85)))

# Bars
for i, row in plot_df.iterrows():
    ax.barh(Y_NC[i],  row['z_T'], height=BAR_H, color=C_NC,  alpha=0.70,
            edgecolor='white', linewidth=0.4, zorder=2)
    ax.barh(Y_COL[i], row['z_C'], height=BAR_H, color=C_COL, alpha=1.00,
            edgecolor='white', linewidth=0.4, zorder=2)

# Reference line + label (placed just below the top of the axis)
ax.axvline(0, color='#222222', lw=1.2, zorder=3)
ax.text(0.04, Y_MID[-1] + BAR_H * 1.2, 'Healthy −Col\nreference',
        ha='left', va='bottom', fontsize=8, color='#005F52',
        fontstyle='italic')

# ── Significance brackets ──────────────────────────────────────────────────────
X_MAX  = plot_df[['z_T','z_C']].abs().max().max()
X_BASE = X_MAX * 1.08
STEP   = X_MAX * 0.22

BRACKET_D = {'color': '#555555', 'lw': 1.4, 'ls': '-',  'zorder': 4}
BRACKET_R = {'color': '#6B1019', 'lw': 1.4, 'ls': '--', 'zorder': 4}

def draw_bracket(ax, x0, y_lo, y_hi, s, style, text_color):
    tick = 0.04
    ax.plot([x0, x0], [y_lo, y_hi], **style)
    ax.plot([x0 - tick, x0], [y_hi, y_hi], **style)
    ax.plot([x0 - tick, x0], [y_lo, y_lo], **style)
    ax.text(x0 + 0.03, (y_hi + y_lo) / 2, s,
            ha='left', va='center', fontsize=9,
            fontweight='bold', color=text_color)

for i, row in plot_df.iterrows():
    col = 0  # bracket column offset

    # Disease: Healthy (y=0 by definition) vs TAA −Col
    s_d = stars(row['fdr_disease']) or (stars(row['p_disease']) + '†' if stars(row['p_disease']) else '')
    if s_d:
        x0   = X_BASE + col * STEP
        # span from bottom of TAA−Col bar to zero line (healthy ref)
        y_lo = Y_NC[i] - BAR_H / 2
        y_hi = Y_NC[i] + BAR_H / 2
        # show bracket centred on this bar, connecting to the zero line
        ax.annotate('', xy=(x0, Y_NC[i]), xytext=(x0 + 0.01, Y_NC[i]),
                    xycoords='data')  # dummy — real drawing below
        draw_bracket(ax, x0, y_lo, y_hi, s_d, BRACKET_D, '#333333')
        col += 1

    # Rescue: TAA −Col vs TAA +Col
    s_r = stars(row['fdr_rescue']) or (stars(row['p_rescue']) + '†' if stars(row['p_rescue']) else '')
    if s_r:
        x0   = X_BASE + col * STEP
        y_lo = Y_COL[i] - BAR_H / 2
        y_hi = Y_NC[i]  + BAR_H / 2
        draw_bracket(ax, x0, y_lo, y_hi, s_r, BRACKET_R, '#6B1019')

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_yticks(Y_MID)
ax.set_yticklabels(plot_df['Label'], fontsize=10)

for i in range(N - 1):
    ax.axhline((Y_MID[i] + Y_MID[i+1]) / 2, color='#e4e4e4', lw=0.7, zorder=0)

ax.grid(axis='x', color='#eeeeee', lw=0.8, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Directional annotation below x-axis
x_lo, x_hi = ax.get_xlim()
ax.set_xlabel('')
fig.text(0.5, -0.03,
         '← TAA lower than healthy          z-score (Δ SD from Healthy −Col mean)          TAA higher than healthy →',
         ha='center', va='top', fontsize=9, color='#444444')

ax.set_title(
    'Layer 4 — Collagen effect within TAA\n'
    'Reference: Healthy −Col  |  * FDR < 0.05   *† p < 0.05 raw',
    fontsize=11, fontweight='bold', pad=12
)

# Legend
legend_handles = [
    mpatches.Patch(color=C_NC,  alpha=0.70, label='TAA −Col  (disease baseline)'),
    mpatches.Patch(color=C_COL, alpha=1.00, label='TAA +Col  (collagen-treated)'),
    plt.Line2D([0],[0], color='#555555', lw=1.4, ls='-',  label='Healthy vs TAA −Col'),
    plt.Line2D([0],[0], color='#6B1019', lw=1.4, ls='--', label='TAA −Col vs TAA +Col'),
]
ax.legend(handles=legend_handles, loc='lower right',
          fontsize=8.5, framealpha=0.95, edgecolor='#cccccc')

plt.tight_layout()

out = Path('classification_results/smc_analysis/5layer/layer4/layer4_3condition_bars.png')
plt.savefig(out, dpi=180, bbox_inches='tight')
plt.close()
print(f'Saved {out}')

# Print summary
print('\nDisease effect (Healthy −Col vs TAA −Col) FDR<0.05:')
sig = plot_df[plot_df['fdr_disease'] < 0.05][['Label','z_T','fdr_disease']]
print(sig.to_string(index=False))
print('\nRescue effect (TAA −Col vs TAA +Col) FDR<0.05:')
sig2 = plot_df[plot_df['fdr_rescue'] < 0.05][['Label','z_T','z_C','fdr_rescue','rescue_toward']]
print(sig2.to_string(index=False) if len(sig2) else '  (none)')
print('\nNominal rescue (p<0.05 raw):')
sig3 = plot_df[plot_df['p_rescue'] < 0.05][['Label','z_T','z_C','p_rescue','rescue_toward']]
print(sig3.to_string(index=False) if len(sig3) else '  (none)')
