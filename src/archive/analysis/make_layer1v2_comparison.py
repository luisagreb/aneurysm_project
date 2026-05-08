"""
Layer 1 vs Layer 2 — 6-feature comparison plot (slide version).

Reads saved CSVs; no re-analysis needed.

Usage:
  cd /home/luisa/aneurysm_project
  python src/analysis/make_layer1v2_comparison.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
L1 = pd.read_csv('classification_results/smc_analysis/5layer/layer1/layer1_raw_disease.csv')
L2 = pd.read_csv('classification_results/smc_analysis/5layer/layer2/layer2_age_corrected.csv')

# 6 curated features in display order (bottom → top in horizontal bar chart)
FEATURES = [
    'Nucleus_Sphericity',   # stays gray → gray
    'Actin_Minor_Axis',     # appears gray → **
    'Mito_Volume',          # disappears *** → gray
    'Mito_Fragment_Count',  # disappears *** → gray
    'Actin_Solidity',       # survives *** → *
    'Mito_Branch_Count',    # survives *** → **
]

LABELS = {
    'Nucleus_Sphericity':  'Nucleus Sphericity',
    'Actin_Minor_Axis':    'Actin Minor Axis',
    'Mito_Volume':         'Mito Volume',
    'Mito_Fragment_Count': 'Mito Fragment Count',
    'Actin_Solidity':      'Actin Solidity',
    'Mito_Branch_Count':   'Mito Branch Count',
}

STORY = {
    'Nucleus_Sphericity':  'stays gray → gray\n(no disease effect)',
    'Actin_Minor_Axis':    'appears gray → **\n(age was suppressing this)',
    'Mito_Volume':         'disappears *** → gray\n(age-driven)',
    'Mito_Fragment_Count': 'disappears *** → gray\n(age-driven)',
    'Actin_Solidity':      'survives *** → *\n(robust disease signal)',
    'Mito_Branch_Count':   'survives *** → **\n(robust disease signal)',
}

def stars(p):
    if pd.isna(p): return ''
    if p < 0.001:  return '***'
    if p < 0.01:   return '**'
    if p < 0.05:   return '*'
    return ''

l1 = L1.set_index('Feature')
l2 = L2.set_index('Feature')

# ── Figure ────────────────────────────────────────────────────────────────────
N   = len(FEATURES)
Y   = np.arange(N)
FIG_W, FIG_H = 13, 7

fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)
fig.subplots_adjust(wspace=0.08)

COLOR_SIG = '#E64A19'
COLOR_NS  = '#AAAAAA'

for ax, df, title in zip(
    axes,
    [l1, l2],
    ['Layer 1: Raw', 'Layer 2: Age-corrected residuals']
):
    for i, feat in enumerate(FEATURES):
        if feat not in df.index:
            continue
        row   = df.loc[feat]
        d     = row['Cohens_d']
        sig   = row['Significant']
        color = COLOR_SIG if sig else COLOR_NS

        ax.barh(i, d, height=0.6,
                color=color, edgecolor='white', linewidth=0.4, zorder=2)

        s = stars(row['p_FDR'])
        if s:
            offset = 0.02 if d >= 0 else -0.02
            ha     = 'left' if d >= 0 else 'right'
            ax.text(d + offset, i, s,
                    ha=ha, va='center', clip_on=True,
                    fontsize=18, fontweight='bold', color=color)

    ax.axvline(0, color='#333333', lw=1.0, zorder=3)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel("Cohen's d  (TAA − Non-Aneurysmal)", fontsize=14)
    ax.grid(axis='x', color='#eeeeee', lw=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=13)
    # expand x limits 15% so stars fit inside the axes
    xlo, xhi = ax.get_xlim()
    pad = (xhi - xlo) * 0.15
    ax.set_xlim(xlo - pad if xlo < 0 else xlo, xhi + pad)

# y-axis labels on left panel only
axes[0].set_yticks(Y)
axes[0].set_yticklabels([LABELS[f] for f in FEATURES], fontsize=15)
axes[0].tick_params(axis='y', length=0)

# Legend
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color=COLOR_SIG, label='FDR < 0.05'),
    mpatches.Patch(color=COLOR_NS,  label='FDR ≥ 0.05'),
]
axes[0].legend(handles=legend_handles, fontsize=13,
               loc='lower left', framealpha=0.9, edgecolor='#cccccc')

fig.suptitle('Layer 1 vs Layer 2: Impact of Age Correction',
             fontsize=17, fontweight='bold', y=1.01)

plt.tight_layout()

out = Path('classification_results/smc_analysis/5layer/layer2/layer2_comparison_slide.png')
plt.savefig(out, dpi=180, bbox_inches='tight')
plt.close()
print(f'Saved {out}')
