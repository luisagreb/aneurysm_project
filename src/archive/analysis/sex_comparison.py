"""
Sex effect comparison: Overall vs Healthy-only vs TAA-only.

Produces:
  - sex_comparison_table.csv       : Cohen's d + significance for all 3 contexts
  - sex_comparison_heatmap.png     : heatmap of Cohen's d with * markers
  - sex_comparison_dotplot.png     : dot plot with CI-style layout

Output dir: classification_results/smc_analysis/sex_comparison/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_FILE = 'outputs/Advanced_Features_Raw_200.csv'

NON_ANEURYSMAL = [
    '01asc-180', '01asc-222', '01asc-230',
    '01c-96', '01c-97', '01c-113', '01c-117', '01c-202'
]

FEATURE_COLS = [
    'Actin_Volume', 'Actin_Skeleton_Length', 'Actin_Solidity',
    'Actin_Fractional_Anisotropy', 'Actin_Major_Axis', 'Actin_Minor_Axis',
    'Mito_Volume', 'Mito_Sphericity', 'Mito_Fragment_Count',
    'Mito_Mean_Fragment_Volume', 'Mito_Branch_Count',
    'Mito_Total_Network_Length', 'Mito_Mean_Tortuosity',
    'Mito_Junction_Count', 'Mito_Cyclomatic_Number',
    'Nucleus_Volume', 'Nucleus_Sphericity', 'Nucleus_Circularity',
    'Nucleus_Elongation', 'Nucleus_Flatness', 'Nucleus_Solidity'
]

OUT_DIR = Path('classification_results/smc_analysis/sex_comparison')

# ── Helpers ───────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 +
                      (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan


def run_sex_test(df, feat):
    male   = df.loc[df['Sex'] == 'M', feat].dropna().values
    female = df.loc[df['Sex'] == 'F', feat].dropna().values
    if len(male) < 2 or len(female) < 2:
        return np.nan, np.nan
    _, pm = stats.shapiro(male)
    _, pf = stats.shapiro(female)
    if pm > 0.05 and pf > 0.05:
        _, p = stats.ttest_ind(male, female)
    else:
        _, p = stats.mannwhitneyu(male, female, alternative='two-sided')
    d = cohens_d(male, female)
    return d, p


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction (handles NaN)."""
    arr = np.array(pvals, dtype=float)
    valid = ~np.isnan(arr)
    q = np.full(len(arr), np.nan)
    pv = arr[valid]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    rank  = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    q_adj = np.minimum(1.0, pv * n / rank)
    # monotone
    q_adj = np.minimum.accumulate(q_adj[::-1])[::-1]
    q[valid] = q_adj
    return q


def compute_sex_effect(df):
    rows = []
    raw_p = []
    for feat in FEATURE_COLS:
        d, p = run_sex_test(df, feat)
        rows.append({'Feature': feat, 'Cohens_d': d, 'p_value': p})
        raw_p.append(p)
    result = pd.DataFrame(rows)
    result['p_FDR'] = fdr_bh(result['p_value'].values)
    result['Significant'] = result['p_FDR'] < 0.05
    return result.set_index('Feature')


def stars(p):
    if np.isnan(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_FILE)
df['SMC_Group'] = df['Cell_ID'].apply(
    lambda x: 'Healthy' if any(x.startswith(s) for s in NON_ANEURYSMAL) else 'TAA'
)

print("Sex distribution:")
print(df.groupby(['SMC_Group', 'Sex']).size().unstack(fill_value=0))
print()

# ── Compute sex effect in 3 contexts ─────────────────────────────────────────
df_overall = df.copy()
df_healthy = df[df['SMC_Group'] == 'Healthy']
df_taa     = df[df['SMC_Group'] == 'TAA']

ov  = compute_sex_effect(df_overall)
hlt = compute_sex_effect(df_healthy)
taa = compute_sex_effect(df_taa)

print("Overall significant:", ov.index[ov['Significant']].tolist())
print("Healthy significant:", hlt.index[hlt['Significant']].tolist())
print("TAA significant:    ", taa.index[taa['Significant']].tolist())

# ── Build combined table ──────────────────────────────────────────────────────
combined = pd.DataFrame({
    'Overall_d':   ov['Cohens_d'],
    'Overall_p':   ov['p_FDR'],
    'Overall_sig': ov['Significant'],
    'Healthy_d':   hlt['Cohens_d'],
    'Healthy_p':   hlt['p_FDR'],
    'Healthy_sig': hlt['Significant'],
    'TAA_d':       taa['Cohens_d'],
    'TAA_p':       taa['p_FDR'],
    'TAA_sig':     taa['Significant'],
})

OUT_DIR.mkdir(parents=True, exist_ok=True)
combined.to_csv(OUT_DIR / 'sex_comparison_table.csv')
print(f"\nSaved table → {OUT_DIR}/sex_comparison_table.csv")

# ── Heatmap ───────────────────────────────────────────────────────────────────
contexts = ['Overall', 'Healthy', 'TAA']
d_matrix = np.column_stack([
    combined['Overall_d'].values,
    combined['Healthy_d'].values,
    combined['TAA_d'].values,
])
p_matrix = np.column_stack([
    combined['Overall_p'].values,
    combined['Healthy_p'].values,
    combined['TAA_p'].values,
])

feat_labels = FEATURE_COLS
# sort by |Healthy_d| descending
sort_idx = np.argsort(np.abs(combined['Healthy_d'].values))[::-1]
feat_labels_sorted = [FEATURE_COLS[i] for i in sort_idx]
d_sorted = d_matrix[sort_idx]
p_sorted = p_matrix[sort_idx]

fig, ax = plt.subplots(figsize=(6, 9))
vmax = max(0.8, np.nanmax(np.abs(d_sorted)))
im = ax.imshow(d_sorted, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
               aspect='auto', interpolation='nearest')

# star annotations
for r in range(len(feat_labels_sorted)):
    for c in range(3):
        s = stars(p_sorted[r, c])
        if s:
            ax.text(c, r, s, ha='center', va='center',
                    fontsize=11, color='black', fontweight='bold')

ax.set_xticks(range(3))
ax.set_xticklabels(contexts, fontsize=12)
ax.set_yticks(range(len(feat_labels_sorted)))
ax.set_yticklabels(feat_labels_sorted, fontsize=9)
ax.set_title("Sex effect (Cohen's d, Male − Female)\n* p_FDR<0.05  ** <0.01  *** <0.001",
             fontsize=11, pad=10)

cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("Cohen's d", fontsize=10)
cbar.ax.axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
out_hm = OUT_DIR / 'sex_comparison_heatmap.png'
plt.savefig(out_hm, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved heatmap → {out_hm}")

# ── Dot / lollipop plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 9))

y = np.arange(len(feat_labels_sorted))
colors_ctx = {'Overall': '#555555', 'Healthy': '#2196F3', 'TAA': '#E53935'}
offsets    = {'Overall': -0.22,     'Healthy':  0.0,      'TAA':  0.22}
markers    = {'Overall': 's',       'Healthy':  'o',       'TAA':  'D'}

for ctx, col_d in zip(contexts, ['Overall_d', 'Healthy_d', 'TAA_d']):
    col_p = ctx + '_p'
    ypos = y + offsets[ctx]
    d_vals = combined[col_d].values[sort_idx]
    p_vals = combined[col_p].values[sort_idx]
    color = colors_ctx[ctx]
    for i, (di, pi) in enumerate(zip(d_vals, p_vals)):
        if np.isnan(di):
            continue
        alpha = 1.0 if (not np.isnan(pi) and pi < 0.05) else 0.35
        ax.scatter(di, ypos[i], color=color, s=55,
                   marker=markers[ctx], alpha=alpha, zorder=3,
                   edgecolors='white', linewidths=0.4)
        ax.plot([0, di], [ypos[i], ypos[i]],
                color=color, lw=1.0, alpha=alpha * 0.6, zorder=2)
        s = stars(pi)
        if s:
            ax.text(di + (0.025 if di >= 0 else -0.025), ypos[i], s,
                    ha='left' if di >= 0 else 'right',
                    va='center', fontsize=8, color=color, fontweight='bold')

ax.axvline(0, color='#aaaaaa', lw=0.8, ls='--')
ax.set_yticks(y)
ax.set_yticklabels(feat_labels_sorted, fontsize=9)
ax.set_xlabel("Cohen's d  (Male − Female)", fontsize=11)
ax.set_title("Sex effect: Overall vs Healthy vs TAA\n(filled = FDR<0.05, faded = ns)",
             fontsize=11, pad=10)

legend_handles = [
    mpatches.Patch(color=colors_ctx[c], label=c) for c in contexts
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=10,
          framealpha=0.9, edgecolor='#cccccc')

ax.invert_yaxis()
ax.grid(axis='x', color='#eeeeee', lw=0.7)
plt.tight_layout()
out_dp = OUT_DIR / 'sex_comparison_dotplot.png'
plt.savefig(out_dp, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved dot plot → {out_dp}")
