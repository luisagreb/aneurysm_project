"""
3-Group Statistical Analysis: TAV-NA vs TAV-ATAA vs BAV-ATAA
=============================================================

Analysis structure:
  Layer 1 — Omnibus Kruskal-Wallis across 3 groups
  Layer 2 — Pairwise Mann-Whitney (TAV-NA vs TAV-ATAA, TAV-NA vs BAV-ATAA,
             TAV-ATAA vs BAV-ATAA) with BH-FDR correction
  Layer 3 — Age-corrected pairwise comparisons
  Plots   — Forest plot, effect heatmap, per-feature boxplots for sig features

Input:  outputs/Advanced_Features_3groups.csv
Output: classification_results/three_group_analysis/

Usage:
  cd /home/luisa/aneurysm_project
  python src/analysis/three_group_analysis.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kruskal, mannwhitneyu, ttest_ind, shapiro, spearmanr
from sklearn.linear_model import LinearRegression
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_FILE = 'outputs/Advanced_Features_3groups.csv'
OUT_DIR       = Path('classification_results/three_group_analysis')

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

GROUPS   = ['TAV-NA', 'TAV-ATAA', 'BAV-ATAA']
PAIRS    = [('TAV-NA', 'TAV-ATAA'), ('TAV-NA', 'BAV-ATAA'), ('TAV-ATAA', 'BAV-ATAA')]
PAIR_LABELS = ['TAV-NA vs TAV-ATAA', 'TAV-NA vs BAV-ATAA', 'TAV-ATAA vs BAV-ATAA']

GROUP_COLORS = {
    'TAV-NA':    '#2196F3',   # blue
    'TAV-ATAA':  '#E53935',   # red
    'BAV-ATAA':  '#FF9800',   # orange
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na-1)*np.std(a, ddof=1)**2 +
                      (nb-1)*np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(b) - np.mean(a)) / pooled if pooled > 0 else np.nan


def adaptive_test(a, b):
    """Shapiro-Wilk → T-test or Mann-Whitney."""
    if len(a) < 3 or len(b) < 3:
        return np.nan, 'insufficient'
    _, pm = shapiro(a)
    _, pf = shapiro(b)
    if pm > 0.05 and pf > 0.05:
        _, p = ttest_ind(a, b)
        return p, 'T-test'
    _, p = mannwhitneyu(a, b, alternative='two-sided')
    return p, 'Mann-Whitney'


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR on a list of p-values (no NaN)."""
    arr = np.array(pvals, dtype=float)
    n = len(arr)
    order = np.argsort(arr)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    q = np.minimum(1.0, arr * n / rank)
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i+1]])
    return q


def stars(p):
    if np.isnan(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def residualize_age(df, feat):
    """Return age-residualized feature values."""
    sub = df[['Age', feat]].dropna()
    if len(sub) < 5:
        return df[feat].copy()
    lr = LinearRegression().fit(sub[['Age']].values, sub[feat].values)
    out = df[feat].copy()
    out.loc[sub.index] = sub[feat].values - lr.predict(sub[['Age']].values)
    return out


# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_FILE)

# Use no-collagen cells for the disease comparison (as in previous analysis)
df_nc = df[df['Collagen_Status'] == 'NoCollagen'].copy()

print("=" * 60)
print("3-Group Analysis: TAV-NA / TAV-ATAA / BAV-ATAA")
print("=" * 60)
print("\nCell counts (no-collagen):")
print(df_nc.groupby('Group').size())
print()
print("Specimen counts (no-collagen):")
print(df_nc.groupby('Group')['Specimen'].nunique())
print()
print("Age by group:")
print(df_nc.groupby('Group')['Age'].describe()[['count', 'mean', 'std', 'min', 'max']].round(1))
print()

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Layer 1 — Kruskal-Wallis omnibus ─────────────────────────────────────────
print("Layer 1 — Kruskal-Wallis omnibus test")
kw_rows = []
for feat in FEATURE_COLS:
    groups = [df_nc[df_nc['Group'] == g][feat].dropna().values for g in GROUPS]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) < 2:
        kw_rows.append({'Feature': feat, 'KW_stat': np.nan, 'KW_p': np.nan})
        continue
    stat, p = kruskal(*groups)
    kw_rows.append({'Feature': feat, 'KW_stat': stat, 'KW_p': p})

kw_df = pd.DataFrame(kw_rows)
kw_df['KW_FDR'] = fdr_bh(kw_df['KW_p'].values)
kw_df['KW_sig'] = kw_df['KW_FDR'] < 0.05
kw_df.to_csv(OUT_DIR / 'layer1_kruskal_wallis.csv', index=False)

sig_kw = kw_df[kw_df['KW_sig']]['Feature'].tolist()
print(f"  Omnibus significant: {len(sig_kw)}/21")
for f in sig_kw:
    row = kw_df[kw_df['Feature'] == f].iloc[0]
    print(f"    {f}: FDR={row['KW_FDR']:.4f}")

# ── Layer 2 — Pairwise Mann-Whitney ──────────────────────────────────────────
print("\nLayer 2 — Pairwise comparisons (raw)")
pw_rows = []
for feat in FEATURE_COLS:
    for (g1, g2), label in zip(PAIRS, PAIR_LABELS):
        a = df_nc[df_nc['Group'] == g1][feat].dropna().values
        b = df_nc[df_nc['Group'] == g2][feat].dropna().values
        p, test = adaptive_test(a, b)
        d = cohens_d(a, b)  # positive = g2 > g1
        pw_rows.append({
            'Feature': feat, 'Comparison': label,
            'G1': g1, 'G2': g2,
            'Mean_G1': np.mean(a) if len(a) else np.nan,
            'Mean_G2': np.mean(b) if len(b) else np.nan,
            'Cohens_d': d, 'p_value': p, 'Test': test,
            'N_G1': len(a), 'N_G2': len(b),
        })

pw_df = pd.DataFrame(pw_rows)
# FDR across all comparisons simultaneously
valid_p = pw_df['p_value'].fillna(1.0).values
pw_df['p_FDR'] = fdr_bh(valid_p)
pw_df['Significant'] = pw_df['p_FDR'] < 0.05
pw_df.to_csv(OUT_DIR / 'layer2_pairwise_raw.csv', index=False)

print(f"  Significant pairs (FDR<0.05):")
for _, row in pw_df[pw_df['Significant']].iterrows():
    print(f"    {row['Comparison']:30s}  {row['Feature']:35s}  d={row['Cohens_d']:+.3f}  FDR={row['p_FDR']:.4f}")

# ── Layer 3 — Age-corrected pairwise ─────────────────────────────────────────
print("\nLayer 3 — Pairwise comparisons (age-corrected)")
df_nc2 = df_nc.copy()
for feat in FEATURE_COLS:
    df_nc2[feat + '_r'] = residualize_age(df_nc2, feat)

pw_age_rows = []
for feat in FEATURE_COLS:
    for (g1, g2), label in zip(PAIRS, PAIR_LABELS):
        a = df_nc2[df_nc2['Group'] == g1][feat + '_r'].dropna().values
        b = df_nc2[df_nc2['Group'] == g2][feat + '_r'].dropna().values
        p, test = adaptive_test(a, b)
        d = cohens_d(a, b)
        pw_age_rows.append({
            'Feature': feat, 'Comparison': label,
            'G1': g1, 'G2': g2,
            'Cohens_d': d, 'p_value': p, 'Test': test,
        })

pw_age_df = pd.DataFrame(pw_age_rows)
valid_p_age = pw_age_df['p_value'].fillna(1.0).values
pw_age_df['p_FDR'] = fdr_bh(valid_p_age)
pw_age_df['Significant'] = pw_age_df['p_FDR'] < 0.05
pw_age_df.to_csv(OUT_DIR / 'layer3_pairwise_age_corrected.csv', index=False)

print(f"  Significant pairs (FDR<0.05, age-corrected):")
sig_age = pw_age_df[pw_age_df['Significant']]
if len(sig_age):
    for _, row in sig_age.iterrows():
        print(f"    {row['Comparison']:30s}  {row['Feature']:35s}  d={row['Cohens_d']:+.3f}  FDR={row['p_FDR']:.4f}")
else:
    print("    (none)")

# ── Plot 1 — Heatmap of Cohen's d (Layer 2, raw) ─────────────────────────────
print("\nGenerating plots...")

for layer_df, layer_label, layer_fname in [
    (pw_df,     'Raw',           'heatmap_raw'),
    (pw_age_df, 'Age-corrected', 'heatmap_age_corrected'),
]:
    pivot_d = layer_df.pivot(index='Feature', columns='Comparison', values='Cohens_d')
    pivot_p = layer_df.pivot(index='Feature', columns='Comparison', values='p_FDR')
    pivot_d = pivot_d.loc[FEATURE_COLS, PAIR_LABELS]
    pivot_p = pivot_p.loc[FEATURE_COLS, PAIR_LABELS]

    # Sort rows by mean |d|
    sort_idx = pivot_d.abs().mean(axis=1).sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(8, 9))
    vmax = max(0.8, float(pivot_d.abs().max().max()))
    im = ax.imshow(pivot_d.loc[sort_idx].values,
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='auto', interpolation='nearest')

    for r, feat in enumerate(sort_idx):
        for c, comp in enumerate(PAIR_LABELS):
            p = pivot_p.loc[feat, comp]
            s = stars(p)
            if s and s != 'ns':
                ax.text(c, r, s, ha='center', va='center',
                        fontsize=10, color='black', fontweight='bold')

    ax.set_xticks(range(3))
    ax.set_xticklabels([l.replace(' vs ', '\nvs\n') for l in PAIR_LABELS],
                       fontsize=9)
    ax.set_yticks(range(len(sort_idx)))
    ax.set_yticklabels(list(sort_idx), fontsize=8.5)
    ax.set_title(f"Cohen's d — {layer_label}\n(* FDR<0.05  ** <0.01  *** <0.001)",
                 fontsize=11, pad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Cohen's d  (G2 − G1)", fontsize=9)
    cbar.ax.axhline(0, color='black', lw=0.8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{layer_fname}.png', dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved {layer_fname}.png")

# ── Plot 2 — Forest plot per pairwise comparison (Layer 2) ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 9), sharey=True)
fig.subplots_adjust(wspace=0.08)

feat_order = FEATURE_COLS[::-1]  # bottom-to-top
y = np.arange(len(feat_order))

for ax, (g1, g2), label in zip(axes, PAIRS, PAIR_LABELS):
    sub = pw_df[pw_df['Comparison'] == label].set_index('Feature')
    for i, feat in enumerate(feat_order):
        if feat not in sub.index:
            continue
        row = sub.loc[feat]
        d  = row['Cohens_d']
        sig = row['Significant']
        color = '#1565C0' if sig else '#9E9E9E'
        alpha = 1.0 if sig else 0.4
        ax.plot([0, d], [i, i], color=color, lw=2.0 if sig else 0.8, alpha=alpha, zorder=2)
        ax.scatter(d, i, color=color, s=70 if sig else 30, alpha=alpha,
                   edgecolors='white' if sig else 'none', lw=0.6, zorder=3)
        s = stars(row['p_FDR'])
        if s and s != 'ns':
            offset = 0.04 if d >= 0 else -0.04
            ax.text(d + offset, i, s, ha='left' if d >= 0 else 'right',
                    va='center', fontsize=7.5, color=color, fontweight='bold')

    ax.axvline(0, color='#aaaaaa', lw=0.8, ls='--', zorder=1)
    ax.set_title(label, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel("Cohen's d", fontsize=10)
    ax.grid(axis='x', color='#eeeeee', lw=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Organelle bands
    actin_ys  = [i for i, f in enumerate(feat_order) if f.startswith('Actin')]
    mito_ys   = [i for i, f in enumerate(feat_order) if f.startswith('Mito')]
    nuc_ys    = [i for i, f in enumerate(feat_order) if f.startswith('Nucleus')]
    for ys, col in [(actin_ys,'#E3F2FD'), (mito_ys,'#E8F5E9'), (nuc_ys,'#FFF8E1')]:
        if ys:
            ax.axhspan(min(ys)-0.5, max(ys)+0.5, color=col, alpha=0.4, zorder=0)

axes[0].set_yticks(y)
axes[0].set_yticklabels(feat_order, fontsize=8.5)

legend_handles = [
    mpatches.Patch(color='#1565C0', label='Significant (FDR<0.05)'),
    mpatches.Patch(color='#9E9E9E', label='Not significant'),
    mpatches.Patch(color='#E3F2FD', alpha=0.8, label='Actin'),
    mpatches.Patch(color='#E8F5E9', alpha=0.8, label='Mitochondria'),
    mpatches.Patch(color='#FFF8E1', alpha=0.8, label='Nucleus'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=5,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

fig.suptitle("Pairwise Disease Effect — Raw (Layer 2)\n3-Group Comparison",
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig(OUT_DIR / 'forest_pairwise_raw.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved forest_pairwise_raw.png")

# ── Plot 3 — Boxplots for omnibus-significant features ────────────────────────
sig_feats = kw_df[kw_df['KW_sig']]['Feature'].tolist()

if sig_feats:
    ncols = 3
    nrows = int(np.ceil(len(sig_feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(sig_feats):
        ax = axes[i]
        data = [df_nc[df_nc['Group'] == g][feat].dropna().values for g in GROUPS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', lw=2))
        for patch, g in zip(bp['boxes'], GROUPS):
            patch.set_facecolor(GROUP_COLORS[g])
            patch.set_alpha(0.7)

        # Add individual points
        for j, (d_vals, g) in enumerate(zip(data, GROUPS)):
            jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(d_vals))
            ax.scatter(np.ones(len(d_vals)) * (j + 1) + jitter, d_vals,
                       color=GROUP_COLORS[g], alpha=0.4, s=15, zorder=3)

        # Add pairwise significance brackets
        for (g1, g2), label in zip(PAIRS, PAIR_LABELS):
            row_pw = pw_df[(pw_df['Feature'] == feat) & (pw_df['Comparison'] == label)]
            if not row_pw.empty and row_pw.iloc[0]['Significant']:
                s = stars(row_pw.iloc[0]['p_FDR'])
                x1 = GROUPS.index(g1) + 1
                x2 = GROUPS.index(g2) + 1
                y_max = max(max(d) for d in data if len(d))
                y_range = y_max - min(min(d) for d in data if len(d))
                y_br = y_max + y_range * 0.08
                h = y_range * 0.04
                ax.plot([x1, x1, x2, x2], [y_br, y_br+h, y_br+h, y_br],
                        lw=1.2, color='#333333')
                ax.text((x1+x2)/2, y_br+h, s, ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='#333333')

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(GROUPS, fontsize=8)
        ax.set_title(feat.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide empty axes
    for j in range(len(sig_feats), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'boxplots_significant.png', dpi=160, bbox_inches='tight')
    plt.close()
    print("  Saved boxplots_significant.png")

# ── Summary table ─────────────────────────────────────────────────────────────
summary_rows = []
for feat in FEATURE_COLS:
    row = {'Feature': feat}
    # KW
    kw_row = kw_df[kw_df['Feature'] == feat].iloc[0]
    row['KW_FDR'] = round(kw_row['KW_FDR'], 4)
    row['KW_sig'] = kw_row['KW_sig']
    # Pairwise
    for label in PAIR_LABELS:
        key = label.replace(' ', '_').replace('-', '').replace('vs_', 'vs')
        pw_row = pw_df[(pw_df['Feature'] == feat) & (pw_df['Comparison'] == label)].iloc[0]
        row[f'd_{key}'] = round(pw_row['Cohens_d'], 3)
        row[f'FDR_{key}'] = round(pw_row['p_FDR'], 4)
        row[f'sig_{key}'] = pw_row['Significant']
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / 'summary_table.csv', index=False)
print("  Saved summary_table.csv")

print(f"\nAll outputs saved to {OUT_DIR}/")
print("\nDone.")
