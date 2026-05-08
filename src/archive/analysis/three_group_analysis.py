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

# ── Sex effect (Male vs Female) ───────────────────────────────────────────────
print("\nSex effects (Male vs Female)...")
sex_rows = []
for feat in FEATURE_COLS:
    d_male   = df_nc[df_nc['Sex'] == 'Male'][feat].dropna().values
    d_female = df_nc[df_nc['Sex'] == 'Female'][feat].dropna().values
    if len(d_male) < 3 or len(d_female) < 3:
        sex_rows.append({'Feature': feat, 'Cohens_d': np.nan, 'p_value': 1.0, 'Test': 'N/A'})
        continue
    p, test = adaptive_test(d_male, d_female)
    d = cohens_d(d_female, d_male)   # positive = male > female
    sex_rows.append({
        'Feature': feat,
        'Mean_Male': np.mean(d_male), 'Mean_Female': np.mean(d_female),
        'Cohens_d': d, 'p_value': p, 'Test': test,
    })

sex_df = pd.DataFrame(sex_rows)
sex_fdr = fdr_bh(sex_df['p_value'].fillna(1).values)
sex_df['p_FDR'] = sex_fdr
sex_df['Significant'] = sex_fdr < 0.05
sex_df = sex_df.set_index('Feature')
n_sex_sig = sex_df['Significant'].sum()
print(f"  Significant sex effects (FDR<0.05): {n_sex_sig} features")

# ── Age correlation ───────────────────────────────────────────────────────────
print("Age correlations (Spearman)...")
from scipy.stats import spearmanr
age_rows = []
for feat in FEATURE_COLS:
    sub = df_nc[['Age', feat]].dropna()
    if len(sub) < 10:
        age_rows.append({'Feature': feat, 'Spearman_r': np.nan, 'p_value': 1.0})
        continue
    r, p = spearmanr(sub[feat], sub['Age'])
    age_rows.append({'Feature': feat, 'Spearman_r': r, 'p_value': p})

age_df_corr = pd.DataFrame(age_rows)
age_fdr = fdr_bh(age_df_corr['p_value'].fillna(1).values)
age_df_corr['p_FDR'] = age_fdr
age_df_corr['Significant'] = age_fdr < 0.05
age_df_corr = age_df_corr.set_index('Feature')

# ── Per-feature boxplots (all features) ──────────────────────────────────────
print("\nGenerating per-feature boxplots...")
BOXPLOT_DIR = OUT_DIR / 'pairwise_boxplots'
BOXPLOT_DIR.mkdir(exist_ok=True)

# Bracket draw order: adjacent pairs first, spanning pair last
BRACKET_ORDER = [
    ('TAV-NA',   'TAV-ATAA'),
    ('TAV-ATAA', 'BAV-ATAA'),
    ('TAV-NA',   'BAV-ATAA'),
]

def add_bracket(ax, x1, x2, y, h, p_fdr, fontsize=9):
    s = stars(p_fdr)
    if not s or s == 'ns':
        return
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='#333333')
    ax.text((x1+x2)/2, y+h, s, ha='center', va='bottom',
            fontsize=fontsize, color='#333333', fontweight='bold')

def shorten(feat):
    return feat.replace('_', ' ')

pos_map = {g: i for i, g in enumerate(GROUPS)}

for feat in FEATURE_COLS:
    fig, ax = plt.subplots(figsize=(6, 5))

    data_by_group = [df_nc[df_nc['Group'] == g][feat].dropna().values for g in GROUPS]
    positions = list(range(len(GROUPS)))

    bp = ax.boxplot(data_by_group,
                    positions=positions,
                    widths=0.5,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=3,
                                    markerfacecolor='none', alpha=0.4),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    boxprops=dict(linewidth=1.2))

    for patch, grp in zip(bp['boxes'], GROUPS):
        patch.set_facecolor(GROUP_COLORS[grp])
        patch.set_alpha(0.8)

    # Mean ± SD diamond
    for i, (grp, vals) in enumerate(zip(GROUPS, data_by_group)):
        if len(vals) > 0:
            m, s = np.mean(vals), np.std(vals)
            ax.plot(i, m, marker='D', color='black', markersize=6, zorder=5,
                    label='Mean ± SD' if i == 0 else '')
            ax.plot([i, i], [m - s, m + s], color='black', linewidth=1.5, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(GROUPS, fontsize=10)
    ax.set_ylabel(feat, fontsize=9)
    ax.set_title(shorten(feat), fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Significance brackets
    all_vals = np.concatenate([v for v in data_by_group if len(v) > 0])
    if len(all_vals) > 0:
        y_max   = np.percentile(all_vals, 97)
        y_range = y_max - np.min(all_vals)
        h   = y_range * 0.04
        gap = y_range * 0.09

        feat_pw = pw_df[pw_df['Feature'] == feat]
        for level, (bg1, bg2) in enumerate(BRACKET_ORDER):
            label_match = next(
                (lbl for (g1, g2), lbl in zip(PAIRS, PAIR_LABELS) if g1 == bg1 and g2 == bg2),
                None)
            if label_match is None:
                continue
            row = feat_pw[feat_pw['Comparison'] == label_match]
            if row.empty:
                continue
            add_bracket(ax, pos_map[bg1], pos_map[bg2],
                        y_max + gap * (level + 1), h, row['p_FDR'].values[0])

        ax.set_ylim(bottom=ax.get_ylim()[0],
                    top=y_max + gap * (len(BRACKET_ORDER) + 1) + h * 2)

    ax.legend(handles=[plt.Line2D([0], [0], marker='D', color='black',
                                  markersize=6, linewidth=0, label='Mean ± SD')],
              loc='upper right', fontsize=8, framealpha=0.9)

    # Age annotation (if significant)
    if feat in age_df_corr.index:
        ar = age_df_corr.loc[feat]
        if ar['Significant'] and not np.isnan(ar['Spearman_r']):
            s = stars(ar['p_FDR'])
            ax.annotate(f"Age: ρ={ar['Spearman_r']:+.2f} {s}",
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        ha='left', va='top', fontsize=8, color='#555555',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='#aaaaaa', alpha=0.85, linewidth=0.8))

    # Sex annotation (if significant)
    if feat in sex_df.index:
        sr = sex_df.loc[feat]
        if sr['Significant'] and not np.isnan(sr['Cohens_d']):
            direction = '♂ > ♀' if sr['Cohens_d'] > 0 else '♀ > ♂'
            s = stars(sr['p_FDR'])
            ax.annotate(f'Sex: {direction} {s}',
                        xy=(0.98, 0.02), xycoords='axes fraction',
                        ha='right', va='bottom', fontsize=9,
                        color='#7B68EE', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='#7B68EE', alpha=0.9, linewidth=1))

    plt.tight_layout()
    plt.savefig(BOXPLOT_DIR / f'{feat}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {feat}.png")

print(f"\nPer-feature boxplots saved to {BOXPLOT_DIR}/")

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Clinical Metadata Analyses
#
# Motivation:
#   Layers 1–3 treated disease group (TAV-NA / TAV-ATAA / BAV-ATAA) as the
#   only independent variable. But patient-level clinical variables can either
#   (a) confound the result — if, say, hypertension is the true driver of SMC
#       remodelling and happens to be more prevalent in aneurysm patients, then
#       what we call a "disease effect" is partly a "hypertension effect"; or
#   (b) modulate the result — if morphology tracks aortic diameter within the
#       aneurysm group, that means SMC phenotype scales with disease severity,
#       not just presence/absence of aneurysm. That is a much stronger finding.
#
# We therefore run two targeted analyses:
#
#   Layer 4a — Aortic diameter ↔ morphology (Spearman correlation)
#              Does SMC phenotype worsen as the aorta dilates further?
#
#   Layer 4b — Hypertension as covariate (Mann-Whitney / T-test)
#              Does HTN status explain morphological variation within the
#              aneurysm group, independent of valve type?
#
# IMPORTANT DESIGN DECISION — specimen level vs cell level:
#   Every cell in df_nc from the same patient shares the same aortic diameter,
#   the same HTN status, the same age. Treating cells as independent would be
#   pseudoreplication: we would be inflating N (e.g. N=75 cells instead of
#   N=15 specimens) while the "signal" is entirely patient-level. This produces
#   artificially narrow confidence intervals and misleadingly low p-values.
#   Solution: collapse each specimen to a single mean per feature BEFORE
#   any statistical test. The specimen is the unit of observation.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LAYER 4 — Clinical metadata analyses")
print("="*60)

import re as _re

LAYER4_DIR = OUT_DIR / 'layer4_clinical'
LAYER4_DIR.mkdir(exist_ok=True)

# ── 4.0 Metadata dictionary (specimen level) ──────────────────────────────────
#
# Variables extracted from STS database records for each specimen:
#
#   max_ao_diam : largest aortic diameter (mm) in the treatment zone
#                 (column trtznlrgdiammeas from pre-op CT or 3D reconstruction)
#                 This is the gold-standard measurement for aneurysm severity
#                 and the clinical threshold for surgery is ≥55 mm.
#                 NaN for TAV-NA controls (no aneurysm) and for specimens where
#                 the primary indication was non-aneurysmal (e.g., 03asc-43 had
#                 stenosis/obstruction as primary diagnosis → no diameter recorded).
#
#   hypertn     : systemic hypertension diagnosis ('Yes' / 'No' / NaN = unknown)
#                 The 01C specimens (isolated CABG) had minimal clinical records,
#                 so HTN status is unknown for most of them.
#
#   bsa, bmi    : body surface area (m²) and body mass index
#                 Used to assess whether cell-size differences (e.g. actin volume)
#                 are explained by patient body size rather than disease.
#
#   tobacco     : smoking history ('Never' / 'Former' / 'Current')
#
# Keys are normalised specimen IDs (lowercase, leading zeros stripped from
# the numeric suffix) to match the 'Specimen' column in the features file.
# Special case: metadata records 03Asc-0021 as BAV-ATAA but the features file
# uses 02Asc-0021 for this specimen — we use '02asc-21' as the key.

CLINICAL_META = {
    # ── TAV-NA (non-aneurysmal controls) ─────────────────────────────────────
    # Heart transplant donors / CABG patients. No ascending aortic aneurysm,
    # so max_ao_diam is NaN throughout. Full clinical records only for 01Asc;
    # the 01C series are CABG-only with minimal chart data.
    '01asc-180': {'age': 62, 'sex': 'Male',   'bsa': 2.47, 'bmi': 34.45,
                  'hypertn': 'Yes', 'tobacco': 'Former',  'max_ao_diam': np.nan},
    '01asc-222': {'age': 66, 'sex': 'Male',   'bsa': 2.12, 'bmi': 26.49,
                  'hypertn': 'Yes', 'tobacco': 'Never',   'max_ao_diam': np.nan},
    '01asc-230': {'age': 63, 'sex': 'Female', 'bsa': 2.08, 'bmi': 27.81,
                  'hypertn': 'No',  'tobacco': 'Never',   'max_ao_diam': np.nan},
    '01c-83':    {'age': np.nan, 'sex': np.nan, 'bsa': np.nan, 'bmi': np.nan,
                  'hypertn': np.nan, 'tobacco': np.nan,   'max_ao_diam': np.nan},
    '01c-96':    {'age': 53, 'sex': 'Male',   'bsa': np.nan, 'bmi': np.nan,
                  'hypertn': np.nan, 'tobacco': np.nan,   'max_ao_diam': np.nan},
    '01c-97':    {'age': 57, 'sex': 'Female', 'bsa': np.nan, 'bmi': np.nan,
                  'hypertn': np.nan, 'tobacco': np.nan,   'max_ao_diam': np.nan},
    '01c-113':   {'age': 64, 'sex': 'Female', 'bsa': np.nan, 'bmi': np.nan,
                  'hypertn': np.nan, 'tobacco': np.nan,   'max_ao_diam': np.nan},
    '01c-117':   {'age': 46, 'sex': 'Female', 'bsa': np.nan, 'bmi': np.nan,
                  'hypertn': np.nan, 'tobacco': np.nan,   'max_ao_diam': np.nan},
    '01c-202':   {'age': 78, 'sex': 'Female', 'bsa': np.nan, 'bmi': np.nan,
                  'hypertn': np.nan, 'tobacco': np.nan,   'max_ao_diam': np.nan},
    '01asc-231': {'age': 60, 'sex': 'Male',   'bsa': np.nan, 'bmi': np.nan,
                  'hypertn': np.nan, 'tobacco': np.nan,   'max_ao_diam': np.nan},
    '01asc-232': {'age': 49, 'sex': 'Male',   'bsa': np.nan, 'bmi': np.nan,
                  'hypertn': np.nan, 'tobacco': np.nan,   'max_ao_diam': np.nan},

    # ── TAV-ATAA (tricuspid aortic valve + ascending thoracic aortic aneurysm)
    # Diameters from pre-op CT or 3D reconstruction (trtznlrgdiammeas, mm).
    # 03asc-43: primary indication = stenosis/obstruction, no aneurysm diameter.
    '03asc-24':  {'age': 62, 'sex': 'Male',   'bsa': 2.00, 'bmi': 27.83,
                  'hypertn': 'Yes', 'tobacco': 'Never',  'max_ao_diam': 54},
    '03asc-43':  {'age': 70, 'sex': 'Male',   'bsa': 2.16, 'bmi': 27.35,
                  'hypertn': 'Yes', 'tobacco': 'Former', 'max_ao_diam': np.nan},
    '03rt-45':   {'age': 75, 'sex': 'Male',   'bsa': 2.36, 'bmi': 32.79,
                  'hypertn': 'Yes', 'tobacco': 'Former', 'max_ao_diam': 40},
    '03asc-46':  {'age': 75, 'sex': 'Female', 'bsa': 1.60, 'bmi': 26.09,
                  'hypertn': 'Yes', 'tobacco': 'Never',  'max_ao_diam': 48},
    '03asc-47':  {'age': 68, 'sex': 'Male',   'bsa': 2.28, 'bmi': 28.10,
                  'hypertn': 'Yes', 'tobacco': 'Never',  'max_ao_diam': 51},
    '03asc-51':  {'age': 65, 'sex': 'Female', 'bsa': 2.08, 'bmi': 43.96,
                  'hypertn': 'Yes', 'tobacco': 'Never',  'max_ao_diam': 55},
    '03asc-54':  {'age': 75, 'sex': 'Female', 'bsa': 1.56, 'bmi': 27.58,
                  'hypertn': 'Yes', 'tobacco': 'Never',  'max_ao_diam': 59},
    '03asc-55':  {'age': 66, 'sex': 'Female', 'bsa': 2.15, 'bmi': 36.94,
                  'hypertn': 'Yes', 'tobacco': 'Former', 'max_ao_diam': 55},
    '03asc-56':  {'age': 44, 'sex': 'Female', 'bsa': 1.70, 'bmi': 25.35,
                  'hypertn': 'Yes', 'tobacco': 'Former', 'max_ao_diam': 50},
    '03asc-57':  {'age': 73, 'sex': 'Female', 'bsa': 1.84, 'bmi': 22.58,
                  'hypertn': 'No',  'tobacco': 'Never',  'max_ao_diam': 49},
    '02asc-47':  {'age': 40, 'sex': 'Male',   'bsa': 2.14, 'bmi': 31.95,
                  'hypertn': 'No',  'tobacco': 'Never',  'max_ao_diam': 30},

    # ── BAV-ATAA (bicuspid aortic valve + ascending thoracic aortic aneurysm)
    # Note: the raw metadata records specimen 03Asc-0021 (a naming discrepancy)
    # but the features file uses 02Asc-0021 — we key on the features-file ID.
    '02asc-17':  {'age': 73, 'sex': 'Male',   'bsa': 2.13, 'bmi': 29.13,
                  'hypertn': 'Yes', 'tobacco': 'Never',  'max_ao_diam': 44},
    '02asc-21':  {'age': 61, 'sex': 'Male',   'bsa': 1.77, 'bmi': 24.95,
                  'hypertn': 'Yes', 'tobacco': 'Former', 'max_ao_diam': 52},
    '02asc-29':  {'age': 64, 'sex': 'Female', 'bsa': 1.57, 'bmi': 21.60,
                  'hypertn': 'No',  'tobacco': 'Former', 'max_ao_diam': 50},
    '02asc-49':  {'age': 58, 'sex': 'Female', 'bsa': 1.97, 'bmi': 32.60,
                  'hypertn': 'Yes', 'tobacco': 'Never',  'max_ao_diam': 36},
    '02asc-77':  {'age': np.nan, 'sex': 'Male', 'bsa': np.nan, 'bmi': np.nan,
                  'hypertn': np.nan, 'tobacco': np.nan,  'max_ao_diam': np.nan},
}

# ── 4.1 ID normalisation and join ─────────────────────────────────────────────
# The 'Specimen' column in df_nc uses inconsistent capitalisation and sometimes
# drops leading zeros (e.g. '03ASC-024' for specimen 03Asc-0024, '01C-202' for
# 01C-0202). We normalise both sides to lowercase with leading zeros removed
# from the numeric suffix before joining, so the match is robust.

def _norm(s):
    """Lowercase + strip leading zeros from trailing number: '03ASC-024' → '03asc-24'."""
    return _re.sub(r'-0*(\d+)$', r'-\1', str(s).lower())

meta_df = (pd.DataFrame.from_dict(CLINICAL_META, orient='index')
             .rename_axis('spec_norm')
             .reset_index())

df_joined = df_nc.copy()
df_joined['spec_norm'] = df_joined['Specimen'].apply(_norm)
df_joined = df_joined.merge(meta_df, on='spec_norm', how='left')

n_with_diam = df_joined['max_ao_diam'].notna().sum()
n_with_htn  = df_joined['hypertn'].notna().sum()
print(f"\n  ID join: {len(df_joined)} cells total")
print(f"  Cells with aortic diameter data : {n_with_diam}")
print(f"  Cells with hypertension data    : {n_with_htn}")

# Print any specimen IDs that did not match
unmatched = df_joined[df_joined['hypertn'].isna()]['spec_norm'].unique()
print(f"  Unmatched / no-metadata specs   : {sorted(unmatched)}")

# ── 4.2 Collapse to specimen-level means ──────────────────────────────────────
# For each specimen we compute the mean of each morphological feature across
# all cells. This is the correct statistical unit: the patient, not the cell.
# Clinical variables (diameter, HTN, group) are constant within a specimen so
# we take 'first'; the mean would be the same.

feat_agg  = {f: 'mean' for f in FEATURE_COLS}
clin_agg  = {'max_ao_diam': 'first', 'hypertn': 'first',
             'Group': 'first', 'bsa': 'first', 'bmi': 'first'}

spec_means = (df_joined
              .groupby('spec_norm')
              .agg({**feat_agg, **clin_agg})
              .reset_index())

print(f"\n  Specimens in specimen-level table: {len(spec_means)}")

# ── Layer 4a — Aortic diameter ↔ morphology ───────────────────────────────────
#
# Hypothesis: if ascending aortic dilation directly imposes abnormal mechanical
# load on SMCs, then cells in larger aortas should show greater morphological
# derangement. A significant positive ρ for (say) Mito_Fragment_Count with
# diameter would mean: the more dilated the aorta, the more fragmented the
# mitochondrial network — consistent with progressive mechanosensory stress.
#
# Statistical choice — Spearman rather than Pearson:
#   With N≈13 specimens the normality assumption for Pearson is not verifiable.
#   Spearman uses only ranks, making it robust to the outlier at 59 mm (03asc-54)
#   and to any monotonic (not necessarily linear) dose-response relationship.
#   We do NOT partial out age or sex here because N=13 specimens means a
#   partial Spearman would have essentially zero power. We note this limitation.
#
# Scope: TAV-ATAA + BAV-ATAA combined (aneurysm specimens with diameter data).
# TAV-NA is excluded because their diameter is undefined (no aneurysm).
# We also run TAV-ATAA alone for sensitivity (N≈9, interpret cautiously).

print("\nLayer 4a — Aortic diameter correlations (Spearman, specimen level)")

diam_sub = spec_means[spec_means['max_ao_diam'].notna()].copy()
n_diam   = len(diam_sub)
print(f"  N = {n_diam} specimens  |  "
      f"diameter range: {diam_sub['max_ao_diam'].min():.0f}–"
      f"{diam_sub['max_ao_diam'].max():.0f} mm")
print(f"  Group counts: {diam_sub.groupby('Group').size().to_dict()}")

diam_rows = []
for feat in FEATURE_COLS:
    sub = diam_sub[['max_ao_diam', feat]].dropna()
    if len(sub) < 4:   # need ≥4 pairs for a meaningful rank correlation
        diam_rows.append({'Feature': feat, 'rho': np.nan, 'p_raw': np.nan, 'N': len(sub)})
        continue
    rho, p = spearmanr(sub['max_ao_diam'], sub[feat])
    diam_rows.append({'Feature': feat, 'rho': rho, 'p_raw': p, 'N': len(sub)})

diam_df = pd.DataFrame(diam_rows)
# FDR across all 21 features simultaneously. Because we are running 21 tests
# on a single outcome variable (diameter), BH-FDR is the standard correction.
diam_df['p_FDR'] = fdr_bh(diam_df['p_raw'].fillna(1).values)
diam_df['sig']   = diam_df['p_FDR'] < 0.05
diam_df.to_csv(LAYER4_DIR / 'diameter_correlation.csv', index=False)

sig_diam = diam_df[diam_df['sig']].sort_values('rho', key=abs, ascending=False)
print(f"\n  FDR-significant diameter correlations: {len(sig_diam)}")
if len(sig_diam):
    for _, r in sig_diam.iterrows():
        print(f"    {r['Feature']:40s}  ρ={r['rho']:+.3f}  FDR={r['p_FDR']:.4f}")
else:
    print("  (none — expected at N≈13; inspect effect sizes below)")

print("\n  Top 5 correlations by |ρ|:")
top5 = diam_df.dropna(subset=['rho']).iloc[
    diam_df.dropna(subset=['rho'])['rho'].abs().argsort()[-5:][::-1].values]
for _, r in top5.iterrows():
    sig_flag = '*' if r['sig'] else ' '
    print(f"    {sig_flag} {r['Feature']:40s}  ρ={r['rho']:+.3f}  "
          f"p_raw={r['p_raw']:.3f}  FDR={r['p_FDR']:.3f}  N={int(r['N'])}")

# Also run within TAV-ATAA alone for sensitivity
diam_tav = diam_sub[diam_sub['Group'] == 'TAV-ATAA']
print(f"\n  Sensitivity: TAV-ATAA only (N={len(diam_tav)})")
tav_sens = []
for feat in FEATURE_COLS:
    sub = diam_tav[['max_ao_diam', feat]].dropna()
    if len(sub) < 4:
        continue
    rho, p = spearmanr(sub['max_ao_diam'], sub[feat])
    if p < 0.05:
        tav_sens.append((feat, rho, p))
if tav_sens:
    for feat, rho, p in sorted(tav_sens, key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat:40s}  ρ={rho:+.3f}  p_raw={p:.3f}")
else:
    print("    (none at p<0.05 uncorrected)")

# ── Layer 4b — Hypertension as morphological covariate ───────────────────────
#
# Hypertension chronically elevates pulsatile wall stress. It independently
# promotes SMC phenotype switching (from contractile to synthetic), actin
# cytoskeletal remodelling, mitochondrial dysfunction, and nuclear stiffening.
# If HTN+ patients show different morphology from HTN- patients within the
# aneurysm group, then:
#   - Some of what we call "disease effect" in Layers 1–2 is actually an
#     "HTN effect" (confounder), OR
#   - HTN amplifies the disease phenotype (interaction effect).
# Either way, it must be reported.
#
# Scope: TAV-ATAA + BAV-ATAA combined (aneurysm group only).
# TAV-NA is excluded: most 01C specimens lack HTN data, and including them
# would perfectly confound group membership with HTN status.
#
# Sample sizes: HTN+ N=12, HTN- N=3 (very imbalanced).
# Consequence: very low power. We prioritise effect size (Cohen's d) over
# p-value. The p-value threshold is advisory only at this N.
# Positive d → HTN+ group has higher feature value than HTN-.
#
# Adaptive test: same Shapiro-Wilk gating as everywhere else.
# At N=3 in HTN-, Shapiro-Wilk returns p>0.05 trivially (small sample never
# rejects normality) → most tests will fall through to T-test. This is fine
# because Welch's T-test is robust to mild non-normality at N=12 vs 3.

print("\nLayer 4b — Hypertension covariate (aneurysm group, specimen level)")

aneurysm_specs = spec_means[spec_means['Group'].isin(['TAV-ATAA', 'BAV-ATAA'])].copy()
htn_known      = aneurysm_specs[aneurysm_specs['hypertn'].isin(['Yes', 'No'])]
n_yes = (htn_known['hypertn'] == 'Yes').sum()
n_no  = (htn_known['hypertn'] == 'No').sum()
print(f"  HTN+: {n_yes} specimens  |  HTN-: {n_no} specimens")
print(f"  Note: low N in HTN- group → interpret effect sizes, not only p-values")

htn_rows = []
for feat in FEATURE_COLS:
    d_yes = htn_known[htn_known['hypertn'] == 'Yes'][feat].dropna().values
    d_no  = htn_known[htn_known['hypertn'] == 'No'][feat].dropna().values
    if len(d_yes) < 2 or len(d_no) < 2:
        htn_rows.append({'Feature': feat, 'Cohens_d': np.nan, 'p_raw': np.nan,
                         'Test': 'N/A', 'N_yes': len(d_yes), 'N_no': len(d_no)})
        continue
    p, test = adaptive_test(d_yes, d_no)
    d = cohens_d(d_no, d_yes)   # positive d → HTN+ > HTN-
    htn_rows.append({'Feature': feat, 'Cohens_d': d, 'p_raw': p, 'Test': test,
                     'N_yes': len(d_yes), 'N_no': len(d_no)})

htn_df = pd.DataFrame(htn_rows)
htn_df['p_FDR'] = fdr_bh(htn_df['p_raw'].fillna(1).values)
htn_df['sig']   = htn_df['p_FDR'] < 0.05
htn_df.to_csv(LAYER4_DIR / 'hypertension_effect.csv', index=False)

print(f"\n  FDR-significant HTN effects: {htn_df['sig'].sum()}")
print("  Top 5 by |Cohen's d| (effect size guide: 0.2=small, 0.5=medium, 0.8=large):")
top_htn = (htn_df.dropna(subset=['Cohens_d'])
               .assign(_abs_d=lambda x: x['Cohens_d'].abs())
               .nlargest(5, '_abs_d'))
for _, r in top_htn.iterrows():
    direction = 'HTN+ ↑' if r['Cohens_d'] > 0 else 'HTN- ↑'
    sig_flag  = '*' if r['sig'] else ' '
    print(f"    {sig_flag} {r['Feature']:40s}  d={r['Cohens_d']:+.3f}  "
          f"{direction}  FDR={r['p_FDR']:.3f}")

# ── 4. Plots ──────────────────────────────────────────────────────────────────

# Plot 4a-1 — Diameter correlation forest plot (all 21 features)
# One row per feature. Bar length = Spearman ρ. Filled = FDR<0.05.
# Two colours: blue = positive (larger aorta → more of feature),
#              red  = negative (larger aorta → less of feature).
# This lets the reader immediately see the direction of each correlation.

feat_order_diam = list(reversed(FEATURE_COLS))  # bottom = actin, top = nucleus
rho_vals  = [diam_df.set_index('Feature').loc[f, 'rho']   for f in feat_order_diam]
sig_vals  = [diam_df.set_index('Feature').loc[f, 'sig']   for f in feat_order_diam]
fdr_vals  = [diam_df.set_index('Feature').loc[f, 'p_FDR'] for f in feat_order_diam]

fig, ax = plt.subplots(figsize=(7, 8))
y = np.arange(len(feat_order_diam))

for i, (feat, rho, sig, fdr) in enumerate(zip(feat_order_diam, rho_vals, sig_vals, fdr_vals)):
    if np.isnan(rho):
        continue
    color = '#1565C0' if rho > 0 else '#C62828'
    alpha = 1.0 if sig else 0.35
    ax.barh(i, rho, color=color, alpha=alpha, height=0.6, zorder=2)
    s = stars(fdr)
    if s and s != 'ns':
        xpos = rho + (0.02 if rho >= 0 else -0.02)
        ax.text(xpos, i, s, va='center',
                ha='left' if rho >= 0 else 'right',
                fontsize=8, color=color, fontweight='bold')

ax.axvline(0, color='#555555', lw=0.8, ls='--', zorder=1)
ax.set_yticks(y)
ax.set_yticklabels(feat_order_diam, fontsize=8)
ax.set_xlabel("Spearman ρ  (feature vs. aortic diameter)", fontsize=10)
ax.set_title(f"Aortic diameter ↔ morphology\n"
             f"TAV-ATAA + BAV-ATAA combined  (N={n_diam} specimens)\n"
             f"Solid = FDR<0.05, faded = ns",
             fontsize=10, pad=8)
ax.set_xlim(-1.05, 1.05)
ax.grid(axis='x', color='#eeeeee', lw=0.6, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Organelle bands
for feats_group, col in [
    ([f for f in feat_order_diam if f.startswith('Actin')],   '#E3F2FD'),
    ([f for f in feat_order_diam if f.startswith('Mito')],    '#E8F5E9'),
    ([f for f in feat_order_diam if f.startswith('Nucleus')], '#FFF8E1'),
]:
    idxs = [feat_order_diam.index(f) for f in feats_group]
    if idxs:
        ax.axhspan(min(idxs) - 0.5, max(idxs) + 0.5, color=col, alpha=0.35, zorder=0)

plt.tight_layout()
plt.savefig(LAYER4_DIR / 'diameter_correlation_forest.png', dpi=180, bbox_inches='tight')
plt.close()
print("\n  Saved diameter_correlation_forest.png")

# Plot 4a-2 — Scatter plots for the 4 features with the strongest |ρ|
# Each panel: x = aortic diameter (mm), y = specimen-mean feature value.
# Points coloured by group (TAV-ATAA vs BAV-ATAA) so the reader can see
# whether the correlation is group-driven or cross-group.
# We annotate each panel with ρ and the FDR-corrected p-value.
# We show a Spearman-based monotone trend line (LOWESS) not a linear fit,
# because Spearman makes no linearity assumption.

from scipy.stats import rankdata

top4_feats = (diam_df.dropna(subset=['rho'])
              .assign(_abs=lambda x: x['rho'].abs())
              .nlargest(4, '_abs')['Feature'].tolist())

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
scatter_colors = {'TAV-ATAA': GROUP_COLORS['TAV-ATAA'],
                  'BAV-ATAA': GROUP_COLORS['BAV-ATAA']}

for ax, feat in zip(axes, top4_feats):
    sub = diam_sub[['max_ao_diam', feat, 'Group']].dropna()
    for grp, grp_data in sub.groupby('Group'):
        ax.scatter(grp_data['max_ao_diam'], grp_data[feat],
                   color=scatter_colors.get(grp, '#888888'),
                   s=70, alpha=0.85, edgecolors='white', lw=0.5,
                   label=grp, zorder=3)

    # Trend line: fit on ranks (consistent with Spearman)
    if len(sub) >= 4:
        x_sorted = sub['max_ao_diam'].sort_values()
        rank_x = rankdata(sub['max_ao_diam'])
        rank_y = rankdata(sub[feat])
        # linear fit on ranks gives the rank-regression line
        m, b = np.polyfit(rank_x, rank_y, 1)
        x_line = np.linspace(sub['max_ao_diam'].min(), sub['max_ao_diam'].max(), 50)
        r_line = np.linspace(rank_x.min(), rank_x.max(), 50)
        # Back-transform ranks to original units via sorted values
        x_orig_sorted = np.sort(sub['max_ao_diam'].values)
        y_orig_sorted = np.sort(sub[feat].values)
        y_line_ranks  = m * r_line + b
        # clamp rank predictions to valid range then interpolate
        y_line_ranks_clamped = np.clip(y_line_ranks, 1, len(sub))
        y_line = np.interp(y_line_ranks_clamped,
                           np.arange(1, len(sub)+1), y_orig_sorted)
        ax.plot(x_line, y_line, color='#444444', lw=1.2, ls='--',
                alpha=0.6, zorder=2)

    row_f = diam_df[diam_df['Feature'] == feat].iloc[0]
    sig_str = stars(row_f['p_FDR']) if not np.isnan(row_f['p_FDR']) else ''
    ax.set_xlabel('Aortic diameter (mm)', fontsize=9)
    ax.set_ylabel(feat.replace('_', ' '), fontsize=8)
    ax.set_title(f"{feat.replace('_', ' ')}\nρ={row_f['rho']:+.2f}  FDR={row_f['p_FDR']:.3f} {sig_str}",
                 fontsize=8.5, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25)

handles = [plt.scatter([], [], color=c, s=50, label=g)
           for g, c in scatter_colors.items()]
axes[-1].legend(handles=handles, fontsize=8, framealpha=0.9)

fig.suptitle("Top 4 features by |ρ| with aortic diameter",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(LAYER4_DIR / 'diameter_scatter_top4.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved diameter_scatter_top4.png")

# Plot 4b — Hypertension forest plot (all 21 features)
# One row per feature. x-axis = Cohen's d (HTN+ minus HTN-).
# Filled bars = FDR<0.05 (likely none given N, but we show effect sizes).
# Colour: orange = HTN+ higher, teal = HTN- higher.
# This is the most honest way to present an underpowered comparison:
# show all effect sizes explicitly so the reader can judge clinical relevance.

htn_order = list(reversed(FEATURE_COLS))
d_vals   = [htn_df.set_index('Feature').loc[f, 'Cohens_d'] for f in htn_order]
htn_sigs  = [htn_df.set_index('Feature').loc[f, 'sig']      for f in htn_order]
htn_fdrs  = [htn_df.set_index('Feature').loc[f, 'p_FDR']    for f in htn_order]

fig, ax = plt.subplots(figsize=(7, 8))
y = np.arange(len(htn_order))

for i, (feat, d, sig, fdr) in enumerate(zip(htn_order, d_vals, htn_sigs, htn_fdrs)):
    if np.isnan(d):
        continue
    color = '#E65100' if d > 0 else '#00695C'   # orange = HTN+↑, teal = HTN-↑
    alpha = 1.0 if sig else 0.4
    ax.barh(i, d, color=color, alpha=alpha, height=0.6, zorder=2)
    s = stars(fdr)
    if s and s != 'ns':
        xpos = d + (0.04 if d >= 0 else -0.04)
        ax.text(xpos, i, s, va='center',
                ha='left' if d >= 0 else 'right',
                fontsize=8, color=color, fontweight='bold')

ax.axvline(0,    color='#555555', lw=0.8, ls='--', zorder=1)
ax.axvline( 0.5, color='#aaaaaa', lw=0.5, ls=':',  zorder=1)  # medium effect threshold
ax.axvline(-0.5, color='#aaaaaa', lw=0.5, ls=':',  zorder=1)
ax.axvline( 0.8, color='#888888', lw=0.5, ls=':',  zorder=1)  # large effect threshold
ax.axvline(-0.8, color='#888888', lw=0.5, ls=':',  zorder=1)

ax.set_yticks(y)
ax.set_yticklabels(htn_order, fontsize=8)
ax.set_xlabel("Cohen's d  (HTN+ − HTN−)", fontsize=10)
ax.set_title(f"Hypertension effect on morphology\n"
             f"Aneurysm group (TAV-ATAA + BAV-ATAA)  HTN+: N={n_yes}  HTN−: N={n_no}\n"
             f"Dotted lines: |d|=0.5 (medium), |d|=0.8 (large)",
             fontsize=9, pad=8)
ax.grid(axis='x', color='#eeeeee', lw=0.6, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for feats_group, col in [
    ([f for f in htn_order if f.startswith('Actin')],   '#E3F2FD'),
    ([f for f in htn_order if f.startswith('Mito')],    '#E8F5E9'),
    ([f for f in htn_order if f.startswith('Nucleus')], '#FFF8E1'),
]:
    idxs = [htn_order.index(f) for f in feats_group]
    if idxs:
        ax.axhspan(min(idxs)-0.5, max(idxs)+0.5, color=col, alpha=0.35, zorder=0)

plt.tight_layout()
plt.savefig(LAYER4_DIR / 'hypertension_forest.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved hypertension_forest.png")

print(f"\nLayer 4 outputs saved to {LAYER4_DIR}/")

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — Sex Study
#
# Why sex matters here:
#   Sex is a fundamental biological variable in aortic disease. Males and
#   females differ in:
#     - Aortic diameter thresholds (females develop symptoms at smaller absolute
#       diameters due to smaller body size)
#     - Oestrogen biology: oestrogen promotes SMC contractile phenotype
#       maintenance; post-menopausal females lose this protection
#     - BAV prevalence: ~3:1 male predominance, so the BAV-ATAA group is
#       inherently sex-imbalanced
#     - SMC biology: sex hormones regulate actin organisation, mitochondrial
#       dynamics, and nuclear mechanics
#
# Three levels of analysis:
#
#   5a — Overall sex effect (Male vs Female, all 3 groups combined, N=12 vs 12)
#        This answers: ignoring disease group, do male and female SMCs
#        look morphologically different?
#
#   5b — Within-group sex effects (separately within TAV-NA, TAV-ATAA, BAV-ATAA)
#        This answers: is the sex effect consistent across disease contexts,
#        or does it only appear in one group?
#        Note: BAV-ATAA has only 3M vs 2F — interpret effect sizes only.
#
#   5c — Age-corrected sex effect
#        Although ages are well-balanced between sexes (Male: 61.6y, Female: 63.7y),
#        we run age residualisation as a robustness check. If results replicate
#        after removing the age trend, the sex effect is independent of age.
#
# Statistical unit: specimen mean (same pseudoreplication rationale as Layer 4).
# Statistical test: adaptive (Shapiro-Wilk → Welch T-test or Mann-Whitney).
# FDR correction: BH across 21 features per analysis (5a, 5b-each-group, 5c
#                 each corrected separately — different biological questions).
# Effect direction: positive Cohen's d = Male > Female.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LAYER 5 — Sex study")
print("="*60)

LAYER5_DIR = OUT_DIR / 'layer5_sex'
LAYER5_DIR.mkdir(exist_ok=True)

# ── 5.0 Build specimen-level table with sex ───────────────────────────────────
# We re-use df_nc (no-collagen cells) directly because Sex is already in the
# features file. We do NOT need the clinical metadata join for this layer —
# sex is already reliably recorded for all specimens except 01C-83.

sex_spec = (df_nc.groupby('Specimen')
            .agg({**{f: 'mean' for f in FEATURE_COLS},
                  'Group': 'first', 'Sex': 'first', 'Age': 'first'})
            .reset_index())

# Drop the one specimen with unknown sex
sex_spec = sex_spec[sex_spec['Sex'].isin(['Male', 'Female'])].copy()

# Age-residualise all features for layer 5c
# We fit the linear age trend across ALL specimens (both sexes) and subtract it.
# This removes the shared age-driven variation while keeping between-sex
# differences that are not explained by age.
for feat in FEATURE_COLS:
    sub = sex_spec[['Age', feat]].dropna()
    if len(sub) < 5:
        sex_spec[feat + '_ageresid'] = sex_spec[feat]
        continue
    lr = LinearRegression().fit(sub[['Age']].values, sub[feat].values)
    resid = sex_spec[feat].copy()
    valid = sex_spec['Age'].notna() & sex_spec[feat].notna()
    resid.loc[valid] = (sex_spec.loc[valid, feat].values
                        - lr.predict(sex_spec.loc[valid, ['Age']].values))
    sex_spec[feat + '_ageresid'] = resid

n_male   = (sex_spec['Sex'] == 'Male').sum()
n_female = (sex_spec['Sex'] == 'Female').sum()
print(f"\n  Specimen-level: {n_male} Male, {n_female} Female")
print(f"  Age: Male {sex_spec[sex_spec['Sex']=='Male']['Age'].mean():.1f}y  "
      f"Female {sex_spec[sex_spec['Sex']=='Female']['Age'].mean():.1f}y")
print(f"  Group × Sex:")
print(sex_spec.groupby(['Group','Sex']).size().unstack(fill_value=0).to_string())

# ── Helper: run sex comparison for a given specimen subset ───────────────────
def run_sex_comparison(df_sub, feat_suffix=''):
    """
    Compare Male vs Female for each feature on specimen-level data.
    feat_suffix: '' for raw features, '_ageresid' for age-residualised.
    Returns a DataFrame with Feature, Cohens_d, p_raw, Test, N_male, N_female,
    p_FDR, sig.
    Positive d = Male > Female.
    """
    rows = []
    for feat in FEATURE_COLS:
        col   = feat + feat_suffix
        d_m   = df_sub[df_sub['Sex'] == 'Male'][col].dropna().values
        d_f   = df_sub[df_sub['Sex'] == 'Female'][col].dropna().values
        if len(d_m) < 2 or len(d_f) < 2:
            rows.append({'Feature': feat, 'Cohens_d': np.nan, 'p_raw': np.nan,
                         'Test': 'N/A', 'N_male': len(d_m), 'N_female': len(d_f)})
            continue
        p, test = adaptive_test(d_m, d_f)
        d = cohens_d(d_f, d_m)   # positive = Male > Female
        rows.append({'Feature': feat, 'Cohens_d': d, 'p_raw': p, 'Test': test,
                     'N_male': len(d_m), 'N_female': len(d_f)})
    result = pd.DataFrame(rows)
    result['p_FDR'] = fdr_bh(result['p_raw'].fillna(1).values)
    result['sig']   = result['p_FDR'] < 0.05
    return result.set_index('Feature')

# ── Layer 5a — Overall sex effect ────────────────────────────────────────────
print("\nLayer 5a — Overall sex effect (all groups, N=12M vs 12F)")
sex_overall = run_sex_comparison(sex_spec)
sex_overall.to_csv(LAYER5_DIR / 'sex_overall.csv')

n_sig_ov = sex_overall['sig'].sum()
print(f"  FDR-significant: {n_sig_ov} features")
print("  All features by |d|:")
for feat, r in sex_overall.dropna(subset=['Cohens_d']).sort_values('Cohens_d', key=abs, ascending=False).iterrows():
    sig_flag = '*' if r['sig'] else ' '
    direction = 'M>F' if r['Cohens_d'] > 0 else 'F>M'
    print(f"    {sig_flag} {feat:40s}  d={r['Cohens_d']:+.3f}  {direction}  FDR={r['p_FDR']:.3f}")

# ── Layer 5b — Within-group sex effects ──────────────────────────────────────
print("\nLayer 5b — Within-group sex effects (specimen level)")
sex_bygroup = {}
for grp in GROUPS:
    sub = sex_spec[sex_spec['Group'] == grp]
    nm  = (sub['Sex'] == 'Male').sum()
    nf  = (sub['Sex'] == 'Female').sum()
    print(f"\n  {grp}  (M={nm}, F={nf})")
    if nm < 2 or nf < 2:
        print("    Insufficient N for both sexes — skipping")
        continue
    res = run_sex_comparison(sub)
    sex_bygroup[grp] = res
    res.to_csv(LAYER5_DIR / f'sex_{grp.replace("-","_")}.csv')
    n_sig = res['sig'].sum()
    print(f"  FDR-significant: {n_sig} features")
    top3 = res.dropna(subset=['Cohens_d']).sort_values('Cohens_d', key=abs, ascending=False).head(3)
    for feat, r in top3.iterrows():
        sig_flag = '*' if r['sig'] else ' '
        direction = 'M>F' if r['Cohens_d'] > 0 else 'F>M'
        print(f"    {sig_flag} {feat:40s}  d={r['Cohens_d']:+.3f}  {direction}  FDR={r['p_FDR']:.3f}")

# ── Layer 5c — Age-corrected sex effect ──────────────────────────────────────
print("\nLayer 5c — Age-corrected sex effect (all groups)")
sex_agecorr = run_sex_comparison(sex_spec, feat_suffix='_ageresid')
sex_agecorr.to_csv(LAYER5_DIR / 'sex_overall_age_corrected.csv')

n_sig_ac = sex_agecorr['sig'].sum()
print(f"  FDR-significant after age correction: {n_sig_ac} features")
if n_sig_ac:
    for feat, r in sex_agecorr[sex_agecorr['sig']].sort_values('Cohens_d', key=abs, ascending=False).iterrows():
        direction = 'M>F' if r['Cohens_d'] > 0 else 'F>M'
        print(f"    * {feat:40s}  d={r['Cohens_d']:+.3f}  {direction}  FDR={r['p_FDR']:.3f}")
else:
    print("  (none)")

# ── 5. Plots ──────────────────────────────────────────────────────────────────

# Plot 5-1 — Overall sex effect: forest plot (raw vs age-corrected side by side)
#
# Two panels sharing the y-axis (feature list). Left = raw, right = age-corrected.
# Bar length = Cohen's d. Orange = Male > Female, Teal = Female > Male.
# Solid fill = FDR<0.05 in that analysis. Faded = ns.
# Vertical reference lines at d = ±0.5 (medium) and ±0.8 (large).
# This layout lets the reader immediately see which effects survive age correction.

feat_order_sex = list(reversed(FEATURE_COLS))

fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
fig.subplots_adjust(wspace=0.05)

for ax, res, label in [
    (axes[0], sex_overall,  'Raw'),
    (axes[1], sex_agecorr,  'Age-corrected'),
]:
    y = np.arange(len(feat_order_sex))
    for i, feat in enumerate(feat_order_sex):
        if feat not in res.index or np.isnan(res.loc[feat, 'Cohens_d']):
            continue
        d   = res.loc[feat, 'Cohens_d']
        sig = res.loc[feat, 'sig']
        fdr = res.loc[feat, 'p_FDR']
        color = '#E65100' if d > 0 else '#00695C'   # orange = M>F, teal = F>M
        alpha = 1.0 if sig else 0.35
        ax.barh(i, d, color=color, alpha=alpha, height=0.6, zorder=2)
        s = stars(fdr)
        if s and s != 'ns':
            xpos = d + (0.04 if d >= 0 else -0.04)
            ax.text(xpos, i, s, va='center',
                    ha='left' if d >= 0 else 'right',
                    fontsize=8, color=color, fontweight='bold')

    ax.axvline(0,    color='#555555', lw=0.8, ls='--', zorder=1)
    ax.axvline( 0.5, color='#aaaaaa', lw=0.5, ls=':',  zorder=1)
    ax.axvline(-0.5, color='#aaaaaa', lw=0.5, ls=':',  zorder=1)
    ax.axvline( 0.8, color='#888888', lw=0.5, ls=':',  zorder=1)
    ax.axvline(-0.8, color='#888888', lw=0.5, ls=':',  zorder=1)
    ax.set_xlabel("Cohen's d  (Male − Female)", fontsize=10)
    ax.set_title(f"Sex effect — {label}\nN={n_male}M vs {n_female}F  "
                 f"(all groups combined)",
                 fontsize=9, pad=6)
    ax.grid(axis='x', color='#eeeeee', lw=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for feats_grp, col in [
        ([f for f in feat_order_sex if f.startswith('Actin')],   '#E3F2FD'),
        ([f for f in feat_order_sex if f.startswith('Mito')],    '#E8F5E9'),
        ([f for f in feat_order_sex if f.startswith('Nucleus')], '#FFF8E1'),
    ]:
        idxs = [feat_order_sex.index(f) for f in feats_grp]
        if idxs:
            ax.axhspan(min(idxs)-0.5, max(idxs)+0.5, color=col, alpha=0.35, zorder=0)

axes[0].set_yticks(np.arange(len(feat_order_sex)))
axes[0].set_yticklabels(feat_order_sex, fontsize=8)

import matplotlib.patches as _mpatches
legend_handles = [
    _mpatches.Patch(color='#E65100', label='Male > Female'),
    _mpatches.Patch(color='#00695C', label='Female > Male'),
    _mpatches.Patch(color='white',   label='Solid=FDR<0.05, faded=ns',
                    edgecolor='#888888'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=9,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Sex effect on SMC morphology\n"
             "Left: raw features   Right: age-residualised features",
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(LAYER5_DIR / 'sex_forest_overall.png', dpi=180, bbox_inches='tight')
plt.close()
print("\n  Saved sex_forest_overall.png")

# Plot 5-2 — Within-group heatmap of Cohen's d
#
# Rows = features (ordered by organelle), columns = groups.
# Cell colour = Cohen's d (red = Male > Female, blue = Female > Male).
# Star annotation for FDR<0.05.
# This is the most compact way to show whether sex effects are consistent
# or specific to one disease group. A feature that shows the same direction
# in all 3 columns is a robust sex difference. One that flips direction is
# a sex × disease interaction (scientifically interesting but interpret carefully).

groups_with_data = [g for g in GROUPS if g in sex_bygroup]
n_grps = len(groups_with_data)

if n_grps >= 2:
    d_matrix  = np.full((len(FEATURE_COLS), n_grps), np.nan)
    p_matrix  = np.full((len(FEATURE_COLS), n_grps), np.nan)
    for j, grp in enumerate(groups_with_data):
        res = sex_bygroup[grp]
        for i, feat in enumerate(FEATURE_COLS):
            if feat in res.index:
                d_matrix[i, j] = res.loc[feat, 'Cohens_d']
                p_matrix[i, j] = res.loc[feat, 'p_FDR']

    # Sort rows by |mean d| across groups
    mean_abs_d = np.nanmean(np.abs(d_matrix), axis=1)
    sort_idx   = np.argsort(mean_abs_d)[::-1]
    feat_sorted = [FEATURE_COLS[i] for i in sort_idx]

    vmax = max(0.8, float(np.nanmax(np.abs(d_matrix))))
    fig, ax = plt.subplots(figsize=(5, 9))
    im = ax.imshow(d_matrix[sort_idx], cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, aspect='auto', interpolation='nearest')

    for r, feat in enumerate(feat_sorted):
        orig_i = FEATURE_COLS.index(feat)
        for c, grp in enumerate(groups_with_data):
            p = p_matrix[orig_i, c]
            s = stars(p) if not np.isnan(p) else ''
            if s and s != 'ns':
                ax.text(c, r, s, ha='center', va='center',
                        fontsize=10, color='black', fontweight='bold')

    # Column labels include N
    col_labels = []
    for grp in groups_with_data:
        sub = sex_spec[sex_spec['Group'] == grp]
        nm = (sub['Sex']=='Male').sum()
        nf = (sub['Sex']=='Female').sum()
        col_labels.append(f"{grp}\n(M={nm}, F={nf})")

    ax.set_xticks(range(n_grps))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(feat_sorted)))
    ax.set_yticklabels(feat_sorted, fontsize=8.5)
    ax.set_title("Sex effect within each group\nCohen's d  (Male − Female)\n"
                 "* FDR<0.05  ** <0.01  *** <0.001",
                 fontsize=10, pad=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Cohen's d", fontsize=9)
    cbar.ax.axhline(0, color='black', lw=0.8)

    plt.tight_layout()
    plt.savefig(LAYER5_DIR / 'sex_heatmap_bygroup.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("  Saved sex_heatmap_bygroup.png")

# Plot 5-3 — Boxplots for FDR-significant features (overall analysis)
#
# For each significant feature: split each group's cells by sex and show
# 6 boxes (TAV-NA M/F, TAV-ATAA M/F, BAV-ATAA M/F).
# This gives the fullest picture of where the sex difference sits in the data.
# We use cell-level data here (not specimen means) for visual richness, but
# note in the title that statistical testing was specimen-level.

sig_sex_feats = sex_overall[sex_overall['sig']].index.tolist()

if not sig_sex_feats:
    # If nothing significant overall, use top 4 by |d| for illustration
    sig_sex_feats = (sex_overall.dropna(subset=['Cohens_d'])
                     .assign(_abs=lambda x: x['Cohens_d'].abs())
                     .nlargest(4, '_abs').index.tolist())
    print(f"  No FDR-significant features overall — plotting top 4 by |d|")

SEX_COLORS = {'Male': '#1565C0', 'Female': '#AD1457'}
GROUP_X = {'TAV-NA': [0, 1], 'TAV-ATAA': [3, 4], 'BAV-ATAA': [6, 7]}
SEXES   = ['Male', 'Female']

for feat in sig_sex_feats:
    fig, ax = plt.subplots(figsize=(9, 5))

    all_vals_plot = []
    boxes_data, positions_plot, colors_list = [], [], []

    for grp in GROUPS:
        for s_idx, sex in enumerate(SEXES):
            vals = df_nc[(df_nc['Group'] == grp) & (df_nc['Sex'] == sex)][feat].dropna().values
            xpos = GROUP_X[grp][s_idx]
            boxes_data.append(vals)
            positions_plot.append(xpos)
            colors_list.append(SEX_COLORS[sex])
            all_vals_plot.extend(vals)

    bp = ax.boxplot(boxes_data, positions=positions_plot, widths=0.6,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=2,
                                    markerfacecolor='none', alpha=0.3),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.1),
                    capprops=dict(linewidth=1.1),
                    boxprops=dict(linewidth=1.1))

    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Mean ± SD diamonds
    for xpos, vals, color in zip(positions_plot, boxes_data, colors_list):
        if len(vals) > 0:
            m, s = np.mean(vals), np.std(vals)
            ax.plot(xpos, m, marker='D', color='black', markersize=5, zorder=5)
            ax.plot([xpos, xpos], [m-s, m+s], color='black', lw=1.3, zorder=4)

    # Group labels at midpoint of each group's two boxes
    for grp in GROUPS:
        mid = np.mean(GROUP_X[grp])
        ax.text(mid, ax.get_ylim()[0] - (max(all_vals_plot) - min(all_vals_plot)) * 0.08,
                grp, ha='center', va='top', fontsize=9, fontweight='bold',
                color=GROUP_COLORS[grp])

    # Add specimen-level Cohen's d annotation per group (from within-group analysis)
    y_annot = np.percentile(all_vals_plot, 97) if all_vals_plot else 1
    for grp in GROUPS:
        if grp not in sex_bygroup or feat not in sex_bygroup[grp].index:
            continue
        r = sex_bygroup[grp].loc[feat]
        if np.isnan(r['Cohens_d']):
            continue
        direction = '♂>♀' if r['Cohens_d'] > 0 else '♀>♂'
        s = stars(r['p_FDR'])
        mid = np.mean(GROUP_X[grp])
        ax.text(mid, y_annot, f"d={r['Cohens_d']:+.2f}\n{direction} {s}",
                ha='center', va='bottom', fontsize=7.5, color='#333333',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#aaaaaa', alpha=0.8, lw=0.7))

    ax.set_xticks([0.5, 3.5, 6.5])
    ax.set_xticklabels(['', '', ''])
    ax.set_xlim(-0.7, 8.2)
    ax.set_ylabel(feat, fontsize=9)
    ax.set_title(f"{feat.replace('_', ' ')} — by Sex × Group\n"
                 f"(stats = specimen-level Cohen's d)",
                 fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.25, lw=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    sex_handles = [_mpatches.Patch(color=SEX_COLORS[s], label=s, alpha=0.8)
                   for s in SEXES]
    ax.legend(handles=sex_handles, loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(LAYER5_DIR / f'{feat}_sex_boxplot.png', dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved {feat}_sex_boxplot.png")

# ── Summary table ─────────────────────────────────────────────────────────────
summary_sex = pd.DataFrame(index=FEATURE_COLS)
summary_sex.index.name = 'Feature'
summary_sex['Overall_d']   = sex_overall['Cohens_d']
summary_sex['Overall_FDR'] = sex_overall['p_FDR']
summary_sex['Overall_sig'] = sex_overall['sig']
summary_sex['AgeCorr_d']   = sex_agecorr['Cohens_d']
summary_sex['AgeCorr_FDR'] = sex_agecorr['p_FDR']
summary_sex['AgeCorr_sig'] = sex_agecorr['sig']
for grp in GROUPS:
    if grp not in sex_bygroup:
        continue
    key = grp.replace('-','_')
    summary_sex[f'{key}_d']   = sex_bygroup[grp]['Cohens_d']
    summary_sex[f'{key}_FDR'] = sex_bygroup[grp]['p_FDR']
    summary_sex[f'{key}_sig'] = sex_bygroup[grp]['sig']

summary_sex.to_csv(LAYER5_DIR / 'sex_summary.csv')
print("\n  Saved sex_summary.csv")
print(f"\nLayer 5 outputs saved to {LAYER5_DIR}/")

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — Collagen effect (paired analysis)
#
# Experimental context:
#   After dissection, each patient's SMCs were split into two conditions:
#     - NoCollagen : cells cultured in standard medium (baseline)
#     - Collagen   : cells cultured on a collagen matrix (mimics the ECM)
#   This is an IN VITRO perturbation on cells from the same patient,
#   not a different patient population. The collagen treatment is therefore
#   a WITHIN-SPECIMEN factor.
#
# Why collagen matters:
#   In the diseased aortic wall, the extracellular matrix (ECM) composition
#   changes dramatically — collagen accumulates, elastic fibres degrade.
#   Culturing cells on collagen tests whether providing a collagen-rich
#   substrate can rescue or worsen the aneurysmal phenotype. If collagen
#   shifts aneurysm cells back toward the healthy morphology, it suggests
#   the ECM environment drives part of the phenotype (not just genetics).
#   If it worsens it, it suggests collagen signalling amplifies disease.
#
# Critical design point — PAIRED, not independent:
#   24 specimens have BOTH conditions (same patient, two cultures).
#   3 specimens have only one condition (02Asc-0021, 03ASC-0054 = NoCollagen only;
#   01C-0113 = Collagen+Unknown). These 3 are excluded from the paired analysis.
#
#   Using a paired test instead of an independent-groups test is correct here
#   because the two measurements (NoCollagen, Collagen) share the same patient
#   background (age, sex, genetics, disease severity). By differencing within
#   each specimen first, we remove all between-patient variability and test
#   only the collagen effect. This dramatically increases power: instead of
#   needing the collagen effect to overcome both the between-patient noise AND
#   the within-patient collagen signal, the paired test only needs to see a
#   consistent within-patient shift.
#
# Three levels:
#   6a — Overall collagen effect (all 3 groups combined, N=24 paired specimens)
#        Does collagen shift morphology at all, regardless of disease status?
#
#   6b — Within-group collagen effect (TAV-NA, TAV-ATAA, BAV-ATAA separately)
#        Is the collagen response the same in healthy and aneurysmal cells?
#        The key question: does collagen RESCUE aneurysm cells toward healthy?
#
#   6c — Rescue analysis: does collagen close the disease gap?
#        Compare the TAV-NA vs TAV-ATAA difference in NoCollagen condition
#        against the TAV-NA vs TAV-ATAA difference in Collagen condition.
#        If the gap shrinks in the Collagen condition → partial rescue.
#        If it grows → collagen amplifies disease. If unchanged → no effect.
#
# Statistical test:
#   Paired Wilcoxon signed-rank test (Wilcoxon sr) for non-normal differences,
#   paired T-test for normal differences (Shapiro-Wilk on the DIFFERENCES,
#   not on the raw values — this is the correct normality check for paired data).
#   Effect size: Cohen's d_z = mean(difference) / SD(difference).
#   Positive d_z → Collagen > NoCollagen.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LAYER 6 — Collagen effect (paired analysis)")
print("="*60)

from scipy.stats import wilcoxon, ttest_rel

LAYER6_DIR = OUT_DIR / 'layer6_collagen'
LAYER6_DIR.mkdir(exist_ok=True)

# ── 6.0 Build paired specimen-level table ─────────────────────────────────────
# For each specimen with BOTH conditions, compute specimen-mean per condition.
# Then compute the difference (Collagen − NoCollagen) per specimen.
# The difference is the unit of analysis for all paired tests.

df_coll = df[df['Collagen_Status'].isin(['Collagen', 'NoCollagen'])].copy()

# Specimen-mean per condition
spec_cond = (df_coll
             .groupby(['Specimen', 'Collagen_Status', 'Group'])
             [FEATURE_COLS]
             .mean()
             .reset_index())

# Pivot: one row per specimen, columns for each feature × condition
nc_means  = spec_cond[spec_cond['Collagen_Status'] == 'NoCollagen'].set_index('Specimen')
col_means = spec_cond[spec_cond['Collagen_Status'] == 'Collagen'].set_index('Specimen')

# Keep only specimens that have BOTH conditions
paired_specs = nc_means.index.intersection(col_means.index)
n_paired = len(paired_specs)
print(f"\n  Paired specimens (both conditions): {n_paired}")
print(f"  Excluded (one condition only): "
      f"{set(nc_means.index) ^ set(col_means.index)}")

nc_paired  = nc_means.loc[paired_specs]
col_paired = col_means.loc[paired_specs]
diff       = col_paired[FEATURE_COLS] - nc_paired[FEATURE_COLS]   # Collagen − NoCollagen
group_map  = nc_paired['Group']   # group label for each paired specimen

print(f"  Group breakdown of paired specimens:")
print(f"  {group_map.value_counts().to_dict()}")

# ── 6.1 Paired test helper ────────────────────────────────────────────────────
def paired_test(diffs):
    """
    Test whether the paired differences are consistently non-zero.
    Steps:
      1. Shapiro-Wilk on the DIFFERENCES (not the raw values) to check
         whether the distribution of within-patient changes is normal.
      2. If normal (p>0.05): paired T-test (most powerful option).
      3. If non-normal: Wilcoxon signed-rank test (non-parametric equivalent).
    Returns (p_value, test_name, mean_diff, sd_diff, cohen_dz).
    Cohen's d_z = mean(diff) / SD(diff) — the paired effect size.
    Positive = Collagen > NoCollagen.
    """
    d = np.array(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return np.nan, 'N/A', np.nan, np.nan, np.nan
    mean_d = np.mean(d)
    sd_d   = np.std(d, ddof=1)
    dz     = mean_d / sd_d if sd_d > 0 else np.nan
    if len(d) <= 5000:
        _, p_sw = shapiro(d)
        is_norm = p_sw > 0.05
    else:
        is_norm = False
    if is_norm:
        _, p = ttest_rel(d, np.zeros(len(d)))
        test = 'Paired T-test'
    else:
        try:
            _, p = wilcoxon(d, alternative='two-sided')
            test = 'Wilcoxon sr'
        except ValueError:
            _, p = ttest_rel(d, np.zeros(len(d)))
            test = 'Paired T-test (fallback)'
    return p, test, mean_d, sd_d, dz

# ── Layer 6a — Overall collagen effect ───────────────────────────────────────
print("\nLayer 6a — Overall collagen effect (N=24 paired specimens)")

overall_rows = []
for feat in FEATURE_COLS:
    p, test, mean_d, sd_d, dz = paired_test(diff[feat].values)
    overall_rows.append({'Feature': feat, 'Cohen_dz': dz, 'Mean_diff': mean_d,
                         'SD_diff': sd_d, 'p_raw': p, 'Test': test})

coll_overall = pd.DataFrame(overall_rows)
coll_overall['p_FDR'] = fdr_bh(coll_overall['p_raw'].fillna(1).values)
coll_overall['sig']   = coll_overall['p_FDR'] < 0.05
coll_overall.to_csv(LAYER6_DIR / 'collagen_overall.csv', index=False)

n_sig_ov = coll_overall['sig'].sum()
print(f"  FDR-significant: {n_sig_ov} features")
print("  All features by |d_z| (positive = Collagen > NoCollagen):")
for _, r in coll_overall.sort_values('Cohen_dz', key=abs, ascending=False).iterrows():
    sig_flag  = '*' if r['sig'] else ' '
    direction = 'Col↑' if r['Cohen_dz'] > 0 else 'Col↓'
    print(f"    {sig_flag} {r['Feature']:40s}  d_z={r['Cohen_dz']:+.3f}  "
          f"{direction}  FDR={r['p_FDR']:.3f}")

# ── Layer 6b — Within-group collagen effect ───────────────────────────────────
print("\nLayer 6b — Within-group collagen effect (paired, per group)")

coll_bygroup = {}
for grp in GROUPS:
    grp_specs = group_map[group_map == grp].index
    n = len(grp_specs)
    print(f"\n  {grp}  (N={n} paired specimens)")
    if n < 3:
        print(f"    Insufficient N — skipping")
        continue
    rows = []
    for feat in FEATURE_COLS:
        d = diff.loc[grp_specs, feat].values
        p, test, mean_d, sd_d, dz = paired_test(d)
        rows.append({'Feature': feat, 'Cohen_dz': dz, 'Mean_diff': mean_d,
                     'SD_diff': sd_d, 'p_raw': p, 'Test': test, 'N': n})
    res = pd.DataFrame(rows)
    res['p_FDR'] = fdr_bh(res['p_raw'].fillna(1).values)
    res['sig']   = res['p_FDR'] < 0.05
    res = res.set_index('Feature')
    coll_bygroup[grp] = res
    res.to_csv(LAYER6_DIR / f'collagen_{grp.replace("-","_")}.csv')
    n_sig = res['sig'].sum()
    print(f"  FDR-significant: {n_sig} features")
    top5 = res.dropna(subset=['Cohen_dz']).sort_values('Cohen_dz', key=abs, ascending=False).head(5)
    for feat, r in top5.iterrows():
        sig_flag  = '*' if r['sig'] else ' '
        direction = 'Col↑' if r['Cohen_dz'] > 0 else 'Col↓'
        print(f"    {sig_flag} {feat:40s}  d_z={r['Cohen_dz']:+.3f}  "
              f"{direction}  FDR={r['p_FDR']:.3f}")

# ── Layer 6c — Rescue analysis ────────────────────────────────────────────────
# Does collagen narrow or widen the TAV-NA vs TAV-ATAA morphological gap?
# Method:
#   For each feature, compute the TAV-NA vs TAV-ATAA pairwise Cohen's d
#   separately in the NoCollagen condition and in the Collagen condition
#   (using specimen means, independent-group comparison between the two groups).
#   Then compute: rescue_index = |d_NoCollagen| - |d_Collagen|
#   Positive rescue_index → gap is smaller in Collagen condition → rescue.
#   Negative rescue_index → gap is larger in Collagen condition → amplification.
# This uses independent-group comparison (not paired) because we are now
# comparing TAV-NA specimens to TAV-ATAA specimens, not the same specimen
# before and after. Paired is only valid for the within-specimen comparison.

print("\nLayer 6c — Rescue analysis: does collagen narrow the TAV-NA vs TAV-ATAA gap?")

rescue_rows = []
for feat in FEATURE_COLS:
    for cond in ['NoCollagen', 'Collagen']:
        sub = spec_cond[spec_cond['Collagen_Status'] == cond].set_index('Specimen')
        tav_na  = sub[sub['Group'] == 'TAV-NA'][feat].dropna().values
        tav_ataa = sub[sub['Group'] == 'TAV-ATAA'][feat].dropna().values
        if len(tav_na) < 2 or len(tav_ataa) < 2:
            rescue_rows.append({'Feature': feat, 'Condition': cond,
                                 'Cohens_d': np.nan, 'N_NA': len(tav_na),
                                 'N_ATAA': len(tav_ataa)})
            continue
        d = cohens_d(tav_na, tav_ataa)   # positive = TAV-ATAA > TAV-NA
        rescue_rows.append({'Feature': feat, 'Condition': cond,
                             'Cohens_d': d, 'N_NA': len(tav_na),
                             'N_ATAA': len(tav_ataa)})

rescue_df = pd.DataFrame(rescue_rows)
rescue_pivot = rescue_df.pivot(index='Feature', columns='Condition', values='Cohens_d')
rescue_pivot['rescue_index'] = rescue_pivot['NoCollagen'].abs() - rescue_pivot['Collagen'].abs()
# Positive rescue_index = disease gap SHRINKS in collagen condition = rescue
rescue_pivot.to_csv(LAYER6_DIR / 'rescue_analysis.csv')

print("\n  Rescue index = |d_NoCollagen| - |d_Collagen|")
print("  Positive = gap shrinks (rescue)  |  Negative = gap grows (amplification)")
print(f"\n  {'Feature':40s}  {'d_NoColl':>9}  {'d_Coll':>8}  {'Rescue':>8}")
for feat, r in rescue_pivot.sort_values('rescue_index', ascending=False).iterrows():
    nc_d = r['NoCollagen']
    co_d = r['Collagen']
    ri   = r['rescue_index']
    marker = '→ RESCUE' if ri > 0.2 else ('→ AMPLIFY' if ri < -0.2 else '→ neutral')
    print(f"  {feat:40s}  {nc_d:+9.3f}  {co_d:+8.3f}  {ri:+8.3f}  {marker}")

# ── 6. Plots ──────────────────────────────────────────────────────────────────

# Plot 6-1 — Paired difference forest (overall, all groups)
# One row per feature. Bar = mean paired difference (Collagen − NoCollagen)
# normalised by SD of differences (= Cohen's d_z).
# Colour: green = collagen increases feature, purple = collagen decreases.
# Solid = FDR<0.05. Reference lines at ±0.5 and ±0.8.

feat_order_col = list(reversed(FEATURE_COLS))
dz_vals  = coll_overall.set_index('Feature').loc[feat_order_col, 'Cohen_dz'].values
sig_vals = coll_overall.set_index('Feature').loc[feat_order_col, 'sig'].values
fdr_vals = coll_overall.set_index('Feature').loc[feat_order_col, 'p_FDR'].values

fig, ax = plt.subplots(figsize=(7, 8))
y = np.arange(len(feat_order_col))

for i, (feat, dz, sig, fdr) in enumerate(zip(feat_order_col, dz_vals, sig_vals, fdr_vals)):
    if np.isnan(dz):
        continue
    color = '#2E7D32' if dz > 0 else '#6A1B9A'   # green = Col↑, purple = Col↓
    alpha = 1.0 if sig else 0.35
    ax.barh(i, dz, color=color, alpha=alpha, height=0.6, zorder=2)
    s = stars(fdr)
    if s and s != 'ns':
        xpos = dz + (0.04 if dz >= 0 else -0.04)
        ax.text(xpos, i, s, va='center',
                ha='left' if dz >= 0 else 'right',
                fontsize=8, color=color, fontweight='bold')

ax.axvline(0,    color='#555555', lw=0.8, ls='--', zorder=1)
ax.axvline( 0.5, color='#aaaaaa', lw=0.5, ls=':',  zorder=1)
ax.axvline(-0.5, color='#aaaaaa', lw=0.5, ls=':',  zorder=1)
ax.axvline( 0.8, color='#888888', lw=0.5, ls=':',  zorder=1)
ax.axvline(-0.8, color='#888888', lw=0.5, ls=':',  zorder=1)
ax.set_yticks(y)
ax.set_yticklabels(feat_order_col, fontsize=8)
ax.set_xlabel("Cohen's d_z  (Collagen − NoCollagen, paired)", fontsize=10)
ax.set_title(f"Collagen effect on morphology\n"
             f"N={n_paired} paired specimens (all groups combined)\n"
             f"Solid = FDR<0.05 | dotted = |d|=0.5 (medium), 0.8 (large)",
             fontsize=9, pad=8)
ax.grid(axis='x', color='#eeeeee', lw=0.6, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for feats_grp, col in [
    ([f for f in feat_order_col if f.startswith('Actin')],   '#E3F2FD'),
    ([f for f in feat_order_col if f.startswith('Mito')],    '#E8F5E9'),
    ([f for f in feat_order_col if f.startswith('Nucleus')], '#FFF8E1'),
]:
    idxs = [feat_order_col.index(f) for f in feats_grp]
    if idxs:
        ax.axhspan(min(idxs)-0.5, max(idxs)+0.5, color=col, alpha=0.35, zorder=0)

plt.tight_layout()
plt.savefig(LAYER6_DIR / 'collagen_forest_overall.png', dpi=180, bbox_inches='tight')
plt.close()
print("\n  Saved collagen_forest_overall.png")

# Plot 6-2 — Within-group collagen heatmap
# Rows = features (sorted by |mean d_z|), columns = groups.
# Colour = Cohen's d_z. Stars = FDR<0.05.
# This shows whether the collagen response is uniform or group-specific.

groups_coll = [g for g in GROUPS if g in coll_bygroup]
if groups_coll:
    dz_mat = np.full((len(FEATURE_COLS), len(groups_coll)), np.nan)
    p_mat  = np.full((len(FEATURE_COLS), len(groups_coll)), np.nan)
    for j, grp in enumerate(groups_coll):
        res = coll_bygroup[grp]
        for i, feat in enumerate(FEATURE_COLS):
            if feat in res.index:
                dz_mat[i, j] = res.loc[feat, 'Cohen_dz']
                p_mat[i, j]  = res.loc[feat, 'p_FDR']

    sort_idx_c   = np.argsort(np.nanmean(np.abs(dz_mat), axis=1))[::-1]
    feat_sorted_c = [FEATURE_COLS[i] for i in sort_idx_c]

    vmax_c = max(0.8, float(np.nanmax(np.abs(dz_mat))))
    fig, ax = plt.subplots(figsize=(5, 9))
    im = ax.imshow(dz_mat[sort_idx_c], cmap='PiYG',
                   vmin=-vmax_c, vmax=vmax_c, aspect='auto', interpolation='nearest')

    for r, feat in enumerate(feat_sorted_c):
        orig_i = FEATURE_COLS.index(feat)
        for c in range(len(groups_coll)):
            p = p_mat[orig_i, c]
            s = stars(p) if not np.isnan(p) else ''
            if s and s != 'ns':
                ax.text(c, r, s, ha='center', va='center',
                        fontsize=10, color='black', fontweight='bold')

    col_labels = [f"{g}\n(N={len(group_map[group_map==g])})" for g in groups_coll]
    ax.set_xticks(range(len(groups_coll)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(feat_sorted_c)))
    ax.set_yticklabels(feat_sorted_c, fontsize=8.5)
    ax.set_title("Collagen effect within each group\n"
                 "Cohen's d_z  (Collagen − NoCollagen, paired)\n"
                 "Green=Col↑  Purple=Col↓  * FDR<0.05",
                 fontsize=9, pad=8)
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Cohen's d_z", fontsize=9)
    cbar.ax.axhline(0, color='black', lw=0.8)
    plt.tight_layout()
    plt.savefig(LAYER6_DIR / 'collagen_heatmap_bygroup.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("  Saved collagen_heatmap_bygroup.png")

# Plot 6-3 — Rescue bubble chart
# x = Cohen's d in NoCollagen condition (TAV-NA vs TAV-ATAA gap without collagen)
# y = Cohen's d in Collagen condition   (TAV-NA vs TAV-ATAA gap with collagen)
# Points above the diagonal y=x → gap grew (amplification)
# Points below the diagonal y=x → gap shrank (rescue)
# Bubble size = |rescue_index|. Colour = rescue (green) or amplify (red).
# This is the most direct visual answer to the rescue question.

fig, ax = plt.subplots(figsize=(7, 7))

for feat, r in rescue_pivot.iterrows():
    nc_d = r['NoCollagen']
    co_d = r['Collagen']
    ri   = r['rescue_index']
    if np.isnan(nc_d) or np.isnan(co_d):
        continue
    color = '#2E7D32' if ri > 0 else '#C62828'
    ax.scatter(nc_d, co_d, s=max(20, abs(ri)*300),
               color=color, alpha=0.7, edgecolors='white', lw=0.5, zorder=3)
    ax.annotate(feat.replace('_', ' '), (nc_d, co_d),
                textcoords='offset points', xytext=(5, 3),
                fontsize=6.5, color='#333333')

# Diagonal y = x: gap unchanged
lim = max(abs(rescue_pivot[['NoCollagen','Collagen']].values.flatten()))
lim = lim * 1.15
ax.plot([-lim, lim], [-lim, lim], color='#aaaaaa', lw=1, ls='--', zorder=1,
        label='No change')
ax.axhline(0, color='#dddddd', lw=0.6, zorder=0)
ax.axvline(0, color='#dddddd', lw=0.6, zorder=0)
ax.fill_between([-lim, lim], [-lim, lim], [lim, lim],
                color='#FFEBEE', alpha=0.2, zorder=0, label='Amplification zone')
ax.fill_between([-lim, lim], [-lim, lim], [-lim, -lim],
                color='#E8F5E9', alpha=0.2, zorder=0, label='Rescue zone')

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("Cohen's d  (TAV-NA vs TAV-ATAA) — NoCollagen condition", fontsize=10)
ax.set_ylabel("Cohen's d  (TAV-NA vs TAV-ATAA) — Collagen condition", fontsize=10)
ax.set_title("Rescue analysis: does collagen narrow the disease gap?\n"
             "Points BELOW diagonal → gap shrank (rescue)\n"
             "Points ABOVE diagonal → gap grew (amplification)",
             fontsize=9, pad=8)
ax.legend(fontsize=8, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(LAYER6_DIR / 'rescue_bubble.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved rescue_bubble.png")

# Plot 6-4 — Paired line plots for FDR-significant features (or top 4)
# Each line connects the specimen mean in NoCollagen to its mean in Collagen.
# One panel per group. Lines slope up or down per specimen.
# This is the clearest visualisation of a paired result: you see each
# patient's individual response, not just group means.

sig_coll_feats = coll_overall[coll_overall['sig']]['Feature'].tolist()
if not sig_coll_feats:
    sig_coll_feats = (coll_overall.dropna(subset=['Cohen_dz'])
                      .assign(_abs=lambda x: x['Cohen_dz'].abs())
                      .nlargest(4, '_abs')['Feature'].tolist())
    print(f"  No FDR-significant collagen features — plotting top 4 by |d_z|")

COLL_COLORS = {'NoCollagen': '#546E7A', 'Collagen': '#00897B'}

for feat in sig_coll_feats:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), sharey=False)
    for ax, grp in zip(axes, GROUPS):
        grp_specs = group_map[group_map == grp].index if grp in group_map.values else []
        if len(grp_specs) == 0:
            ax.set_visible(False)
            continue

        nc_vals  = nc_paired.loc[nc_paired.index.intersection(grp_specs), feat]
        col_vals = col_paired.loc[col_paired.index.intersection(grp_specs), feat]
        common   = nc_vals.index.intersection(col_vals.index)

        for spec in common:
            v_nc  = nc_vals.loc[spec]
            v_col = col_vals.loc[spec]
            if np.isnan(v_nc) or np.isnan(v_col):
                continue
            color = '#2E7D32' if v_col > v_nc else '#6A1B9A'
            ax.plot([0, 1], [v_nc, v_col], color=color, alpha=0.55,
                    lw=1.5, marker='o', markersize=4, zorder=3)

        # Group means as thick line
        mean_nc  = nc_vals.loc[common].mean()
        mean_col = col_vals.loc[common].mean()
        ax.plot([0, 1], [mean_nc, mean_col], color='black',
                lw=2.5, marker='D', markersize=7, zorder=5,
                label='Group mean')

        if grp in coll_bygroup and feat in coll_bygroup[grp].index:
            r = coll_bygroup[grp].loc[feat]
            dz_str  = f"d_z={r['Cohen_dz']:+.2f}"
            sig_str = stars(r['p_FDR'])
            ax.set_title(f"{grp}\nN={len(common)}  {dz_str}  {sig_str}",
                         fontsize=9, fontweight='bold')
        else:
            ax.set_title(f"{grp}\nN={len(common)}", fontsize=9, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['NoCollagen', 'Collagen'], fontsize=9)
        ax.set_ylabel(feat.replace('_', ' '), fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.25)

    fig.suptitle(f"{feat.replace('_', ' ')} — paired collagen response\n"
                 f"Each line = one specimen  |  Black = group mean",
                 fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(LAYER6_DIR / f'{feat}_paired_lines.png', dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved {feat}_paired_lines.png")

# ── Summary table ─────────────────────────────────────────────────────────────
summary_coll = pd.DataFrame({'Feature': FEATURE_COLS}).set_index('Feature')
summary_coll['Overall_dz']  = coll_overall.set_index('Feature')['Cohen_dz']
summary_coll['Overall_FDR'] = coll_overall.set_index('Feature')['p_FDR']
summary_coll['Overall_sig'] = coll_overall.set_index('Feature')['sig']
for grp in GROUPS:
    if grp not in coll_bygroup:
        continue
    key = grp.replace('-','_')
    summary_coll[f'{key}_dz']  = coll_bygroup[grp]['Cohen_dz']
    summary_coll[f'{key}_FDR'] = coll_bygroup[grp]['p_FDR']
    summary_coll[f'{key}_sig'] = coll_bygroup[grp]['sig']
summary_coll['Rescue_index'] = rescue_pivot['rescue_index']
summary_coll.to_csv(LAYER6_DIR / 'collagen_summary.csv')
print("\n  Saved collagen_summary.csv")
print(f"\nLayer 6 outputs saved to {LAYER6_DIR}/")
print("\nDone.")
