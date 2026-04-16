#!/usr/bin/env python3
"""
Final Comprehensive Statistical Analysis  —  All Layers
=========================================================
Data:
  outputs/Advanced_Features_Raw_Final.csv
    (Healthy n=110, TAA n=111, BAV n=50 | ±Collagen | Male/Female)
  Book1 (2).xlsx  →  specimen information
    (Hypertension status, Aorta diameter mm)

Unit of analysis: CELL-LEVEL (each cell is one observation).
  Layer 3 additionally uses Subject as a random intercept to account for
  the clustering of multiple cells per patient.

  Layer 1  — Disease effect  (Shapiro-Wilk → KW + BH-FDR + Dunn)
  Layer 2  — Age-residualized disease effect
  Layer 3  — Linear Mixed Model  (Disease + Collagen + Age + Sex + interaction,
                                   Subject = random intercept)
  Layer 4  — Collagen rescue effect  (Mann-Whitney per disease group, cell-level)
  Layer 5  — Sex effect  (overall + per disease group)
  Layer 6  — Hypertension effect  (Yes vs No, TAA + BAV patients)
  Layer 7  — Aorta diameter correlations  (Spearman, TAA + BAV patients)
  Summary  — PCA (by disease / by sex / by collagen) + feature heatmap

Output: outputs/statistical_analysis/
"""

import re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro, kruskal, mannwhitneyu, ttest_ind, spearmanr, wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print("WARNING: scikit_posthocs not installed  →  Dunn's test skipped.\n"
          "         pip install scikit-posthocs")

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed  →  LMM + BH-FDR skipped.\n"
          "         pip install statsmodels")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
FEATURES_FILE = 'outputs/Advanced_Features_Raw_Final.csv'
METADATA_FILE = 'Book1 (2).xlsx'
OUTPUT_DIR    = Path('outputs/statistical_analysis')
ALPHA         = 0.05

DISEASE_ORDER  = ['Healthy', 'TAA', 'BAV']
DISEASE_LABELS = {'Healthy': 'Non-Aneurysmal', 'TAA': 'TAA', 'BAV': 'BAV'}
DISEASE_COLORS = {'Healthy': '#4C9BE8', 'TAA': '#E74C3C', 'BAV': '#F39C12'}

COLLAGEN_COLORS = {'NoCollagen': '#AAAAAA', 'Collagen': '#9B59B6'}
SEX_COLORS      = {'Male': '#3498DB', 'Female': '#E91E8C'}
HYPERT_COLORS   = {'Yes': '#C0392B', 'No': '#27AE60'}

ORGANELLE_COLORS = {
    'Nucleus': '#4C9BE8', 'Actin': '#E74C3C', 'Mitochondria': '#2ECC71'
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_organelle(feat):
    fl = feat.lower()
    if 'nucleus' in fl: return 'Nucleus'
    if 'actin'   in fl: return 'Actin'
    if 'mito'    in fl: return 'Mitochondria'
    return 'Other'

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def cohens_d(a, b):
    s = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(b) - np.mean(a)) / s if s > 0 else 0.0

def bh_fdr(pvals):
    if HAS_STATSMODELS:
        _, q, _, _ = multipletests(np.nan_to_num(pvals, nan=1.0), method='fdr_bh')
        return q
    return np.array(pvals)

def is_normal(data):
    if len(data) < 3: return False
    _, p = shapiro(data)
    return p > ALPHA

def adaptive_test(d1, d2):
    """Shapiro-Wilk → Welch t-test if both normal, else Mann-Whitney U."""
    if is_normal(d1) and is_normal(d2):
        _, p = ttest_ind(d1, d2, equal_var=False)
        return p, 'T-test'
    _, p = mannwhitneyu(d1, d2, alternative='two-sided')
    return p, 'Mann-Whitney'

def norm_specimen_id(s):
    """Normalise specimen ID: '01Asc-0180' or '02Asc-0017/...' → '01asc-180'."""
    m = re.search(r'(0[123][A-Za-z]+)-?0*(\d+)', str(s), re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()}-{m.group(2)}"
    return None

def save(fig, name):
    fig.savefig(OUTPUT_DIR / name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    print("=" * 65)
    print("LOADING DATA")
    print("=" * 65)

    # ── Features ──────────────────────────────────────────────────────────────
    df = pd.read_csv(FEATURES_FILE)
    df = df[df['Disease'].isin(DISEASE_ORDER)].copy()

    df['Subject']     = df['CellName'].apply(norm_specimen_id)
    df['Disease_TAA'] = (df['Disease'] == 'TAA').astype(int)
    df['Disease_BAV'] = (df['Disease'] == 'BAV').astype(int)
    df['Coll_bin']    = (df['Collagen_Status'] == 'Collagen').astype(int)
    df['Sex_bin']     = (df['Gender'] == 'Male').astype(int)

    feat_cols = [c for c in df.columns
                 if c.startswith(('Actin_', 'Mito_', 'Nucleus_'))]
    df[feat_cols] = (df[feat_cols]
                     .replace([np.inf, -np.inf], np.nan)
                     .fillna(0))

    # ── Metadata: hypertension + aorta diameter ────────────────────────────
    try:
        meta = pd.read_excel(METADATA_FILE, sheet_name='specimen information')
        meta['Subject']     = meta['Aortic Specimen ID'].apply(norm_specimen_id)
        meta['Hypertn']     = meta['hypertn'].map({'Yes': 'Yes', 'No': 'No'})
        meta['Hypertn_bin'] = meta['hypertn'].map({'Yes': 1, 'No': 0})
        meta['AortaDiam_mm']= pd.to_numeric(meta['trtznlrgdiammeas'], errors='coerce')
        meta_sub = meta[['Subject', 'Hypertn', 'Hypertn_bin',
                          'AortaDiam_mm']].dropna(subset=['Subject'])
        df = df.merge(meta_sub, on='Subject', how='left')
        n_hyp  = df['Hypertn'].notna().sum()
        n_diam = df['AortaDiam_mm'].notna().sum()
        print(f"  Hypertension data matched: {n_hyp} cells")
        print(f"  Aorta diameter data matched: {n_diam} cells")
    except Exception as e:
        print(f"  WARNING: Could not load metadata ({e})")
        df['Hypertn']      = np.nan
        df['Hypertn_bin']  = np.nan
        df['AortaDiam_mm'] = np.nan

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nTotal cells: {len(df)}")
    for d in DISEASE_ORDER:
        sub = df[df['Disease'] == d]
        nc  = (sub['Collagen_Status'] == 'NoCollagen').sum()
        co  = (sub['Collagen_Status'] == 'Collagen').sum()
        m   = (sub['Gender'] == 'Male').sum()
        f   = (sub['Gender'] == 'Female').sum()
        hy  = (sub.get('Hypertn', '') == 'Yes').sum()
        print(f"  {d:8s}: {len(sub):3d} cells  "
              f"(NoCol={nc}, +Col={co} | M={m}, F={f} | HyperTN={hy})")
    print(f"Features: {len(feat_cols)}")

    return df, feat_cols

# ══════════════════════════════════════════════════════════════════════════════
# SHARED PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _plot_sw_summary(sw, fname, layer_label):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{layer_label} — Shapiro-Wilk Normality Test',
                 fontsize=14, fontweight='bold')
    for ax, col, title, palette in [
        (axes[0], 'Organelle', 'By Organelle', ORGANELLE_COLORS),
        (axes[1], 'Disease',   'By Disease',   DISEASE_COLORS),
    ]:
        order = list(palette)
        summary = (sw.groupby(col)['Normal']
                   .apply(lambda x: (~x).mean() * 100)
                   .reindex(order).reset_index())
        summary.columns = [col, 'Pct']
        colors = [palette.get(k, '#AAA') for k in summary[col]]
        bars   = ax.bar(summary[col], summary['Pct'],
                        color=colors, edgecolor='black', lw=0.8)
        ax.axhline(50, color='red', ls='--', alpha=0.5, label='50%')
        ax.set_ylim(0, 112)
        ax.set_ylabel('% Non-normal', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, summary['Pct']):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1.5,
                        f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    save(fig, fname)

def _plot_kw_pairwise(kw, fname):
    """Grouped bar chart: -log10(BH-FDR q) for each pairwise comparison per feature.
    Shows Non-A vs TAA, Non-A vs BAV, and TAA vs BAV side by side."""
    pairs = [
        ('Healthy', 'TAA', 'Non-A vs TAA',  '#E74C3C'),
        ('Healthy', 'BAV', 'Non-A vs BAV',  '#F39C12'),
        ('TAA',     'BAV', 'TAA vs BAV',    '#9B59B6'),
    ]
    # Sort features by Non-A vs TAA q-value (most significant first)
    kw = kw.copy()
    for a, b, _, __ in pairs:
        qkey = f'q_{a}_{b}'
        if qkey not in kw.columns:
            kw[qkey] = 1.0
    kw = kw.sort_values('q_Healthy_TAA')

    n = len(kw)
    bar_height = 0.25
    y = np.arange(n)
    thr = -np.log10(ALPHA)

    fig, ax = plt.subplots(figsize=(13, max(7, n * 0.55)))

    for i, (a, b, label, color) in enumerate(pairs):
        qkey = f'q_{a}_{b}'
        vals = -np.log10(kw[qkey].clip(1e-10).values)
        offset = (i - 1) * bar_height
        bars = ax.barh(y + offset, vals, height=bar_height,
                       color=color, alpha=0.82, label=label, edgecolor='white', lw=0.3)

    ax.axvline(thr, color='black', ls='--', lw=1.2, alpha=0.6, label=f'q=0.05 threshold')
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace('_', ' ') for f in kw['Feature']], fontsize=10)
    ax.set_xlabel('-log₁₀ (BH-FDR q-value, Mann-Whitney)', fontsize=12)
    ax.set_title('Layer 1 — Pairwise Disease Effect\n'
                 'Non-A vs TAA  |  Non-A vs BAV  |  TAA vs BAV  (cell-level)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    fig.text(0.98, 0.01, '* p<0.05   ** p<0.01   *** p<0.001   (BH-FDR, cell-level)',
             ha='right', fontsize=10, style='italic', color='#444')
    plt.tight_layout()
    save(fig, fname)


def _plot_kw_lollipop(kw, fname, title):
    sub = kw.copy()
    sub['-l10q'] = -np.log10(sub['BH_q'].clip(1e-10))
    sub = sub.sort_values('-l10q')
    colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in sub['Organelle']]

    fig, ax = plt.subplots(figsize=(11, max(6, len(sub) * 0.45)))
    ax.barh(range(len(sub)), sub['-l10q'],
            color=colors, edgecolor='white', lw=0.4)
    thr = -np.log10(ALPHA)
    ax.axvline(thr, color='red', ls='--', lw=1.5)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub['Feature'], fontsize=11)
    ax.set_xlabel('-log₁₀ (BH-FDR q-value)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)

    # Asterisks at bar tips for significant features
    xlim_max = sub['-l10q'].max() * 1.18
    ax.set_xlim(0, xlim_max)
    pad = xlim_max * 0.02
    for i, (_, row) in enumerate(sub.iterrows()):
        if row['Significant']:
            ax.text(row['-l10q'] + pad, i, sig_stars(row['BH_q']),
                    va='center', ha='left', fontsize=11, fontweight='bold')

    fig.text(0.98, 0.01, '* p<0.05   ** p<0.01   *** p<0.001   (BH-FDR, cell-level)',
             ha='right', fontsize=10, style='italic', color='#444')
    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, fontsize=11, loc='lower right')
    plt.tight_layout()
    save(fig, fname)

def _plot_cohens_d_bars(df_feats, fname, title, g1_label, g2_label,
                        g1_vals_col, g2_vals_col, color):
    """Horizontal bar chart of Cohen's d with significance stars (cell-level)."""
    sub = df_feats.sort_values('Cohen_d')
    org_colors = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA')
                  for f in sub['Feature']]

    fig, ax = plt.subplots(figsize=(11, max(6, len(sub) * 0.45)))
    ax.barh(range(len(sub)), sub['Cohen_d'], color=org_colors, edgecolor='white', height=0.7)
    ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=11)
    ax.set_xlabel(f"Cohen's d  ({g2_label} vs {g1_label})", fontsize=13)
    ax.tick_params(axis='x', labelsize=11)
    n_sig = sub['Significant'].sum()
    ax.set_title(f'{title}\n({n_sig}/{len(sub)} significant, BH-FDR, cell-level)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    xlim = ax.get_xlim()
    pad  = (xlim[1] - xlim[0]) * 0.025
    for i, (_, row) in enumerate(sub.iterrows()):
        if row['Significant']:
            x = row['Cohen_d']
            ax.text(x + (pad if x >= 0 else -pad), i,
                    sig_stars(row['BH_q']),
                    va='center', ha='left' if x >= 0 else 'right',
                    fontsize=11, fontweight='bold')
    cur = ax.get_xlim()
    ax.set_xlim(cur[0] - abs(cur[0]) * 0.12, cur[1] + abs(cur[1]) * 0.12)

    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, fontsize=11, loc='lower right')
    fig.text(0.98, 0.01, '* p<0.05   ** p<0.01   *** p<0.001   (BH-FDR, cell-level)',
             ha='right', fontsize=10, style='italic', color='#444')
    plt.tight_layout()
    save(fig, fname)

def _plot_dunn_heatmaps(dunn, kw, fname, group_labels=None, max_features=12):
    if not dunn: return
    gls = group_labels or DISEASE_ORDER
    top_feats = (kw[kw['Feature'].isin(dunn)]
                 .sort_values('KW_stat', ascending=False)
                 .head(max_features)['Feature'].tolist())
    if not top_feats: return

    ncols = 3
    nrows = int(np.ceil(len(top_feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5.2, nrows*4.8))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(top_feats):
        ax  = axes[i]
        pm  = dunn[feat]
        lpm = -np.log10(pm.values.clip(1e-10))
        np.fill_diagonal(lpm, 0)
        short = [g[:5] for g in gls]
        sns.heatmap(lpm, annot=True, fmt='.1f', cmap='YlOrRd',
                    xticklabels=short, yticklabels=short, ax=ax,
                    cbar_kws={'shrink': 0.7},
                    annot_kws={'size': 9}, linewidths=0.5)
        for r in range(len(gls)):
            for c in range(len(gls)):
                if r != c and pm.values[r, c] < ALPHA:
                    ax.text(c+0.5, r+0.78, '*', ha='center',
                            fontsize=14, fontweight='bold')
        ax.set_title(feat.replace('_', ' '), fontsize=8.5, fontweight='bold')
        ax.tick_params(labelsize=8)

    for j in range(len(top_feats), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Dunn's Post-hoc (Bonferroni)  —  -log₁₀(p),  * = p < 0.05",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, fname)

def _plot_boxplots_by_group(df, feat_cols, kw, dunn, prefix,
                            group_col, group_order, group_colors,
                            group_labels=None, title_suffix=''):
    """Boxplots of significant features per organelle, grouped by group_col."""
    sig_feats = kw.loc[kw['Significant'], 'Feature'].tolist()
    if not sig_feats:
        print(f"  No significant features — boxplots skipped ({prefix})")
        return
    gls = group_labels or {g: g for g in group_order}

    for org in ['Nucleus', 'Actin', 'Mitochondria']:
        org_feats = [f for f in sig_feats
                     if get_organelle(f) == org][:6]
        if not org_feats: continue

        ncols = 3
        nrows = int(np.ceil(len(org_feats) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4.5))
        axes = np.array(axes).flatten()

        for i, feat in enumerate(org_feats):
            ax   = axes[i]
            data = [df.loc[df[group_col] == g, feat].dropna().values
                    for g in group_order]
            bp   = ax.boxplot(data, patch_artist=True,
                              medianprops=dict(color='black', lw=2),
                              flierprops=dict(marker='o', ms=3, alpha=0.5, ls='none'))
            for patch, g in zip(bp['boxes'], group_order):
                patch.set_facecolor(group_colors[g])
                patch.set_alpha(0.82)

            ax.set_xticks(range(1, len(group_order)+1))
            ax.set_xticklabels([gls.get(g, g) for g in group_order],
                               fontsize=8, rotation=10)
            ax.set_title(feat.replace('_', ' '), fontsize=9, fontweight='bold')
            ax.set_ylabel('Value', fontsize=8)
            ax.grid(axis='y', alpha=0.3)

            if dunn and feat in dunn:
                pm = dunn[feat]
                all_v = np.concatenate([d for d in data if len(d)])
                y_max = np.percentile(all_v, 97) * 1.12 if len(all_v) else 1
                n = len(group_order)
                pairs = [(a, b) for a in range(n) for b in range(a+1, n)]
                for offset_i, (a, b) in enumerate(pairs[:3]):
                    pv = pm.values[a, b]
                    if pv < ALPHA:
                        y = y_max * (1.0 + offset_i * 0.10)
                        ax.plot([a+1, a+1, b+1, b+1],
                                [y*0.97, y, y, y*0.97], 'k-', lw=1.2)
                        ax.text((a+b)/2+1, y*1.01, sig_stars(pv),
                                ha='center', fontsize=11, fontweight='bold')

        for j in range(len(org_feats), len(axes)):
            axes[j].set_visible(False)

        leg = [mpatches.Patch(color=group_colors[g], label=gls.get(g, g))
               for g in group_order]
        fig.legend(handles=leg, loc='lower center', ncol=len(group_order),
                   fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f'{org} — Significant Features  {title_suffix}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        save(fig, f'{prefix}_{org.lower()}.png')

def _plot_lmm_forest(lmm, fname, top_n=12):
    terms       = ['Disease (TAA)', 'Collagen', 'Disease × Collagen',
                   'Age', 'Sex (Male)']
    term_colors = ['#E74C3C', '#9B59B6', '#8E44AD', '#2ECC71', '#3498DB']

    fig, axes = plt.subplots(1, len(terms),
                             figsize=(len(terms)*4.5, max(6, top_n*0.52)))
    fig.suptitle('Layer 3 — LMM Fixed Effects\n'
                 '(Subject = random intercept, BH-FDR corrected,  * = q < 0.05)',
                 fontsize=13, fontweight='bold')

    for ax, term, color in zip(axes, terms, term_colors):
        sub = (lmm[lmm['Term'] == term]
               .assign(absCoef=lambda d: d['Coef'].abs())
               .sort_values('absCoef', ascending=False)
               .head(top_n))
        if sub.empty:
            ax.set_visible(False); continue

        y = range(len(sub))
        ax.errorbar(sub['Coef'], list(y),
                    xerr=[sub['Coef'] - sub['CI_low'],
                          sub['CI_high'] - sub['Coef']],
                    fmt='o', color=color, ecolor='#555',
                    capsize=4, ms=6, lw=1.2)
        ax.axvline(0, color='black', lw=0.9, ls='--')
        ax.set_yticks(list(y))
        ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']],
                           fontsize=8)
        ax.set_xlabel('Coefficient', fontsize=10)
        ax.set_title(term, fontsize=10, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3, axis='x')

        xlim = ax.get_xlim()
        for j, (_, row) in enumerate(sub.iterrows()):
            if row.get('Significant', False):
                ax.text(xlim[1], j, ' *', va='center',
                        fontsize=13, fontweight='bold')

    plt.tight_layout()
    save(fig, fname)

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — DISEASE EFFECT  (raw)
# ══════════════════════════════════════════════════════════════════════════════
def layer1_disease(df, feat_cols):
    print("\n" + "=" * 65)
    print("LAYER 1 — RAW DISEASE EFFECT  (Shapiro-Wilk → KW + Dunn)")
    print("=" * 65)

    # ── Cell-level analysis ───────────────────────────────────────────────────
    n_cells = {d: int((df['Disease'] == d).sum()) for d in DISEASE_ORDER}
    print(f"  Cell-level n: " +
          "  ".join(f"{d}={n}" for d, n in n_cells.items()))

    # Shapiro-Wilk on cell-level data
    sw_rows = []
    for feat in feat_cols:
        for dis in DISEASE_ORDER:
            vals = df.loc[df['Disease'] == dis, feat].dropna().values
            if len(vals) >= 3:
                _, p = shapiro(vals)
                sw_rows.append({'Feature': feat, 'Disease': dis,
                                'Organelle': get_organelle(feat),
                                'SW_p': p, 'Normal': p > ALPHA})
    sw = pd.DataFrame(sw_rows)
    pct_nn = (~sw['Normal']).mean() * 100
    print(f"  Shapiro-Wilk: {pct_nn:.1f}% non-normal → non-parametric tests")

    # Kruskal-Wallis on cell-level data
    kw_rows = []
    for feat in feat_cols:
        groups = [df.loc[df['Disease'] == d, feat].dropna().values
                  for d in DISEASE_ORDER]
        ok = [g for g in groups if len(g) >= 2]
        if len(ok) < 2: continue
        try:   stat, p = kruskal(*ok)
        except: stat, p = np.nan, np.nan

        pairs = [('Healthy', 'TAA'), ('Healthy', 'BAV'), ('TAA', 'BAV')]
        ds = {}
        for a, b in pairs:
            ga = df.loc[df['Disease'] == a, feat].dropna().values
            gb = df.loc[df['Disease'] == b, feat].dropna().values
            ds[f'd_{a}_{b}'] = cohens_d(ga, gb) if (len(ga) > 1 and len(gb) > 1) else np.nan
            if len(ga) >= 3 and len(gb) >= 3:
                try:
                    _, pp = mannwhitneyu(ga, gb, alternative='two-sided')
                except Exception:
                    pp = np.nan
            else:
                pp = np.nan
            ds[f'p_{a}_{b}'] = pp
        kw_rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                        'KW_stat': stat, 'KW_p': p, **ds})

    kw = pd.DataFrame(kw_rows)
    kw['BH_q']       = bh_fdr(kw['KW_p'].values)
    kw['Significant'] = kw['BH_q'] < ALPHA

    # BH-FDR for each pairwise comparison
    for a, b in [('Healthy', 'TAA'), ('Healthy', 'BAV'), ('TAA', 'BAV')]:
        pkey = f'p_{a}_{b}'
        if pkey in kw.columns:
            kw[f'q_{a}_{b}'] = bh_fdr(kw[pkey].fillna(1).values)

    print(f"  KW significant (BH-FDR, cell-level): {kw['Significant'].sum()} / {len(kw)}")
    kw.to_csv(OUTPUT_DIR / 'L1_kruskal_wallis.csv', index=False)
    sw.to_csv(OUTPUT_DIR / 'L1_shapiro_wilk.csv',   index=False)

    # Dunn's post-hoc on cell-level data
    dunn = {}
    if HAS_POSTHOCS:
        for feat in kw.loc[kw['Significant'], 'Feature']:
            data = [df.loc[df['Disease'] == d, feat].dropna().tolist()
                    for d in DISEASE_ORDER]
            try:
                pm = sp.posthoc_dunn(data, p_adjust='bonferroni')
                pm.index = pm.columns = DISEASE_ORDER
                dunn[feat] = pm
            except: pass
        print(f"  Dunn's post-hoc: {len(dunn)} features")

    _plot_sw_summary(sw, '01_L1_shapiro_wilk.png', 'Layer 1')
    _plot_kw_pairwise(kw, '02_L1_kruskal_wallis.png')
    _plot_cohens_d_3way(kw, '03_L1_cohens_d.png')
    if dunn:
        _plot_dunn_heatmaps(dunn, kw, '04_L1_dunn_posthoc.png',
                            group_labels=DISEASE_ORDER)
    _plot_boxplots_by_group(df, feat_cols, kw, dunn,
                            '05_L1_boxplots', 'Disease',
                            DISEASE_ORDER, DISEASE_COLORS,
                            DISEASE_LABELS, '(Layer 1 — Disease)')

    return kw, sw, dunn

def _plot_cohens_d_3way(kw, fname, title=None):
    pairs = [('d_Healthy_TAA', 'Non-A vs TAA', '#E74C3C'),
             ('d_Healthy_BAV', 'Non-A vs BAV', '#F39C12'),
             ('d_TAA_BAV',     'TAA vs BAV',   '#9B59B6')]

    # Shared feature order: sort by d_Healthy_TAA (primary comparison)
    # so all three panels share the same y-axis layout
    feature_order = (kw[['Feature', 'd_Healthy_TAA']]
                     .dropna()
                     .sort_values('d_Healthy_TAA')['Feature']
                     .tolist())

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    suptitle = title or ("Layer 1 — Cohen's d  (pairwise, cell-level, BH-FDR)\n"
                         "Sorted by Non-A vs TAA effect size")
    fig.suptitle(suptitle, fontsize=17, fontweight='bold')

    for ax, (col, label, color) in zip(axes, pairs):
        if col not in kw.columns: ax.set_visible(False); continue

        # Apply shared order; fill missing d values with 0
        sub = (kw.set_index('Feature')
                 .reindex(feature_order)
                 .reset_index())[['Feature', col, 'Significant', 'BH_q']]
        sub[col]          = sub[col].fillna(0)
        sub['Significant'] = sub['Significant'].fillna(False)
        sub['BH_q']        = sub['BH_q'].fillna(1.0)

        oc = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA') for f in sub['Feature']]
        ax.barh(range(len(sub)), sub[col], color=oc, edgecolor='white', height=0.7)
        ax.axvline(0, color='black', lw=0.9)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=13)
        ax.set_xlabel("Cohen's d", fontsize=15)
        ax.set_title(label, fontsize=15, fontweight='bold', color=color, pad=12)
        ax.tick_params(axis='x', labelsize=13)
        ax.grid(axis='x', alpha=0.3)

        # Asterisks at bar tips
        ax.autoscale(axis='x', tight=False)
        xlim = ax.get_xlim()
        pad  = (xlim[1] - xlim[0]) * 0.025
        for i, (_, row) in enumerate(sub.iterrows()):
            if row['Significant']:
                x = row[col]
                ax.text(x + (pad if x >= 0 else -pad), i,
                        sig_stars(row['BH_q']),
                        va='center', ha='left' if x >= 0 else 'right',
                        fontsize=13, fontweight='bold')
        # extend xlim to avoid clipping asterisks
        cur = ax.get_xlim()
        ax.set_xlim(cur[0] - abs(cur[0]) * 0.12, cur[1] + abs(cur[1]) * 0.12)

    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=len(ORGANELLE_COLORS),
               fontsize=13, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.05, '* p<0.05   ** p<0.01   *** p<0.001   (BH-FDR, omnibus KW, cell-level)',
             ha='center', fontsize=12, style='italic', color='#444')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, fname)

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — AGE-RESIDUALIZED
# ══════════════════════════════════════════════════════════════════════════════
def layer2_age_residualized(df, feat_cols, kw_l1):
    print("\n" + "=" * 65)
    print("LAYER 2 — AGE-RESIDUALIZED DISEASE EFFECT  (cell-level)")
    print("=" * 65)

    # Step 1 — regress age out at cell level
    df_r = df.copy()
    n_resid = 0
    for feat in feat_cols:
        sub = df[['Age', feat]].dropna()
        if len(sub) < 10: continue
        mdl = LinearRegression().fit(sub[['Age']], sub[feat])
        pred = mdl.predict(df[['Age']].fillna(df['Age'].mean()))
        df_r[feat] = df[feat] - pred
        n_resid += 1
    print(f"  Regressed age out of {n_resid} features")

    n_cells = {d: int((df_r['Disease'] == d).sum()) for d in DISEASE_ORDER}
    print(f"  Cell-level n: " +
          "  ".join(f"{d}={n}" for d, n in n_cells.items()))

    # Step 2 — KW on age-residualized cell-level data
    kw_rows = []
    for feat in feat_cols:
        groups = [df_r.loc[df_r['Disease'] == d, feat].dropna().values
                  for d in DISEASE_ORDER]
        ok = [g for g in groups if len(g) >= 2]
        if len(ok) < 2: continue
        try:   stat, p = kruskal(*ok)
        except: stat, p = np.nan, np.nan
        kw_rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                        'KW_stat': stat, 'KW_p': p})

    kw2 = pd.DataFrame(kw_rows)
    kw2['BH_q']       = bh_fdr(kw2['KW_p'].values)
    kw2['Significant'] = kw2['BH_q'] < ALPHA

    l1_sig = set(kw_l1.loc[kw_l1['Significant'], 'Feature'])
    l2_sig = set(kw2.loc[kw2['Significant'], 'Feature'])
    print(f"  After age correction (cell-level): {len(l2_sig)} significant  "
          f"(kept={len(l1_sig & l2_sig)}, "
          f"lost={len(l1_sig - l2_sig)}, "
          f"new={len(l2_sig - l1_sig)})")
    kw2.to_csv(OUTPUT_DIR / 'L2_kruskal_wallis_age_corrected.csv', index=False)

    # L1 vs L2 scatter
    merged = kw_l1[['Feature', 'BH_q']].merge(
        kw2[['Feature', 'BH_q']], on='Feature', suffixes=('_L1', '_L2'))
    merged['-l10_L1'] = -np.log10(merged['BH_q_L1'].clip(1e-10))
    merged['-l10_L2'] = -np.log10(merged['BH_q_L2'].clip(1e-10))
    colors = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA') for f in merged['Feature']]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(merged['-l10_L1'], merged['-l10_L2'], c=colors, alpha=0.7, s=55)
    lim = max(merged[['-l10_L1', '-l10_L2']].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.4)
    thr = -np.log10(ALPHA)
    ax.axhline(thr, color='red', lw=0.8, ls=':')
    ax.axvline(thr, color='red', lw=0.8, ls=':')
    ax.set_xlabel('-log₁₀(q)  Layer 1 (raw)', fontsize=11)
    ax.set_ylabel('-log₁₀(q)  Layer 2 (age-corrected)', fontsize=11)
    ax.set_title('Layer 2 — Age Correction Impact\n'
                 'Above diagonal = effect strengthened after correction',
                 fontsize=12, fontweight='bold')
    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, fontsize=9)
    save(fig, '06_L2_age_comparison_scatter.png')

    _plot_kw_lollipop(kw2, '07_L2_kruskal_wallis_agecorrected.png',
                      'Layer 2 — Age-corrected Disease Effect')
    _plot_boxplots_by_group(df_r, feat_cols, kw2, {},
                            '08_L2_boxplots', 'Disease',
                            DISEASE_ORDER, DISEASE_COLORS,
                            DISEASE_LABELS, '(Layer 2 — age-corrected)')
    return kw2

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — LINEAR MIXED MODELS
# ══════════════════════════════════════════════════════════════════════════════
def layer3_lmm(df, feat_cols, kw_l1):
    print("\n" + "=" * 65)
    print("LAYER 3 — LINEAR MIXED MODELS")
    print("  Fixed:  Disease_TAA + Coll_bin + Age + Sex_bin + Disease_TAA:Coll_bin")
    print("  Random: Subject (random intercept)")
    print("=" * 65)

    if not HAS_STATSMODELS:
        print("  SKIPPED — pip install statsmodels")
        return None

    sig_feats = kw_l1.loc[kw_l1['Significant'], 'Feature'].tolist() or feat_cols
    print(f"  Running LMM on {len(sig_feats)} features...")

    records = []
    for feat in sig_feats:
        needed = ['Subject', 'Disease_TAA', 'Coll_bin', 'Age', 'Sex_bin', feat]
        sub = df[needed].rename(columns={feat: 'Y'}).dropna()
        if sub['Subject'].nunique() < 3: continue
        try:
            res = smf.mixedlm(
                'Y ~ Disease_TAA * Coll_bin + Age + Sex_bin',
                data=sub, groups=sub['Subject']
            ).fit(reml=True, method='lbfgs')

            term_map = {
                'Disease (TAA)':       'Disease_TAA',
                'Collagen':            'Coll_bin',
                'Age':                 'Age',
                'Sex (Male)':          'Sex_bin',
                'Disease × Collagen':  'Disease_TAA:Coll_bin',
            }
            for label, param in term_map.items():
                if param in res.params:
                    ci = res.conf_int().loc[param]
                    records.append({
                        'Feature': feat, 'Organelle': get_organelle(feat),
                        'Term': label,
                        'Coef': res.params[param], 'SE': res.bse[param],
                        'p': res.pvalues[param],
                        'CI_low': ci[0], 'CI_high': ci[1],
                    })
        except: pass

    if not records:
        print("  No LMM results.")
        return None

    lmm = pd.DataFrame(records)
    for term in lmm['Term'].unique():
        mask = lmm['Term'] == term
        lmm.loc[mask, 'BH_q'] = bh_fdr(lmm.loc[mask, 'p'].values)
    lmm['Significant'] = lmm['BH_q'] < ALPHA
    lmm.to_csv(OUTPUT_DIR / 'L3_lmm_results.csv', index=False)
    print(f"  LMM: {lmm['Significant'].sum()} significant term-feature pairs")

    _plot_lmm_forest(lmm, '09_L3_lmm_forest.png')
    return lmm

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — COLLAGEN RESCUE
# ══════════════════════════════════════════════════════════════════════════════
def layer4_collagen(df, feat_cols):
    print("\n" + "=" * 65)
    print("LAYER 4 — COLLAGEN RESCUE  (cell-level, Mann-Whitney)")
    print("=" * 65)

    # ── Cell-level Mann-Whitney: NoCollagen vs Collagen cells per disease ─────
    all_rows  = []
    res_by_dis = {}
    for dis in DISEASE_ORDER:
        sub    = df[df['Disease'] == dis]
        nc_cells  = sub[sub['Collagen_Status'] == 'NoCollagen']
        col_cells = sub[sub['Collagen_Status'] == 'Collagen']
        print(f"  {dis}: {len(nc_cells)} NoCollagen cells  +  {len(col_cells)} Collagen cells")

        rows = []
        for feat in feat_cols:
            d1 = nc_cells[feat].dropna().values
            d2 = col_cells[feat].dropna().values
            if len(d1) < 3 or len(d2) < 3: continue
            _, p = mannwhitneyu(d1, d2, alternative='two-sided')
            rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                         'Cohen_d': cohens_d(d1, d2), 'p': p, 'Test': 'Mann-Whitney'})

        res = pd.DataFrame(rows)
        res['BH_q']        = bh_fdr(res['p'].values)
        res['Significant']  = res['BH_q'] < ALPHA
        all_rows.append(res.assign(Disease=dis))
        res_by_dis[dis] = res
        n_sig = res['Significant'].sum()
        print(f"    Significant (BH-FDR): {n_sig} / {len(res)}")

    # Shared feature order: sort by Healthy Cohen_d
    feature_order = (res_by_dis['Healthy'][['Feature', 'Cohen_d']]
                     .sort_values('Cohen_d')['Feature'].tolist())

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    fig.suptitle("Layer 4 — Collagen Effect per Disease Group\n"
                 "Cohen's d: positive = higher in +Collagen  |  cell-level  |  Mann-Whitney BH-FDR\n"
                 "Sorted by Healthy effect size",
                 fontsize=15, fontweight='bold')

    for ax, dis in zip(axes, DISEASE_ORDER):
        res = res_by_dis[dis]
        # Apply shared order
        res = (res.set_index('Feature')
                  .reindex(feature_order)
                  .reset_index())
        res['Cohen_d']     = res['Cohen_d'].fillna(0)
        res['Significant'] = res['Significant'].fillna(False)
        res['BH_q']        = res['BH_q'].fillna(1.0)

        oc = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA') for f in res['Feature']]
        ax.barh(range(len(res)), res['Cohen_d'], color=oc, edgecolor='white', height=0.7)
        ax.axvline(0, color='black', lw=0.9)
        ax.set_yticks(range(len(res)))
        ax.set_yticklabels([f.replace('_', ' ') for f in res['Feature']], fontsize=11)
        ax.set_xlabel("Cohen's d  (+Col vs NoCol)", fontsize=13)
        ax.tick_params(axis='x', labelsize=11)
        n_sig = res['Significant'].sum()
        ax.set_title(f'{DISEASE_LABELS[dis]}  ({n_sig}/{len(res)} sig)',
                     fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Asterisks at bar tips
        ax.autoscale(axis='x', tight=False)
        xlim = ax.get_xlim()
        pad  = (xlim[1] - xlim[0]) * 0.025
        for i, (_, row) in enumerate(res.iterrows()):
            if row['Significant']:
                x = row['Cohen_d']
                ax.text(x + (pad if x >= 0 else -pad), i, sig_stars(row['BH_q']),
                        va='center', ha='left' if x >= 0 else 'right',
                        fontsize=11, fontweight='bold')
        cur = ax.get_xlim()
        ax.set_xlim(cur[0] - abs(cur[0]) * 0.12, cur[1] + abs(cur[1]) * 0.12)

    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=len(ORGANELLE_COLORS),
               fontsize=13, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.05, '* p<0.05   ** p<0.01   *** p<0.001   (BH-FDR, Mann-Whitney, cell-level)',
             ha='center', fontsize=12, style='italic', color='#444')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, '10_L4_collagen_effect.png')

    pd.concat(all_rows).to_csv(OUTPUT_DIR / 'L4_collagen_results.csv', index=False)

    # Distance to Healthy centroid — cell-level
    # Fit scaler on all cells, compute centroid from Healthy NoCollagen cells
    scaler   = StandardScaler()
    X_sc     = pd.DataFrame(scaler.fit_transform(df[feat_cols]),
                            columns=feat_cols, index=df.index)
    centroid = X_sc[df['Disease'] == 'Healthy'].mean().values

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6))
    fig2.suptitle('Layer 4 — Distance to Healthy Centroid  (rescue effect, cell-level)',
                  fontsize=13, fontweight='bold')
    for ax2, dis in zip(axes2, DISEASE_ORDER):
        idx_nc  = df.index[(df['Disease'] == dis) & (df['Collagen_Status'] == 'NoCollagen')]
        idx_col = df.index[(df['Disease'] == dis) & (df['Collagen_Status'] == 'Collagen')]

        d_nc  = np.linalg.norm(X_sc.loc[idx_nc].values  - centroid, axis=1)
        d_col = np.linalg.norm(X_sc.loc[idx_col].values - centroid, axis=1)

        bp = ax2.boxplot([d_nc, d_col], patch_artist=True,
                         medianprops=dict(color='black', lw=2))
        bp['boxes'][0].set_facecolor(COLLAGEN_COLORS['NoCollagen']); bp['boxes'][0].set_alpha(0.8)
        bp['boxes'][1].set_facecolor(COLLAGEN_COLORS['Collagen']);   bp['boxes'][1].set_alpha(0.8)
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(['No Collagen', '+Collagen'], fontsize=11)
        ax2.set_ylabel('Distance to Healthy centroid', fontsize=11)
        ax2.grid(axis='y', alpha=0.3)

        if len(d_nc) >= 3 and len(d_col) >= 3:
            _, p = mannwhitneyu(d_nc, d_col, alternative='two-sided')
            y_top = max(d_nc.max(), d_col.max()) * 1.08
            ax2.plot([1,1,2,2], [y_top*0.97, y_top, y_top, y_top*0.97], 'k-', lw=1.2)
            ax2.text(1.5, y_top*1.01, sig_stars(p) + f' (p={p:.3f})',
                     ha='center', fontsize=11, fontweight='bold')

        direction = 'RESCUE' if (len(d_col) > 0 and d_col.mean() < d_nc.mean()) else 'NO RESCUE'
        ax2.set_title(f'{DISEASE_LABELS[dis]}\n{direction}  '
                      f'(NoCol={len(d_nc)}, +Col={len(d_col)} cells)',
                      fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig2, '11_L4_rescue_distance.png')

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — SEX EFFECT
# ══════════════════════════════════════════════════════════════════════════════
def layer5_sex(df, feat_cols):
    print("\n" + "=" * 65)
    print("LAYER 5 — SEX EFFECT  (cell-level, overall + per disease)")
    print("=" * 65)

    def sex_test(sub):
        rows = []
        for feat in feat_cols:
            m = sub[sub['Gender']=='Male'][feat].dropna().values
            f = sub[sub['Gender']=='Female'][feat].dropna().values
            if len(m) < 3 or len(f) < 3: continue
            p, test = adaptive_test(f, m)
            rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                         'Cohen_d': cohens_d(f, m), 'p': p, 'Test': test})
        if not rows: return pd.DataFrame()
        res = pd.DataFrame(rows)
        res['BH_q']        = bh_fdr(res['p'].values)
        res['Significant']  = res['BH_q'] < ALPHA
        return res

    sex_all = sex_test(df)
    sex_all.to_csv(OUTPUT_DIR / 'L5_sex_overall.csv', index=False)
    n_m = int((df['Gender']=='Male').sum())
    n_f = int((df['Gender']=='Female').sum())
    print(f"  Overall: {n_m} male / {n_f} female cells")
    print(f"  Overall sex: {sex_all['Significant'].sum()} / {len(sex_all)} significant")

    # Shared feature order from overall result
    feature_order = (sex_all[['Feature','Cohen_d']]
                     .sort_values('Cohen_d')['Feature'].tolist())

    datasets = [('Overall', df)] + \
               [(d, df[df['Disease']==d]) for d in DISEASE_ORDER]
    fig, axes = plt.subplots(1, 4, figsize=(26, 10))
    fig.suptitle("Layer 5 — Sex Effect  (Male vs Female, +d = higher in Males)\n"
                 "Cell-level  |  sorted by Overall effect size",
                 fontsize=15, fontweight='bold')

    for ax, (label, sub_pt) in zip(axes, datasets):
        res = sex_test(sub_pt)
        if res.empty: ax.set_visible(False); continue
        # Apply shared order
        res = (res.set_index('Feature').reindex(feature_order).reset_index())
        res['Cohen_d']     = res['Cohen_d'].fillna(0)
        res['Significant'] = res['Significant'].fillna(False)
        res['BH_q']        = res['BH_q'].fillna(1.0)

        oc = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA') for f in res['Feature']]
        ax.barh(range(len(res)), res['Cohen_d'], color=oc, edgecolor='white', height=0.7)
        ax.axvline(0, color='black', lw=0.9)
        ax.set_yticks(range(len(res)))
        ax.set_yticklabels([f.replace('_', ' ') for f in res['Feature']], fontsize=11)
        ax.set_xlabel("Cohen's d", fontsize=13)
        ax.tick_params(axis='x', labelsize=11)
        n_sig = int(res['Significant'].sum())
        ax.set_title(f'{label}\n({n_sig}/{len(res)} sig)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        xlim = ax.get_xlim()
        pad  = (xlim[1] - xlim[0]) * 0.025
        for i, (_, row) in enumerate(res.iterrows()):
            if row['Significant']:
                x = row['Cohen_d']
                ax.text(x + (pad if x >= 0 else -pad), i, sig_stars(row['BH_q']),
                        va='center', ha='left' if x >= 0 else 'right',
                        fontsize=11, fontweight='bold')
        cur = ax.get_xlim()
        ax.set_xlim(cur[0] - abs(cur[0])*0.12, cur[1] + abs(cur[1])*0.12)

    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=len(ORGANELLE_COLORS),
               fontsize=13, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.05, '* p<0.05   ** p<0.01   *** p<0.001   (BH-FDR, cell-level)',
             ha='center', fontsize=12, style='italic', color='#444')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, '12_L5_sex_effect.png')

    # Sex × Disease boxplots for top significant features
    sig_sex = sex_all.loc[sex_all['Significant'], 'Feature'].tolist()[:6]
    if sig_sex:
        ncols = 3
        nrows = int(np.ceil(len(sig_sex) / ncols))
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4.5))
        axes2 = np.array(axes2).flatten()
        for i, feat in enumerate(sig_sex):
            ax = axes2[i]
            for j, dis in enumerate(DISEASE_ORDER):
                sub = df[df['Disease'] == dis]
                m_v = sub[sub['Gender']=='Male'][feat].dropna().values
                f_v = sub[sub['Gender']=='Female'][feat].dropna().values
                xm, xf = j*3+1, j*3+2
                bp = ax.boxplot([m_v, f_v], positions=[xm, xf],
                                widths=0.7, patch_artist=True,
                                medianprops=dict(color='black', lw=1.5))
                bp['boxes'][0].set_facecolor(SEX_COLORS['Male'])
                bp['boxes'][1].set_facecolor(SEX_COLORS['Female'])
                for b in bp['boxes']: b.set_alpha(0.75)
            ax.set_xticks([1.5, 4.5, 7.5])
            ax.set_xticklabels(DISEASE_ORDER, fontsize=9)
            ax.set_title(feat.replace('_', ' '), fontsize=9, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        for j in range(len(sig_sex), len(axes2)):
            axes2[j].set_visible(False)

        leg = [mpatches.Patch(color=SEX_COLORS['Male'],   label='Male'),
               mpatches.Patch(color=SEX_COLORS['Female'], label='Female')]
        fig2.legend(handles=leg, loc='lower center', ncol=2, fontsize=10,
                    bbox_to_anchor=(0.5, -0.02))
        fig2.suptitle('Layer 5 — Sex Effect: Significant Features × Disease',
                      fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        save(fig2, '13_L5_sex_by_disease_boxplots.png')

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — HYPERTENSION EFFECT
# ══════════════════════════════════════════════════════════════════════════════
def layer6_hypertension(df, feat_cols):
    print("\n" + "=" * 65)
    print("LAYER 6 — HYPERTENSION EFFECT  (Yes vs No)")
    print("=" * 65)

    df_h = df[df['Hypertn'].isin(['Yes', 'No'])].copy()
    if len(df_h) < 10:
        print("  Insufficient data with hypertension labels — skipping.")
        return

    # Cell-level analysis
    n_y = int((df_h['Hypertn']=='Yes').sum())
    n_n = int((df_h['Hypertn']=='No').sum())
    print(f"  Hypertensive: {n_y} cells | Non-hypertensive: {n_n} cells")

    # Overall
    rows_all = []
    for feat in feat_cols:
        hy = df_h[df_h['Hypertn']=='Yes'][feat].dropna().values
        no = df_h[df_h['Hypertn']=='No'][feat].dropna().values
        if len(hy) < 3 or len(no) < 3: continue
        p, test = adaptive_test(no, hy)
        rows_all.append({'Feature': feat, 'Organelle': get_organelle(feat),
                         'Cohen_d': cohens_d(no, hy), 'p': p, 'Test': test})

    if not rows_all:
        print("  No features could be tested.")
        return

    hyp_res = pd.DataFrame(rows_all)
    hyp_res['BH_q']       = bh_fdr(hyp_res['p'].values)
    hyp_res['Significant'] = hyp_res['BH_q'] < ALPHA
    hyp_res.to_csv(OUTPUT_DIR / 'L6_hypertension_results.csv', index=False)
    print(f"  Hypertension: {hyp_res['Significant'].sum()} / {len(hyp_res)} significant")

    _plot_cohens_d_bars(hyp_res, '14_L6_hypertension_cohens_d.png',
                        'Layer 6 — Hypertension Effect  (cell-level)',
                        'No HyperTN', 'HyperTN', None, None, '#C0392B')

    # Stratified by disease — cell-level, shared feature order
    feature_order = (hyp_res[['Feature','Cohen_d']]
                     .sort_values('Cohen_d')['Feature'].tolist())

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    fig.suptitle("Layer 6 — Hypertension Effect Stratified by Disease\n"
                 "Cell-level  |  sorted by Overall effect size",
                 fontsize=15, fontweight='bold')
    for ax, dis in zip(axes, DISEASE_ORDER):
        sub = df_h[df_h['Disease'] == dis]
        n_hy = int((sub['Hypertn']=='Yes').sum())
        n_no = int((sub['Hypertn']=='No').sum())
        if n_hy < 2 or n_no < 2:
            ax.set_visible(False); continue
        rows = []
        for feat in feat_cols:
            d1 = sub[sub['Hypertn'] == 'No'][feat].dropna().values
            d2 = sub[sub['Hypertn'] == 'Yes'][feat].dropna().values
            if len(d1) < 2 or len(d2) < 2: continue
            p, _ = adaptive_test(d1, d2)
            rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                         'Cohen_d': cohens_d(d1, d2), 'p': p})
        if not rows: ax.set_visible(False); continue
        res = pd.DataFrame(rows)
        res['BH_q']        = bh_fdr(res['p'].values)
        res['Significant']  = res['BH_q'] < ALPHA
        res = (res.set_index('Feature').reindex(feature_order).reset_index())
        res['Cohen_d']     = res['Cohen_d'].fillna(0)
        res['Significant'] = res['Significant'].fillna(False)
        res['BH_q']        = res['BH_q'].fillna(1.0)

        oc = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA') for f in res['Feature']]
        ax.barh(range(len(res)), res['Cohen_d'], color=oc, edgecolor='white', height=0.7)
        ax.axvline(0, color='black', lw=0.9)
        ax.set_yticks(range(len(res)))
        ax.set_yticklabels([f.replace('_',' ') for f in res['Feature']], fontsize=11)
        ax.set_xlabel("Cohen's d", fontsize=13)
        ax.tick_params(axis='x', labelsize=11)
        n_sig = int(res['Significant'].sum())
        ax.set_title(f'{DISEASE_LABELS[dis]}  (HT={n_hy}, No-HT={n_no})\n({n_sig}/{len(res)} sig)',
                     fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        xlim = ax.get_xlim()
        pad  = (xlim[1] - xlim[0]) * 0.025
        for i, (_, row) in enumerate(res.iterrows()):
            if row['Significant']:
                x = row['Cohen_d']
                ax.text(x + (pad if x >= 0 else -pad), i, sig_stars(row['BH_q']),
                        va='center', ha='left' if x >= 0 else 'right',
                        fontsize=11, fontweight='bold')
        cur = ax.get_xlim()
        ax.set_xlim(cur[0] - abs(cur[0])*0.12, cur[1] + abs(cur[1])*0.12)

    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=len(ORGANELLE_COLORS),
               fontsize=13, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.05, '* p<0.05   ** p<0.01   *** p<0.001   (BH-FDR, cell-level)',
             ha='center', fontsize=12, style='italic', color='#444')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, '16_L6_hypertension_by_disease.png')

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7 — AORTA DIAMETER CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
def layer7_aorta_diameter(df, feat_cols):
    print("\n" + "=" * 65)
    print("LAYER 7 — AORTA DIAMETER CORRELATIONS  (Spearman)")
    print("=" * 65)

    df_d = df[df['AortaDiam_mm'].notna()].copy()
    if len(df_d) < 10:
        print("  Insufficient data with aorta diameter — skipping.")
        return

    # Cell-level: each cell is one observation
    print(f"  Cells with diameter data: {len(df_d)}")
    print(f"  Diameter range: {df_d['AortaDiam_mm'].min():.0f} – "
          f"{df_d['AortaDiam_mm'].max():.0f} mm  "
          f"(mean {df_d['AortaDiam_mm'].mean():.1f} mm)")

    diam = df_d['AortaDiam_mm'].values
    rows = []
    for feat in feat_cols:
        vals = df_d[feat].values
        valid = ~np.isnan(vals) & ~np.isnan(diam)
        if valid.sum() < 5: continue
        r, p = spearmanr(diam[valid], vals[valid])
        rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                     'Spearman_r': r, 'p': p})

    if not rows:
        print("  No correlations computed.")
        return

    corr = pd.DataFrame(rows)
    corr['BH_q']       = bh_fdr(corr['p'].values)
    corr['Significant'] = corr['BH_q'] < ALPHA
    corr.to_csv(OUTPUT_DIR / 'L7_aorta_diameter_correlations.csv', index=False)
    print(f"  Diameter correlations: {corr['Significant'].sum()} / {len(corr)} significant")

    # ── Lollipop plot of Spearman r (cell-level) ─────────────────────────────
    corr_sorted = corr.sort_values('Spearman_r')
    colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in corr_sorted['Organelle']]

    fig, ax = plt.subplots(figsize=(11, max(6, len(corr_sorted) * 0.45)))
    ax.barh(range(len(corr_sorted)), corr_sorted['Spearman_r'],
            color=colors, edgecolor='white', height=0.7)
    ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(range(len(corr_sorted)))
    ax.set_yticklabels([f.replace('_', ' ') for f in corr_sorted['Feature']], fontsize=11)
    ax.set_xlabel('Spearman r  (vs Aorta Diameter mm)', fontsize=13)
    ax.tick_params(axis='x', labelsize=11)
    n_sig = corr['Significant'].sum()
    ax.set_title(f'Layer 7 — Aorta Diameter Correlations  '
                 f'({n_sig}/{len(corr)} significant, BH-FDR, cell-level)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    xlim = ax.get_xlim()
    pad  = (xlim[1] - xlim[0]) * 0.025
    for i, (_, row) in enumerate(corr_sorted.iterrows()):
        if row['Significant']:
            x = row['Spearman_r']
            ax.text(x + (pad if x >= 0 else -pad), i, sig_stars(row['BH_q']),
                    va='center', ha='left' if x >= 0 else 'right',
                    fontsize=11, fontweight='bold')
    cur = ax.get_xlim()
    ax.set_xlim(cur[0] - abs(cur[0])*0.12, cur[1] + abs(cur[1])*0.12)

    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, fontsize=11, loc='lower right')
    fig.text(0.98, 0.01, '* p<0.05   ** p<0.01   *** p<0.001   (BH-FDR, cell-level)',
             ha='right', fontsize=10, style='italic', color='#444')
    plt.tight_layout()
    save(fig, '17_L7_aorta_diameter_correlations.png')

    # ── Scatter plots for top significant correlations ───────────────────────
    sig_feats = corr.loc[corr['Significant']].sort_values(
        'Spearman_r', key=abs, ascending=False).head(6)['Feature'].tolist()
    if sig_feats:
        ncols = 3
        nrows = int(np.ceil(len(sig_feats) / ncols))
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4.5))
        axes2 = np.array(axes2).flatten()

        for i, feat in enumerate(sig_feats):
            ax = axes2[i]
            for dis in DISEASE_ORDER:
                sub = df_d[df_d['Disease'] == dis]
                ax.scatter(sub['AortaDiam_mm'], sub[feat],
                           c=DISEASE_COLORS[dis], label=DISEASE_LABELS[dis],
                           s=25, alpha=0.55, edgecolors='white', lw=0.3)

            # Regression line (all cell-level data)
            valid = df_d[['AortaDiam_mm', feat]].dropna()
            if len(valid) > 3:
                z = np.polyfit(valid['AortaDiam_mm'], valid[feat], 1)
                xr = np.linspace(valid['AortaDiam_mm'].min(),
                                 valid['AortaDiam_mm'].max(), 50)
                ax.plot(xr, np.polyval(z, xr), 'k--', lw=1.5, alpha=0.7)

            row = corr[corr['Feature'] == feat].iloc[0]
            ax.set_xlabel('Aorta Diameter (mm)', fontsize=9)
            ax.set_title(f"{feat.replace('_', ' ')}\n"
                         f"r={row['Spearman_r']:.2f}, "
                         f"q={row['BH_q']:.3f} {sig_stars(row['BH_q'])}",
                         fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # One shared legend
        handles, labels_ = axes2[0].get_legend_handles_labels()
        unique = dict(zip(labels_, handles))
        fig2.legend(unique.values(), unique.keys(), loc='lower center',
                    ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

        for j in range(len(sig_feats), len(axes2)):
            axes2[j].set_visible(False)

        fig2.suptitle('Layer 7 — Aorta Diameter vs Significant Features',
                      fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        save(fig2, '18_L7_aorta_diameter_scatterplots.png')

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def summary_plots(df, feat_cols, kw_l1):
    print("\n  Generating summary plots...")

    X    = df[feat_cols].values
    X_sc = StandardScaler().fit_transform(X)
    pca  = PCA(n_components=2, random_state=42)
    Xp   = pca.fit_transform(X_sc)
    ev   = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Summary — PCA of Morphological Feature Space',
                 fontsize=14, fontweight='bold')

    # By Disease
    for dis in DISEASE_ORDER:
        mask = df['Disease'].values == dis
        axes[0].scatter(Xp[mask, 0], Xp[mask, 1],
                        c=DISEASE_COLORS[dis], label=DISEASE_LABELS[dis],
                        s=50, alpha=0.72, edgecolors='white', lw=0.3)
        ctr = Xp[mask].mean(axis=0)
        axes[0].scatter(*ctr, c=DISEASE_COLORS[dis], s=220, marker='D',
                        edgecolors='black', lw=1.5, zorder=6)
    axes[0].set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=11)
    axes[0].set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=11)
    axes[0].set_title('Coloured by Disease', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # By Sex
    for sex, color in SEX_COLORS.items():
        mask = df['Gender'].values == sex
        axes[1].scatter(Xp[mask, 0], Xp[mask, 1], c=color, label=sex,
                        s=50, alpha=0.72, edgecolors='white', lw=0.3,
                        marker='o' if sex == 'Male' else 's')
    axes[1].set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=11)
    axes[1].set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=11)
    axes[1].set_title('Coloured by Sex', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    # By Collagen
    for col, color in COLLAGEN_COLORS.items():
        mask = df['Collagen_Status'].values == col
        axes[2].scatter(Xp[mask, 0], Xp[mask, 1], c=color, label=col,
                        s=50, alpha=0.72, edgecolors='white', lw=0.3)
    axes[2].set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=11)
    axes[2].set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=11)
    axes[2].set_title('Coloured by Collagen', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, '19_summary_pca.png')

    # Feature heatmap — 6 sub-groups (Disease × Collagen)
    top_n     = 20
    sig_sorted = (kw_l1[kw_l1['Significant']]
                  .sort_values('KW_stat', ascending=False)
                  .head(top_n)['Feature'].tolist()
                  or kw_l1.sort_values('KW_stat', ascending=False)
                     .head(top_n)['Feature'].tolist())

    X_sc_df = pd.DataFrame(
        StandardScaler().fit_transform(df[feat_cols]),
        columns=feat_cols, index=df.index)

    hm = {}
    for dis in DISEASE_ORDER:
        for col in ['NoCollagen', 'Collagen']:
            mask = (df['Disease'] == dis) & (df['Collagen_Status'] == col)
            label = f"{DISEASE_LABELS[dis]}\n{'−Col' if col=='NoCollagen' else '+Col'}"
            hm[label] = X_sc_df.loc[mask, sig_sorted].mean()

    hm_df = pd.DataFrame(hm, index=sig_sorted)
    hm_df.index = [f.replace('_', ' ') for f in hm_df.index]

    fig2, ax2 = plt.subplots(figsize=(13, max(8, top_n * 0.46)))
    sns.heatmap(hm_df, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                linewidths=0.4, linecolor='lightgray', ax=ax2,
                cbar_kws={'label': 'Z-score (group mean)'})
    ax2.set_title(f'Summary — Top {top_n} Significant Features\n'
                  'Standardised Group Means (KW + BH-FDR)',
                  fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=25, labelsize=9)
    ax2.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    save(fig2, '20_summary_feature_heatmap.png')

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("FINAL STATISTICAL ANALYSIS — ALL LAYERS")
    print("=" * 65)

    df, feat_cols = load_data()

    kw_l1, sw_l1, dunn_l1 = layer1_disease(df, feat_cols)
    kw_l2                  = layer2_age_residualized(df, feat_cols, kw_l1)
    lmm                    = layer3_lmm(df, feat_cols, kw_l1)
    layer4_collagen(df, feat_cols)
    layer5_sex(df, feat_cols)
    layer6_hypertension(df, feat_cols)
    layer7_aorta_diameter(df, feat_cols)
    summary_plots(df, feat_cols, kw_l1)

    print("\n" + "=" * 65)
    print(f"DONE — all outputs in: {OUTPUT_DIR}/")
    print("=" * 65)
    print("\nPNGs:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  {f.name}")
    print("\nCSVs:")
    for f in sorted(OUTPUT_DIR.glob('*.csv')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
