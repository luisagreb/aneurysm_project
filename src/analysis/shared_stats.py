"""
Shared Statistical Engine for Thesis Analysis
==============================================
Two-group pairwise analysis (patient-level primary + cell-level exploratory).

Import this module in final_thesis_part1/run.py and final_thesis_part2/run.py.
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import mannwhitneyu, shapiro, ttest_ind, spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed → pip install statsmodels")

# ── paths (callers set PROJ_ROOT before importing) ────────────────────────────
PROJ_ROOT     = Path(__file__).resolve().parents[2]
FEATURES_FILE = PROJ_ROOT / 'outputs' / 'Advanced_Features_Raw_Final.csv'
METADATA_FILE = PROJ_ROOT / 'Book1 (2).xlsx'
ALPHA         = 0.05

# ── display constants ─────────────────────────────────────────────────────────
GROUP_COLORS = {
    'Healthy': '#4C9BE8',  # TAV-NA
    'TAA':     '#E74C3C',  # TAV-ATAA
    'BAV':     '#F39C12',  # BAV-ATAA
}
GROUP_LABELS = {
    'Healthy': 'TAV-NA',
    'TAA':     'TAV-ATAA',
    'BAV':     'BAV-ATAA',
}
ORGANELLE_COLORS = {
    'Nucleus':      '#1558A0',  # dark blue
    'Actin':        '#2ECC71',  # green
    'Mitochondria': '#E74C3C',  # red
}
LEVEL_COLORS = {'Patient': '#8E44AD', 'Cell': '#E67E22'}

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


def norm_id(s):
    m = re.search(r'(0[123][A-Za-z]+)-?0*(\d+)', str(s), re.IGNORECASE)
    return f"{m.group(1).lower()}-{m.group(2)}" if m else None


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def footer(fig, level='patient'):
    note = ('Patient-level (BH-FDR)' if level == 'patient'
            else 'Cell-level — exploratory (BH-FDR)')
    fig.text(0.98, 0.005,
             f'* p<0.05  ** p<0.01  *** p<0.001  ({note})',
             ha='right', fontsize=7, style='italic', color='#555')


def _paper_rc():
    plt.rcParams.update({
        'font.family':        'Arial',
        'font.sans-serif':    ['Arial'],
        'font.size':          11,
        'font.weight':        'bold',
        'axes.labelsize':     12,
        'axes.labelweight':   'bold',
        'axes.titlesize':     11,
        'axes.titleweight':   'bold',
        'xtick.labelsize':    10,
        'ytick.labelsize':    10,
        'legend.fontsize':    10,
        'axes.linewidth':     0.8,
        'xtick.major.width':  0.8,
        'ytick.major.width':  0.8,
        'xtick.major.size':   3,
        'ytick.major.size':   3,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.grid':          False,
        'savefig.dpi':        300,
    })


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _grid(ax, axis='x'):
    if axis == 'x':
        ax.xaxis.grid(True, color='#CCCCCC', linewidth=0.5, zorder=0)
    else:
        ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def adaptive_test(a, b):
    """Shapiro-Wilk → t-test if both normal (n≥3), else Mann-Whitney U."""
    if len(a) < 3 or len(b) < 3:
        return np.nan
    try:
        n_ok = (len(a) >= 8 and shapiro(a).pvalue > 0.05 and
                len(b) >= 8 and shapiro(b).pvalue > 0.05)
        if n_ok:
            _, p = ttest_ind(a, b)
        else:
            _, p = mannwhitneyu(a, b, alternative='two-sided')
    except Exception:
        p = np.nan
    return p


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data(g1, g2, out_dir=None):
    """
    Load features for two groups only (g1, g2 are Disease column values).
    Returns (df_all, df_nc, df_pt, feat_cols).
      df_all : all cells for the two groups (including Collagen rows)
      df_nc  : NoCollagen cells only
      df_pt  : patient-level means (NoCollagen, for disease comparison)
    """
    print("=" * 70)
    print(f"LOADING DATA  ({GROUP_LABELS[g1]} vs {GROUP_LABELS[g2]})")
    print("=" * 70)

    df = pd.read_csv(FEATURES_FILE)
    df = df[df['Disease'].isin([g1, g2])].copy()
    df['Subject'] = df['CellName'].apply(norm_id)

    feat_cols = sorted([c for c in df.columns
                        if c.startswith(('Actin_', 'Mito_', 'Nucleus_'))])
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)

    # Metadata (hypertension, aorta diameter) — graceful fallback
    try:
        meta = pd.read_excel(METADATA_FILE, sheet_name='specimen information')
        meta['Subject']      = meta['Aortic Specimen ID'].apply(norm_id)
        meta['Hypertn']      = meta['hypertn'].map({'Yes': 'Yes', 'No': 'No'})
        meta['AortaDiam_mm'] = pd.to_numeric(meta['trtznlrgdiammeas'], errors='coerce')
        df = df.merge(meta[['Subject', 'Hypertn', 'AortaDiam_mm']].dropna(subset=['Subject']),
                      on='Subject', how='left')
        print(f"  Hypertension matched: {df['Hypertn'].notna().sum()} cells")
        print(f"  Aorta diameter matched: {df['AortaDiam_mm'].notna().sum()} cells")
    except Exception as e:
        print(f"  Metadata not loaded: {e}")
        df['Hypertn']      = np.nan
        df['AortaDiam_mm'] = np.nan

    df_nc = df[df['Collagen_Status'] == 'NoCollagen'].copy()

    agg_f = df_nc.groupby('Subject')[feat_cols].mean().reset_index()
    agg_m = (df_nc.groupby('Subject')
             .agg(Disease=('Disease', 'first'), Gender=('Gender', 'first'),
                  Age=('Age', 'mean'), N_cells=('Disease', 'count'),
                  Hypertn=('Hypertn', 'first'),
                  AortaDiam_mm=('AortaDiam_mm', 'first'))
             .reset_index())
    df_pt = agg_m.merge(agg_f, on='Subject')

    print(f"\n  Cell-level (all): {len(df)} cells")
    print(f"  Cell-level (NoCollagen): {len(df_nc)} cells")
    print(f"  Patient-level: {len(df_pt)} patients")
    for g in [g1, g2]:
        sp = df_pt[df_pt['Disease'] == g]
        sc = df_nc[df_nc['Disease'] == g]
        ages = sp['Age'].dropna()
        mean_age = ages.mean() if len(ages) else float('nan')
        std_age  = ages.std()  if len(ages) else float('nan')
        print(f"    {GROUP_LABELS[g]:12s}: {len(sp)} patients ({len(sc)} cells) | "
              f"Age {mean_age:.1f}±{std_age:.1f}yr | "
              f"M={(sp['Gender'] == 'Male').sum()} F={(sp['Gender'] == 'Female').sum()}")

    if out_dir:
        df_pt.to_csv(Path(out_dir) / 'patient_means.csv', index=False)

    return df, df_nc, df_pt, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# CORE: PAIRWISE 2-GROUP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_pairwise(df_input, g1, g2, feat_cols, label=''):
    """
    Adaptive test (Shapiro-Wilk → t-test if both normal, else MWU) +
    BH-FDR correction + Cohen's d for g1 vs g2.
    Returns a DataFrame with one row per feature.
    """
    rows = []
    for feat in feat_cols:
        a = df_input[df_input['Disease'] == g1][feat].dropna().values
        b = df_input[df_input['Disease'] == g2][feat].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        p = adaptive_test(a, b)
        d = cohens_d(a, b)
        rows.append({
            'Feature':    feat,
            'Organelle':  get_organelle(feat),
            f'n_{g1}':    len(a),
            f'n_{g2}':    len(b),
            f'mean_{g1}': np.mean(a),
            f'mean_{g2}': np.mean(b),
            'Cohen_d':    d,
            'p':          p,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res['BH_q']        = bh_fdr(res['p'].fillna(1).values)
    res['Significant'] = res['BH_q'] < ALPHA
    n_sig = res['Significant'].sum()
    print(f"  {label}: {n_sig}/{len(res)} significant (adaptive test, BH-FDR)")
    return res


def run_age_correction(df_input, feat_cols, label=''):
    """Regress age out of each feature, return residualized copy."""
    df_r = df_input.copy()
    n = 0
    for feat in feat_cols:
        sub = df_input[['Age', feat]].dropna()
        if len(sub) < 5:
            continue
        mdl  = LinearRegression().fit(sub[['Age']], sub[feat])
        age_ = df_input['Age'].fillna(df_input['Age'].mean()).values.reshape(-1, 1)
        df_r[feat] = df_input[feat] - mdl.predict(age_)
        n += 1
    print(f"  {label}: age regressed out of {n}/{len(feat_cols)} features")
    return df_r


# ══════════════════════════════════════════════════════════════════════════════
# L3 — LMM
# ══════════════════════════════════════════════════════════════════════════════
def run_lmm(df, g1, g2, feat_cols, sig_feats=None, label=''):
    """
    LMM: Disease + Age + Sex fixed effects, Subject random intercept.
    Fits on all cells for g1 and g2 (not just NoCollagen).
    sig_feats: list of features to test (default = all).
    """
    if not HAS_STATSMODELS:
        print("  LMM skipped — pip install statsmodels")
        return None

    target = sig_feats if sig_feats else feat_cols
    print(f"  {label}: LMM on {len(target)} features")

    df2 = df[df['Disease'].isin([g1, g2])].copy()
    df2['Disease_bin'] = (df2['Disease'] == g2).astype(float)
    df2['Sex_bin']     = (df2['Gender']  == 'Male').astype(float)

    records = []
    for feat in target:
        needed = ['Subject', 'Disease_bin', 'Age', 'Sex_bin', feat]
        sub    = df2[needed].rename(columns={feat: 'Y'}).dropna()
        if sub['Subject'].nunique() < 5:
            continue
        try:
            res = smf.mixedlm(
                'Y ~ Disease_bin + Age + Sex_bin',
                data=sub, groups=sub['Subject']
            ).fit(reml=True, method='lbfgs')
            g2_label = f'{GROUP_LABELS[g2]} vs {GROUP_LABELS[g1]}'
            for term_label, param in [(g2_label, 'Disease_bin'),
                                       ('Age',    'Age'),
                                       ('Sex',    'Sex_bin')]:
                if param in res.params:
                    ci = res.conf_int().loc[param]
                    records.append({
                        'Feature':    feat,
                        'Organelle':  get_organelle(feat),
                        'Term':       term_label,
                        'Coef':       res.params[param],
                        'SE':         res.bse[param],
                        'p':          res.pvalues[param],
                        'CI_low':     ci[0],
                        'CI_high':    ci[1],
                        'N_cells':    len(sub),
                        'N_patients': sub['Subject'].nunique(),
                    })
        except Exception:
            pass

    if not records:
        print(f"  {label}: no LMM results")
        return None

    lmm = pd.DataFrame(records)
    for term in lmm['Term'].unique():
        mask = lmm['Term'] == term
        lmm.loc[mask, 'BH_q'] = bh_fdr(lmm.loc[mask, 'p'].values)
    lmm['Significant'] = lmm['BH_q'] < ALPHA

    print(f"  Significant per term:")
    for term in lmm['Term'].unique():
        sig = lmm[(lmm['Term'] == term) & lmm['Significant']]['Feature'].tolist()
        print(f"    {term:30s}: {len(sig)} — {sig[:5]}{'...' if len(sig)>5 else ''}")
    return lmm


# ══════════════════════════════════════════════════════════════════════════════
# L4 — COLLAGEN RESCUE
# ══════════════════════════════════════════════════════════════════════════════
def collagen_rescue(df, g1, g2, feat_cols, label='', out_dir=None):
    """
    Patient-level: paired Wilcoxon on per-patient means (NoCollagen vs Collagen).
    Cell-level:    Mann-Whitney on all cells.
    Returns (pt_coll, cell_coll) DataFrames, one row per (Disease, Feature).
    """
    print(f"\n  {label} — Collagen Rescue")
    groups = [g1, g2]

    # ── Patient-level: paired Wilcoxon ────────────────────────────────────────
    pt_agg = (df.groupby(['Subject', 'Disease', 'Collagen_Status'])[feat_cols]
              .mean().reset_index())
    pt_rows = []
    for dis in groups:
        for feat in feat_cols:
            nc  = (pt_agg[(pt_agg['Disease'] == dis) &
                          (pt_agg['Collagen_Status'] == 'NoCollagen')]
                   .set_index('Subject')[feat])
            col = (pt_agg[(pt_agg['Disease'] == dis) &
                          (pt_agg['Collagen_Status'] == 'Collagen')]
                   .set_index('Subject')[feat])
            common = nc.index.intersection(col.index)
            if len(common) < 4:
                continue
            diff = col.loc[common].values - nc.loc[common].values
            try:
                _, p = wilcoxon(diff)
            except Exception:
                p = np.nan
            d = cohens_d(nc.loc[common].values, col.loc[common].values)
            pt_rows.append({'Disease': dis, 'Feature': feat,
                            'Organelle': get_organelle(feat),
                            'Cohen_d': d, 'p': p, 'N_pairs': len(common)})

    pt_coll = pd.DataFrame(pt_rows)
    if not pt_coll.empty:
        for dis in groups:
            mask = pt_coll['Disease'] == dis
            pt_coll.loc[mask, 'BH_q'] = bh_fdr(pt_coll.loc[mask, 'p'].fillna(1).values)
        pt_coll['Significant'] = pt_coll['BH_q'] < ALPHA
        for dis in groups:
            sub = pt_coll[pt_coll['Disease'] == dis]
            n_p = int(sub['N_pairs'].iloc[0]) if len(sub) else 0
            print(f"    Patient  {GROUP_LABELS[dis]:12s}: "
                  f"{sub['Significant'].sum()}/{len(sub)} sig  (paired Wilcoxon, n={n_p})")

    # ── Cell-level: Mann-Whitney ───────────────────────────────────────────────
    cell_rows = []
    for dis in groups:
        sub_d = df[df['Disease'] == dis]
        nc_c  = sub_d[sub_d['Collagen_Status'] == 'NoCollagen']
        col_c = sub_d[sub_d['Collagen_Status'] == 'Collagen']
        for feat in feat_cols:
            d1 = nc_c[feat].dropna().values
            d2 = col_c[feat].dropna().values
            if len(d1) < 3 or len(d2) < 3:
                continue
            try:
                _, p = mannwhitneyu(d1, d2, alternative='two-sided')
            except Exception:
                p = np.nan
            cell_rows.append({'Disease': dis, 'Feature': feat,
                               'Organelle': get_organelle(feat),
                               'Cohen_d': cohens_d(d1, d2), 'p': p})

    cell_coll = pd.DataFrame(cell_rows)
    if not cell_coll.empty:
        for dis in groups:
            mask = cell_coll['Disease'] == dis
            cell_coll.loc[mask, 'BH_q'] = bh_fdr(
                cell_coll.loc[mask, 'p'].fillna(1).values)
        cell_coll['Significant'] = cell_coll['BH_q'] < ALPHA
        for dis in groups:
            sub = cell_coll[cell_coll['Disease'] == dis]
            print(f"    Cell     {GROUP_LABELS[dis]:12s}: "
                  f"{sub['Significant'].sum()}/{len(sub)} sig  (Mann-Whitney)")

    if out_dir:
        out = Path(out_dir)
        if not pt_coll.empty:
            pt_coll.to_csv(out / 'L4_collagen_wilcoxon_pt.csv', index=False)
        if not cell_coll.empty:
            cell_coll.to_csv(out / 'L4_collagen_mw_cell.csv', index=False)

    return pt_coll, cell_coll


# ══════════════════════════════════════════════════════════════════════════════
# L5 — SEX EFFECT
# ══════════════════════════════════════════════════════════════════════════════
def sex_effect(df_input, feat_cols, label=''):
    """Mann-Whitney Male vs Female."""
    rows = []
    for feat in feat_cols:
        m = df_input[df_input['Gender'] == 'Male'][feat].dropna().values
        f = df_input[df_input['Gender'] == 'Female'][feat].dropna().values
        if len(m) < 3 or len(f) < 3:
            continue
        try:
            _, p = mannwhitneyu(m, f, alternative='two-sided')
        except Exception:
            p = np.nan
        rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                     'n_male': len(m), 'n_female': len(f),
                     'Cohen_d': cohens_d(f, m), 'p': p})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res['BH_q']        = bh_fdr(res['p'].fillna(1).values)
    res['Significant'] = res['BH_q'] < ALPHA
    print(f"  {label}: {res['Significant'].sum()}/{len(res)} sig  (MWU Male vs Female)")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# L6 — HYPERTENSION EFFECT  (Part 2 only)
# ══════════════════════════════════════════════════════════════════════════════
def hypertension_effect(df_input, feat_cols, label=''):
    """Mann-Whitney Hypertension=Yes vs No."""
    df_h = df_input[df_input['Hypertn'].isin(['Yes', 'No'])].copy()
    if len(df_h) < 6:
        print(f"  {label}: insufficient hypertension data (n={len(df_h)})")
        return pd.DataFrame()
    n_y = (df_h['Hypertn'] == 'Yes').sum()
    n_n = (df_h['Hypertn'] == 'No').sum()
    rows = []
    for feat in feat_cols:
        hy = df_h[df_h['Hypertn'] == 'Yes'][feat].dropna().values
        no = df_h[df_h['Hypertn'] == 'No'][feat].dropna().values
        if len(hy) < 3 or len(no) < 3:
            continue
        try:
            _, p = mannwhitneyu(hy, no, alternative='two-sided')
        except Exception:
            p = np.nan
        rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                     'n_HT': len(hy), 'n_noHT': len(no),
                     'Cohen_d': cohens_d(no, hy), 'p': p})
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    res['BH_q']        = bh_fdr(res['p'].fillna(1).values)
    res['Significant'] = res['BH_q'] < ALPHA
    print(f"  {label} (HT={n_y}, no-HT={n_n}): "
          f"{res['Significant'].sum()}/{len(res)} sig")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# L7 — AORTA DIAMETER  (Part 2 only)
# ══════════════════════════════════════════════════════════════════════════════
def diameter_correlation(df_input, feat_cols, label=''):
    """Spearman correlation with aorta diameter."""
    df_d = df_input[df_input['AortaDiam_mm'].notna()].copy()
    if len(df_d) < 5:
        print(f"  {label}: insufficient diameter data (n={len(df_d)})")
        return pd.DataFrame()
    rows = []
    for feat in feat_cols:
        sub = df_d[['AortaDiam_mm', feat]].dropna()
        if len(sub) < 5:
            continue
        r, p = spearmanr(sub['AortaDiam_mm'], sub[feat])
        rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                     'Spearman_r': r, 'p': p, 'n': len(sub)})
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    res['BH_q']        = bh_fdr(res['p'].fillna(1).values)
    res['Significant'] = res['BH_q'] < ALPHA
    print(f"  {label} (n={len(df_d)}): {res['Significant'].sum()}/{len(res)} sig")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: PER-FEATURE BOXPLOTS (primary)
# ══════════════════════════════════════════════════════════════════════════════
def plot_boxplots_per_feature(df_pt, df_nc, res_pt, res_cell,
                               g1, g2, out_dir, prefix='L1'):
    _paper_rc()
    out = Path(out_dir) / 'per_feature'
    out.mkdir(parents=True, exist_ok=True)

    g1_label = GROUP_LABELS[g1]
    g2_label = GROUP_LABELS[g2]
    c1, c2   = GROUP_COLORS[g1], GROUP_COLORS[g2]

    feat_cols = res_pt['Feature'].tolist() if not res_pt.empty else []
    if not feat_cols:
        feat_cols = res_cell['Feature'].tolist() if not res_cell.empty else []

    rng = np.random.default_rng(42)
    bp_kw = dict(patch_artist=True,
                 medianprops=dict(color='#333333', lw=1.2),
                 whiskerprops=dict(lw=0.8, color='#555555'),
                 capprops=dict(lw=0.8, color='#555555'),
                 flierprops=dict(marker='o', markersize=2.5, alpha=0.35,
                                 markeredgecolor='none'))

    for feat in feat_cols:
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.2))
        parts = feat.split('_', 1)
        title_str = parts[1].replace('_', ' ') if len(parts) > 1 else feat.replace('_', ' ')
        fig.suptitle(title_str, fontsize=9)

        for ax, df_use, res, level in [
            (axes[0], df_pt,  res_pt,   'Patient'),
            (axes[1], df_nc,  res_cell, 'Cell'),
        ]:
            vals_g1 = df_use[df_use['Disease'] == g1][feat].dropna().values
            vals_g2 = df_use[df_use['Disease'] == g2][feat].dropna().values

            bp = ax.boxplot([vals_g1, vals_g2], **bp_kw)
            bp['boxes'][0].set_facecolor(c1); bp['boxes'][0].set_alpha(0.65)
            bp['boxes'][1].set_facecolor(c2); bp['boxes'][1].set_alpha(0.65)

            for xi, vals, c in [(1, vals_g1, c1), (2, vals_g2, c2)]:
                jit = rng.uniform(-0.12, 0.12, len(vals))
                ax.scatter(np.full(len(vals), xi) + jit, vals,
                           color=c, alpha=0.5, s=8, zorder=4, edgecolors='none')

            ax.set_xticks([1, 2])
            ax.set_xticklabels([g1_label, g2_label], fontsize=10, fontweight='bold')
            ax.set_title(level, fontsize=10, pad=3, fontweight='bold')
            if ax == axes[0]:
                ax.set_ylabel(title_str, fontsize=11, fontweight='bold')
            _despine(ax)
            _grid(ax, 'y')

            if not res.empty and feat in res['Feature'].values:
                row = res[res['Feature'] == feat].iloc[0]
                d, q = row.get('Cohen_d', np.nan), row.get('BH_q', 1.0)
                stars = sig_stars(q)
                ax.set_title(f"{level}  d={d:.2f}  {stars}", fontsize=10, pad=3, fontweight='bold')
                if q < ALPHA and len(vals_g1) > 0 and len(vals_g2) > 0:
                    ymin, ymax = ax.get_ylim()
                    yr = ymax - ymin
                    by = ymax + yr * 0.03
                    ax.set_ylim(ymin, ymax + yr * 0.18)
                    ax.plot([1, 1, 2, 2], [ymax + yr*0.01, by, by, ymax + yr*0.01],
                            'k-', lw=0.9)
                    ax.text(1.5, by + yr*0.01, stars, ha='center', va='bottom',
                            fontsize=8, fontweight='bold')

        plt.tight_layout()
        fname = feat.replace('/', '_').replace(' ', '_')
        save(fig, out / f'{prefix}_{fname}.png')

    print(f"  Per-feature boxplots saved → {out}/")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: COHEN'S D BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
def plot_cohens_d_bars(res, path, g1_label, g2_label, level='patient'):
    _paper_rc()
    if res.empty:
        return
    sub = res.sort_values('Cohen_d', ascending=True).copy()
    colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in sub['Organelle']]
    n = len(sub)

    fig, ax = plt.subplots(figsize=(7.0, max(4.5, n * 0.36)))
    ax.barh(range(n), sub['Cohen_d'], color=colors,
            edgecolor='white', linewidth=0, height=0.72, alpha=0.85)
    ax.axvline(0, color='#333333', lw=0.8)

    labels = []
    for f in sub['Feature']:
        parts = f.split('_', 1)
        labels.append(parts[1].replace('_', ' ') if len(parts) > 1 else f.replace('_', ' '))
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_xlabel(f"Cohen's d  ({g2_label} − {g1_label})", fontsize=12, fontweight='bold')

    xlim = ax.get_xlim()
    pad  = (xlim[1] - xlim[0]) * 0.04
    star_positions = []
    for i, (_, row) in enumerate(sub.iterrows()):
        if row.get('Significant', False):
            x = row['Cohen_d']
            sx = x + (pad if x >= 0 else -pad)
            star_positions.append(sx)
            ax.text(sx, i, sig_stars(row['BH_q']),
                    va='center', ha='left' if x >= 0 else 'right', fontsize=11)

    # expand xlim so stars are never clipped by the axis boundary
    if star_positions:
        extra = (xlim[1] - xlim[0]) * 0.08
        new_left  = min(xlim[0], min(star_positions)) - extra
        new_right = max(xlim[1], max(star_positions)) + extra
        ax.set_xlim(new_left, new_right)

    patches = [mpatches.Patch(color=c, label=o, alpha=0.85)
               for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, frameon=False, loc='lower right', fontsize=11)
    footer(fig, level)
    _despine(ax)
    _grid(ax, 'x')
    plt.tight_layout()
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: PATIENT vs CELL COMPARISON (dual -log10 q bars)
# ══════════════════════════════════════════════════════════════════════════════
def plot_level_comparison(res_pt, res_cell, path):
    _paper_rc()
    if res_pt.empty or res_cell.empty:
        return
    merged = (res_pt[['Feature', 'BH_q', 'Organelle']]
              .merge(res_cell[['Feature', 'BH_q']], on='Feature', suffixes=('_pt', '_cell')))
    merged['-l10_pt']   = -np.log10(merged['BH_q_pt'].clip(1e-10))
    merged['-l10_cell'] = -np.log10(merged['BH_q_cell'].clip(1e-10))
    merged = merged.sort_values('-l10_pt', ascending=True)
    thr = -np.log10(ALPHA)

    n = len(merged)
    y = np.arange(n)
    fig, ax = plt.subplots(figsize=(5.5, max(3.5, n * 0.28)))
    bw = 0.35
    ax.barh(y - bw/2, merged['-l10_pt'],   height=bw,
            color=LEVEL_COLORS['Patient'], label='Patient', alpha=0.85,
            edgecolor='white', linewidth=0)
    ax.barh(y + bw/2, merged['-l10_cell'], height=bw,
            color=LEVEL_COLORS['Cell'],    label='Cell', alpha=0.85,
            edgecolor='white', linewidth=0)
    ax.axvline(thr, color='#C0392B', ls='--', lw=0.9, label='FDR=0.05')

    labels = []
    for f in merged['Feature']:
        parts = f.split('_', 1)
        labels.append(parts[1].replace('_', ' ') if len(parts) > 1 else f.replace('_', ' '))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_xlabel(r'$-\log_{10}$(BH-FDR $q$)', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, loc='lower right')
    _despine(ax)
    _grid(ax, 'x')
    plt.tight_layout()
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: LMM FOREST PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_lmm_forest(lmm, path):
    _paper_rc()
    if lmm is None or lmm.empty:
        return
    terms = lmm['Term'].unique().tolist()
    max_feats = max((len(lmm[lmm['Term'] == t]) for t in terms), default=6)
    fig, axes = plt.subplots(1, len(terms),
                             figsize=(len(terms) * 2.8, max(3.5, max_feats * 0.28)))
    if len(terms) == 1:
        axes = [axes]

    for ax, term in zip(axes, terms):
        sub = lmm[lmm['Term'] == term].sort_values('Coef')
        colors_feat = ['#C0392B' if s else '#AAAAAA' for s in sub['Significant']]
        y = np.arange(len(sub))
        ax.scatter(sub['Coef'], y, color=colors_feat, zorder=4, s=18)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.plot([row['CI_low'], row['CI_high']], [i, i],
                    color='#C0392B' if row['Significant'] else '#CCCCCC', lw=1.2)
        ax.axvline(0, color='#333333', lw=0.8, ls='--')

        labels = []
        for f in sub['Feature']:
            parts = f.split('_', 1)
            labels.append(parts[1].replace('_', ' ') if len(parts) > 1 else f.replace('_', ' '))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_xlabel('Coefficient (95% CI)', fontsize=12, fontweight='bold')
        n_sig = sub['Significant'].sum()
        ax.set_title(f'{term}  ({n_sig}/{len(sub)} sig)', fontsize=10, pad=4, fontweight='bold')
        _despine(ax)
        _grid(ax, 'x')

    plt.tight_layout()
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# LMM FULL MODEL  (Disease × Collagen + Age + Sex)
# ══════════════════════════════════════════════════════════════════════════════
def _abbrev_feature(feat):
    """Actin_Volume_µm³ → Act_Vol  (short label for heatmap y-axis)"""
    if feat.startswith('Actin_'):     prefix = 'Act'
    elif feat.startswith('Mito_'):    prefix = 'Mit'
    elif feat.startswith('Nucleus_'): prefix = 'Nuc'
    else:                              prefix = feat[:3]
    name = re.sub(r'^(Actin_|Mito_|Nucleus_)', '', feat)
    name = re.sub(r'[_]?(µm[²³]?|ratio|_n)$', '', name)
    abbrevs = {
        'Volume': 'Vol', 'Surface_Area': 'SurfArea',
        'Skeleton_Length': 'SkelLen', 'Total_Network_Length': 'NetLen',
        'Mean_Branch_Length': 'BranchLen', 'Branch_Count': 'BranchCnt',
        'Fragment_Count': 'FragCnt', 'Mean_Fragment_Volume': 'FragVol',
        'Mean_Fragment_Sphericity': 'Spher', 'Min_Fragment_Sphericity': 'MinSpher',
        'Max_Fragment_Sphericity': 'MaxSpher', 'Std_Fragment_Sphericity': 'StdSpher',
        'Sphericity': 'Spher', 'Cyclomatic_Number': 'Cyclomatic',
        'Junction_Count': 'JunctionCnt', 'Mean_Tortuosity': 'Tortuosity',
        'Solidity': 'Solid', 'Circularity': 'Circ', 'Elongation': 'Elong',
        'Flatness': 'Flat', 'Convex_Hull_Volume': 'ConvexHull',
        'Extent': 'Extent', 'Fractional_Anisotropy': 'FA',
        'Major_Axis': 'MajorAx', 'Minor_Axis': 'MinorAx',
        'Intermediate_Axis': 'InterAx',
    }
    return f"{prefix}_{abbrevs.get(name, name)}"


def run_lmm_full(df, g1, g2, feat_cols, label=''):
    """
    Full LMM: Y ~ Disease_bin * Coll_bin + Age + Sex_bin, Subject random intercept.
    Includes Collagen as fixed effect + Disease×Collagen interaction.
    Used to build the heatmap showing all effects simultaneously.
    """
    if not HAS_STATSMODELS:
        return None

    df2 = df[df['Disease'].isin([g1, g2])].copy()
    df2['Disease_bin'] = (df2['Disease'] == g2).astype(float)
    df2['Coll_bin']    = (df2['Collagen_Status'] == 'Collagen').astype(float)
    df2['Sex_bin']     = (df2['Gender'] == 'Male').astype(float)

    print(f"  {label}: full LMM on {len(feat_cols)} features")

    term_map = {
        'Disease':     'Disease_bin',
        'Collagen':    'Coll_bin',
        'Interaction': 'Disease_bin:Coll_bin',
        'Age':         'Age',
        'Sex':         'Sex_bin',
    }

    records = []
    for feat in feat_cols:
        needed = ['Subject', 'Disease_bin', 'Coll_bin', 'Age', 'Sex_bin', feat]
        sub = df2[needed].rename(columns={feat: 'Y'}).dropna()
        if sub['Subject'].nunique() < 5:
            continue
        try:
            res = smf.mixedlm(
                'Y ~ Disease_bin * Coll_bin + Age + Sex_bin',
                data=sub, groups=sub['Subject']
            ).fit(reml=True, method='lbfgs')
            for term_label, param in term_map.items():
                if param in res.params:
                    records.append({
                        'Feature':   feat,
                        'Organelle': get_organelle(feat),
                        'Term':      term_label,
                        'Coef':      res.params[param],
                        'p':         res.pvalues[param],
                    })
        except Exception:
            pass

    if not records:
        print(f"  {label}: no results")
        return None

    lmm = pd.DataFrame(records)
    for term in lmm['Term'].unique():
        mask = lmm['Term'] == term
        lmm.loc[mask, 'BH_q'] = bh_fdr(lmm.loc[mask, 'p'].values)
    lmm['Significant'] = lmm['BH_q'] < ALPHA

    print(f"  {label}: {lmm['Significant'].sum()} significant term-feature pairs")
    for term in lmm['Term'].unique():
        sig = lmm[(lmm['Term'] == term) & lmm['Significant']]['Feature'].tolist()
        if sig:
            print(f"    {term:15s}: {len(sig)} — {sig[:3]}{'...' if len(sig)>3 else ''}")
    return lmm


def plot_lmm_heatmap(lmm, path):
    _paper_rc()
    if lmm is None or lmm.empty:
        return

    term_order = ['Disease', 'Collagen', 'Interaction', 'Age', 'Sex']
    term_order = [t for t in term_order if t in lmm['Term'].unique()]

    lmm = lmm.copy()
    lmm['score'] = -np.log10(lmm['BH_q'].clip(1e-10)) * np.sign(lmm['Coef'])

    feat_order = (lmm.drop_duplicates('Feature')
                     .sort_values(['Organelle', 'Feature'])['Feature'].tolist())

    pivot = (lmm.pivot(index='Feature', columns='Term', values='score')
                .reindex(index=feat_order, columns=term_order).fillna(0))
    sig_pv = (lmm.pivot(index='Feature', columns='Term', values='Significant')
                 .reindex(index=feat_order, columns=term_order).fillna(False))

    def _full_label(feat):
        for prefix in ('Actin_', 'Mito_', 'Nucleus_'):
            if feat.startswith(prefix):
                feat = feat[len(prefix):]
                break
        return feat.replace('_', ' ')

    x_labels = [_full_label(f) for f in feat_order]
    n_feat, n_term = len(feat_order), len(term_order)
    vmax = max(3.0, float(np.abs(pivot.values).max()))

    # Horizontal layout: terms = rows (y), features = columns (x)
    data = pivot.T.values   # shape: (n_term, n_feat)
    sig  = sig_pv.T.values

    fig, ax = plt.subplots(figsize=(max(8.0, n_feat * 0.42), max(2.5, n_term * 1.1)))
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    ax.set_yticks(np.arange(n_term))
    ax.set_yticklabels(term_order, fontsize=12, fontfamily='Arial', fontweight='bold')
    ax.set_xticks(np.arange(n_feat))
    ax.set_xticklabels(x_labels, fontsize=10, fontfamily='Arial', fontweight='bold',
                        rotation=45, ha='right', rotation_mode='anchor')

    ax.set_xticks(np.arange(-0.5, n_feat, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_term, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(n_term):
        for j in range(n_feat):
            if sig[i, j]:
                ax.text(j, i, '*', ha='center', va='center', fontsize=13,
                        fontfamily='Arial', color='black')

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r'$-\log_{10}(p_\mathrm{FDR})\times\mathrm{sign(coef)}$',
                   fontsize=10, fontfamily='Arial', fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    thr = -np.log10(ALPHA)
    for sgn in [1, -1]:
        cbar.ax.axhline(sgn * thr, color='black', ls='--', lw=0.8)

    plt.tight_layout()
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: LMM CATERPILLAR (random effects per subject)
# ══════════════════════════════════════════════════════════════════════════════
def plot_lmm_caterpillar(df, g1, g2, feat_col, path, title=None):
    """
    Caterpillar plot of LMM random effects for a single feature.
    Tries a random-slope model (Disease_bin | Subject) first;
    falls back to random-intercept only.
    Two panels: Random Slopes (top) + Random Intercepts (bottom).
    """
    if not HAS_STATSMODELS:
        return

    df2 = df[df['Disease'].isin([g1, g2])].copy()
    df2['Disease_bin'] = (df2['Disease'] == g2).astype(float)
    df2['Sex_bin']     = (df2['Gender']  == 'Male').astype(float)

    needed = ['Subject', 'Disease_bin', 'Age', 'Sex_bin', feat_col]
    sub = df2[needed].rename(columns={feat_col: 'Y'}).dropna()

    if sub['Subject'].nunique() < 5:
        print(f"  Caterpillar: too few subjects ({sub['Subject'].nunique()}) — skipped")
        return

    # ── fit model ─────────────────────────────────────────────────────────────
    has_slope = False
    res = None
    try:
        res = smf.mixedlm(
            'Y ~ Disease_bin + Age + Sex_bin',
            data=sub, groups=sub['Subject'],
            re_formula='~Disease_bin'
        ).fit(reml=True, method='lbfgs')
        has_slope = True
        print(f"  Caterpillar: random-slope model for {feat_col}")
    except Exception:
        pass

    if res is None:
        try:
            res = smf.mixedlm(
                'Y ~ Disease_bin + Age + Sex_bin',
                data=sub, groups=sub['Subject']
            ).fit(reml=True, method='lbfgs')
            print(f"  Caterpillar: random-intercept model for {feat_col}")
        except Exception as e:
            print(f"  Caterpillar: model failed — {e}")
            return

    # ── extract BLUPs and posterior SEs ───────────────────────────────────────
    re_blups = res.random_effects          # {subject: pd.Series}
    subjects = sorted(re_blups.keys())
    n_subj   = len(subjects)

    int_vals = np.array([re_blups[s].iloc[0] for s in subjects])
    if has_slope and all(len(re_blups[s]) > 1 for s in subjects):
        slp_vals = np.array([re_blups[s].iloc[1] for s in subjects])
    else:
        has_slope = False

    try:
        re_cov   = res.random_effects_cov  # {subject: pd.DataFrame}
        int_ses  = np.array([np.sqrt(re_cov[s].iloc[0, 0]) for s in subjects])
        slp_ses  = (np.array([np.sqrt(re_cov[s].iloc[1, 1]) for s in subjects])
                    if has_slope else None)
    except Exception:
        # fallback: uniform SE from population covariance diagonal
        cov_re  = res.cov_re.values
        int_ses = np.full(n_subj, np.sqrt(cov_re[0, 0]))
        slp_ses = (np.full(n_subj, np.sqrt(cov_re[1, 1]))
                   if has_slope and cov_re.shape[0] > 1 else None)
        if has_slope and slp_ses is None:
            has_slope = False

    # ── build panels list ─────────────────────────────────────────────────────
    panels = []
    if has_slope:
        panels.append((slp_vals, slp_ses, 'Random Slopes',     'Slope'))
    panels.append(    (int_vals, int_ses, 'Random Intercepts', 'Intercept'))

    _paper_rc()
    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(4.5, max(3.0, n_subj * 0.28) * n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax, (vals, ses, panel_title, xlabel) in zip(axes, panels):
        order = np.argsort(vals)
        for rank, idx in enumerate(order):
            ax.errorbar(vals[idx], rank, xerr=1.96 * ses[idx],
                        fmt='o', color='#333333',
                        markerfacecolor='white', markeredgecolor='#333333',
                        markeredgewidth=1.2, markersize=6,
                        elinewidth=1.0, capsize=0, zorder=3)
        ax.axvline(0, color='#C0392B', ls='--', lw=0.9)
        ax.set_yticks(range(n_subj))
        ax.set_yticklabels([str(subjects[i]) for i in order], fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_title(panel_title, fontsize=10, pad=3, fontweight='bold')
        _despine(ax)
        _grid(ax, 'x')

    if title:
        fig.suptitle(title, fontsize=9)

    plt.tight_layout()
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: COLLAGEN RESCUE BARS
# ══════════════════════════════════════════════════════════════════════════════
def plot_collagen_bars(pt_res, cell_res, g1, g2, path):
    _paper_rc()
    groups = [g1, g2]
    levels = [('Patient', pt_res), ('Cell', cell_res)]

    if pt_res.empty and cell_res.empty:
        return

    n_per_panel = max(
        *(len(res[res['Disease'] == d]) for res in [pt_res, cell_res]
          if not res.empty for d in groups), 10)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, max(5.0, n_per_panel * 0.30 * 2)))

    for row_i, (level_label, res) in enumerate(levels):
        for col_i, dis in enumerate(groups):
            ax  = axes[row_i][col_i]
            sub = res[res['Disease'] == dis].sort_values('Cohen_d', ascending=True)
            if sub.empty:
                ax.set_visible(False)
                continue
            org_colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in sub['Organelle']]
            ax.barh(range(len(sub)), sub['Cohen_d'], color=org_colors,
                    edgecolor='white', linewidth=0, height=0.72, alpha=0.85)
            ax.axvline(0, color='#333333', lw=0.8)
            ax.set_yticks(range(len(sub)))

            labels = []
            for f in sub['Feature']:
                parts = f.split('_', 1)
                labels.append(parts[1].replace('_', ' ') if len(parts) > 1 else f.replace('_', ' '))
            if col_i == 0:
                ax.set_yticklabels(labels, fontsize=9, fontweight='bold')
            else:
                ax.set_yticklabels([])
            if row_i == 1:
                ax.set_xlabel("Cohen's d  (NoCol → +Col)", fontsize=12, fontweight='bold')
            n_sig = sub['Significant'].sum()
            ax.set_title(f'{GROUP_LABELS[dis]} — {level_label}\n{n_sig}/{len(sub)} sig',
                         fontsize=10, pad=4, fontweight='bold')

            xlim = ax.get_xlim()
            pad  = (xlim[1] - xlim[0]) * 0.025
            for i, (_, row) in enumerate(sub.iterrows()):
                if row.get('Significant', False):
                    x = row['Cohen_d']
                    ax.text(x + (pad if x >= 0 else -pad), i, sig_stars(row['BH_q']),
                            va='center', ha='left' if x >= 0 else 'right', fontsize=7.5)
            _despine(ax)
            _grid(ax, 'x')

    patches = [mpatches.Patch(color=c, label=o, alpha=0.85)
               for o, c in ORGANELLE_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: SEX EFFECT BARS
# ══════════════════════════════════════════════════════════════════════════════
def plot_sex_bars(res_pt, res_cell, path):
    _paper_rc()
    if res_pt.empty and res_cell.empty:
        return
    levels  = [('Patient', res_pt, 'patient'), ('Cell', res_cell, 'cell')]
    n_feats = len(res_pt) if not res_pt.empty else len(res_cell) if not res_cell.empty else 15
    fig, axes = plt.subplots(1, 2, figsize=(7.5, max(3.5, n_feats * 0.28)))
    for ax, (label, res, lvl) in zip(axes, levels):
        if res.empty:
            ax.set_visible(False)
            continue
        sub    = res.sort_values('Cohen_d', ascending=True)
        colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in sub['Organelle']]
        ax.barh(range(len(sub)), sub['Cohen_d'], color=colors,
                edgecolor='white', linewidth=0, height=0.72, alpha=0.85)
        ax.axvline(0, color='#333333', lw=0.8)
        ax.set_yticks(range(len(sub)))
        labels = []
        for f in sub['Feature']:
            parts = f.split('_', 1)
            labels.append(parts[1].replace('_', ' ') if len(parts) > 1 else f.replace('_', ' '))
        ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_xlabel("Cohen's d  (Female → Male)", fontsize=12, fontweight='bold')
        n_sig = sub['Significant'].sum()
        ax.set_title(f'{label}  ({n_sig}/{len(sub)} sig)', fontsize=10, pad=4, fontweight='bold')
        footer(fig, lvl)
        _despine(ax)
        _grid(ax, 'x')
    plt.tight_layout()
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: SUMMARY HEATMAP + PCA
# ══════════════════════════════════════════════════════════════════════════════
def plot_summary(df_pt, feat_cols, res_pt, g1, g2, out_dir):
    """Z-score heatmap (top features) + PCA scatter at patient level."""
    _paper_rc()
    out = Path(out_dir)
    sig_feats = (res_pt.loc[res_pt['Significant'], 'Feature'].tolist()
                 if not res_pt.empty else [])
    feats = sig_feats if sig_feats else feat_cols[:15]

    # ── Heatmap ───────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_sc   = pd.DataFrame(scaler.fit_transform(df_pt[feat_cols].fillna(0)),
                           columns=feat_cols, index=df_pt.index)
    hm = {GROUP_LABELS[g]: X_sc[df_pt['Disease'] == g][feats].mean()
          for g in [g1, g2]}
    hm_df = pd.DataFrame(hm, index=feats)
    row_labels = [f.replace('_', ' ') + (' *' if f in set(sig_feats) else '')
                  for f in feats]
    hm_df.index = row_labels

    n_feat = len(feats)
    fig, ax = plt.subplots(figsize=(max(3.5, len([g1, g2]) * 0.9),
                                    max(4.0, n_feat * 0.28)))
    vmax = max(1.0, float(hm_df.abs().values.max()))
    im = ax.imshow(hm_df.values, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(hm_df.columns)))
    ax.set_xticklabels(hm_df.columns, fontsize=10, fontweight='bold')
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(row_labels, fontsize=10, fontweight='bold')
    ax.set_xticks(np.arange(-0.5, len(hm_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_feat, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)
    for i in range(n_feat):
        for j in range(len(hm_df.columns)):
            ax.text(j, i, f'{hm_df.iloc[i, j]:.2f}',
                    ha='center', va='center', fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label('Z-score (patient mean)', fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title(f'* = FDR-sig (patient)', fontsize=8, pad=4)
    plt.tight_layout()
    save(fig, out / 'SUMMARY_heatmap.png')

    # ── PCA ───────────────────────────────────────────────────────────────────
    X  = df_pt[feat_cols].fillna(0).values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(2, Xs.shape[1]), random_state=42)
    Xp  = pca.fit_transform(Xs)
    ev  = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    for g in [g1, g2]:
        mask = df_pt['Disease'].values == g
        ax.scatter(Xp[mask, 0], Xp[mask, 1], c=GROUP_COLORS[g],
                   label=GROUP_LABELS[g], s=30, alpha=0.8,
                   edgecolors='white', lw=0.4)
        ctr = Xp[mask].mean(axis=0)
        ax.scatter(*ctr, c=GROUP_COLORS[g], s=100, marker='D',
                   edgecolors='black', lw=1.2, zorder=6)
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=12, fontweight='bold')
    ax.legend(frameon=False)
    _despine(ax)
    plt.tight_layout()
    save(fig, out / 'SUMMARY_pca.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: AORTA DIAMETER — STYLED SPEARMAN BARS
# ══════════════════════════════════════════════════════════════════════════════
def plot_diameter_spearman(res, path):
    """Spearman ρ bar chart: red = negative, blue = positive; faded = ns."""
    _paper_rc()
    if res.empty:
        return

    org_rank = {'Actin': 0, 'Mitochondria': 1, 'Nucleus': 2, 'Other': 3}
    sub = res.copy()
    sub['_org_rank'] = sub['Organelle'].map(org_rank).fillna(3)
    sub = sub.sort_values(['_org_rank', 'Feature']).reset_index(drop=True)
    n = len(sub)

    fig, ax = plt.subplots(figsize=(5.5, max(3.5, n * 0.28)))

    for i, row in sub.iterrows():
        rho = row['Spearman_r']
        sig = row.get('Significant', False)
        alp = 0.88 if sig else 0.22
        col = '#C0392B' if rho < 0 else '#2471A3'
        ax.barh(i, rho, color=col, alpha=alp, height=0.72, edgecolor='none')

    ax.axvline(0, color='#555', ls='--', lw=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks(np.arange(-1.0, 1.25, 0.25))

    labels = []
    for f in sub['Feature']:
        parts = f.split('_', 1)
        labels.append(parts[1].replace('_', ' ') if len(parts) > 1 else f.replace('_', ' '))
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_xlabel(r'Spearman $\rho$  (feature vs. aortic diameter)', fontsize=12, fontweight='bold')

    prev_org = None
    for i, row in sub.iterrows():
        if prev_org is not None and row['Organelle'] != prev_org:
            ax.axhline(i - 0.5, color='#CCCCCC', lw=0.6)
        prev_org = row['Organelle']

    for i, row in sub.iterrows():
        if row.get('Significant', False):
            x = row['Spearman_r']
            col = '#C0392B' if x < 0 else '#2471A3'
            ax.text(x + (0.03 if x >= 0 else -0.03), i, sig_stars(row['BH_q']),
                    va='center', ha='left' if x >= 0 else 'right',
                    fontsize=7.5, fontweight='bold', color=col)

    solid_p = mpatches.Patch(color='#777', alpha=0.88, label='FDR < 0.05')
    faded_p = mpatches.Patch(color='#777', alpha=0.22, label='ns')
    ax.legend(handles=[solid_p, faded_p], frameon=False, loc='lower right', fontsize=7.5)
    _despine(ax)
    _grid(ax, 'x')
    plt.tight_layout()
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: SEX EFFECT HEATMAP (Cohen's d per group)
# ══════════════════════════════════════════════════════════════════════════════
def plot_sex_heatmap(df_pt, feat_cols, g1, g2, path):
    """Heatmap of Cohen's d (Male − Female) within each group."""
    _paper_rc()
    groups = [g1, g2]
    d_cols, q_cols, col_labels = [], [], []

    for g in groups:
        sub_g = df_pt[df_pt['Disease'] == g]
        nm = int((sub_g['Gender'] == 'Male').sum())
        nf = int((sub_g['Gender'] == 'Female').sum())
        col_labels.append(f"{GROUP_LABELS[g]}\n(M={nm}, F={nf})")

        ds, pvals = [], []
        for feat in feat_cols:
            m = sub_g[sub_g['Gender'] == 'Male'][feat].dropna().values
            f = sub_g[sub_g['Gender'] == 'Female'][feat].dropna().values
            if len(m) >= 2 and len(f) >= 2:
                d = cohens_d(f, m)
                try:
                    _, p = mannwhitneyu(m, f, alternative='two-sided')
                except Exception:
                    p = np.nan
            else:
                d, p = 0.0, np.nan
            ds.append(d)
            pvals.append(p if not np.isnan(p) else 1.0)

        qs = bh_fdr(np.array(pvals))
        d_cols.append(ds)
        q_cols.append(qs)

    d_df   = pd.DataFrame(np.column_stack(d_cols), index=feat_cols, columns=col_labels)
    q_df   = pd.DataFrame(np.column_stack(q_cols), index=feat_cols, columns=col_labels)
    sig_df = q_df < ALPHA

    order  = d_df.abs().mean(axis=1).sort_values().index
    d_df   = d_df.loc[order]
    q_df   = q_df.loc[order]
    sig_df = sig_df.loc[order]

    y_labels = [_abbrev_feature(f) for f in d_df.index]
    n_feat, n_col = len(d_df), len(groups)
    vmax = max(2.0, float(d_df.abs().values.max()))

    fig, ax = plt.subplots(figsize=(max(3.5, n_col * 0.9), max(4.0, n_feat * 0.28)))
    im = ax.imshow(d_df.values, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_col))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight='bold')
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(y_labels, fontsize=10, fontweight='bold')

    ax.set_xticks(np.arange(-0.5, n_col, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_feat, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(n_feat):
        for j in range(n_col):
            if sig_df.iloc[i, j]:
                ax.text(j, i, sig_stars(q_df.iloc[i, j]),
                        ha='center', va='center', fontsize=8, color='black')

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Cohen's d  (Male − Female)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: PER-FEATURE 4-GROUP BOXPLOTS (NoCollagen vs +Collagen × 2 groups)
# ══════════════════════════════════════════════════════════════════════════════
def plot_boxplots_collagen_per_feature(df, feat_cols, g1, g2, out_dir, prefix='col',
                                       res_pt=None, res_cell=None):
    """
    For every feature: two panels (patient-level left, cell-level right).
    Each panel has 4 groups:
      g1 NoCollagen | g1 +Collagen | g2 NoCollagen | g2 +Collagen
    Collagen rescue p-value (paired Wilcoxon patient / MWU cell) annotated.
    If res_pt/res_cell provided, L1 pairwise significance also annotated.
    Saves one PNG per feature into out_dir/per_feature_collagen/.
    """
    from scipy.stats import wilcoxon as _wilcoxon

    out = Path(out_dir) / 'per_feature_collagen'
    out.mkdir(parents=True, exist_ok=True)

    g1_label = GROUP_LABELS[g1]
    g2_label = GROUP_LABELS[g2]
    c1 = GROUP_COLORS[g1]
    c2 = GROUP_COLORS[g2]

    # lighter shades for +Collagen
    def lighten(hex_color, factor=0.45):
        import matplotlib.colors as mc
        rgb = mc.to_rgb(hex_color)
        return tuple(min(1.0, v + (1 - v) * factor) for v in rgb)

    c1_col = lighten(c1)
    c2_col = lighten(c2)

    # patient-level aggregation per Subject × Collagen_Status
    pt_agg = (df.groupby(['Subject', 'Disease', 'Collagen_Status'])[feat_cols]
              .mean().reset_index())

    groups_def = [
        (g1, 'NoCollagen', f'{g1_label}\nNo Coll', c1,     False),
        (g1, 'Collagen',   f'{g1_label}\n+Coll',   c1_col, True),
        (g2, 'NoCollagen', f'{g2_label}\nNo Coll', c2,     False),
        (g2, 'Collagen',   f'{g2_label}\n+Coll',   c2_col, True),
    ]

    rng = np.random.default_rng(42)
    for feat in feat_cols:
        _paper_rc()
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.5))
        feat_title = feat.split('_', 1)[1].replace('_', ' ') if '_' in feat else feat.replace('_', ' ')
        fig.suptitle(feat_title, fontsize=9, y=1.01)

        for ax, df_src, level_label, is_cell in [
            (axes[0], pt_agg, 'Patient', False),
            (axes[1], df,     'Cell',    True),
        ]:
            data_groups = []
            for dis, col_status, xlbl, color, hatched in groups_def:
                vals = (df_src[(df_src['Disease'] == dis) &
                               (df_src['Collagen_Status'] == col_status)][feat]
                        .dropna().values)
                data_groups.append((vals, xlbl, color, hatched))

            positions = [1, 2, 3.4, 4.4]
            bp = ax.boxplot([d[0] for d in data_groups],
                            positions=positions, patch_artist=True,
                            medianprops=dict(color='black', lw=1.2),
                            whiskerprops=dict(lw=0.8),
                            capprops=dict(lw=0.8),
                            flierprops=dict(marker='o', markersize=2, alpha=0.4),
                            widths=0.55)

            for i, (bx, (vals, xlbl, color, hatched)) in enumerate(
                    zip(bp['boxes'], data_groups)):
                bx.set_facecolor(color)
                bx.set_alpha(0.72)
                if hatched:
                    bx.set_hatch('///')
                jit = rng.uniform(-0.12, 0.12, len(vals))
                ax.scatter(np.full(len(vals), positions[i]) + jit, vals,
                           color=color, alpha=0.55, s=8, zorder=4, edgecolors='none')

            ax.set_xticks(positions)
            ax.set_xticklabels([d[1] for d in data_groups], fontsize=9, fontweight='bold')
            ax.set_title(level_label, fontsize=10, pad=3, fontweight='bold')
            _despine(ax)
            _grid(ax, 'y')

            p_vals = {}
            for dis, pos_nc, pos_col in [(g1, positions[0], positions[1]),
                                          (g2, positions[2], positions[3])]:
                if is_cell:
                    d1 = df_src[(df_src['Disease'] == dis) &
                                (df_src['Collagen_Status'] == 'NoCollagen')][feat].dropna().values
                    d2 = df_src[(df_src['Disease'] == dis) &
                                (df_src['Collagen_Status'] == 'Collagen')][feat].dropna().values
                    p = np.nan
                    if len(d1) >= 3 and len(d2) >= 3:
                        try:
                            _, p = mannwhitneyu(d1, d2, alternative='two-sided')
                        except Exception:
                            pass
                else:
                    nc_s  = pt_agg[(pt_agg['Disease'] == dis) &
                                   (pt_agg['Collagen_Status'] == 'NoCollagen')].set_index('Subject')[feat]
                    col_s = pt_agg[(pt_agg['Disease'] == dis) &
                                   (pt_agg['Collagen_Status'] == 'Collagen')].set_index('Subject')[feat]
                    common = nc_s.index.intersection(col_s.index)
                    p = np.nan
                    if len(common) >= 4:
                        try:
                            _, p = _wilcoxon(col_s.loc[common].values - nc_s.loc[common].values)
                        except Exception:
                            pass
                p_vals[(dis, pos_nc, pos_col)] = p

            sig_pairs = [(pos_nc, pos_col, p)
                         for (_, pos_nc, pos_col), p in p_vals.items()
                         if not np.isnan(p) and p < ALPHA]

            # L1 pairwise significance (g1 vs g2, NoColl condition)
            l1_q = None
            res_src = res_cell if is_cell else res_pt
            if res_src is not None:
                row = res_src[res_src['Feature'] == feat]
                if not row.empty and bool(row.iloc[0]['Significant']):
                    l1_q = float(row.iloc[0]['BH_q'])

            n_rescue = len(sig_pairs)
            n_bars   = n_rescue + (1 if l1_q is not None else 0)
            if n_bars:
                ymin, ymax = ax.get_ylim()
                yr = ymax - ymin
                ax.set_ylim(ymin, ymax + yr * 0.15 * n_bars)
                for k, (pos_nc, pos_col, p) in enumerate(sig_pairs):
                    b_y = ymax + yr * (0.03 + 0.13 * k)
                    ax.plot([pos_nc, pos_nc, pos_col, pos_col],
                            [ymax + yr * 0.01, b_y, b_y, ymax + yr * 0.01],
                            'k-', lw=0.9)
                    ax.text((pos_nc + pos_col) / 2, b_y + yr * 0.01,
                            sig_stars(p), ha='center', va='bottom',
                            fontsize=8, fontweight='bold')
                if l1_q is not None:
                    b_y = ymax + yr * (0.03 + 0.13 * n_rescue)
                    ax.plot([positions[0], positions[0], positions[2], positions[2]],
                            [ymax + yr * 0.01, b_y, b_y, ymax + yr * 0.01],
                            'k-', lw=0.9)
                    ax.text((positions[0] + positions[2]) / 2, b_y + yr * 0.01,
                            sig_stars(l1_q), ha='center', va='bottom',
                            fontsize=8, fontweight='bold', color='#1a1a1a')

        patches = [
            mpatches.Patch(facecolor=c1,     label=f'{g1_label} NoColl'),
            mpatches.Patch(facecolor=c1_col, label=f'{g1_label} +Coll', hatch='///'),
            mpatches.Patch(facecolor=c2,     label=f'{g2_label} NoColl'),
            mpatches.Patch(facecolor=c2_col, label=f'{g2_label} +Coll', hatch='///'),
        ]
        fig.legend(handles=patches, loc='lower center', ncol=4,
                   frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        fname = feat.replace('/', '_').replace(' ', '_')
        save(fig, out / f'{prefix}_{fname}.png')

    print(f"  Collagen per-feature boxplots saved → {out}/")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: RESCUE COMPARISON — g1 NoCol | g1 +Col | g2 NoCol | g2 +Col
# ══════════════════════════════════════════════════════════════════════════════
def plot_boxplots_rescue_per_feature(df_pt, df_nc, df, feat_cols,
                                     res_pt, res_cell, g1, g2,
                                     out_dir, prefix='rescue'):
    """
    Per-feature boxplots with 4 groups per panel:
      g1 NoCol | g1 +Col | g2 NoCol | g2 +Col
    Left panel = patient-level, right panel = cell-level.
    Significance brackets (only drawn when significant):
      - g1 NoCol vs g2 NoCol : disease effect  (BH-q from res_pt / res_cell)
      - g1 NoCol vs g1 +Col  : g1 rescue       (Wilcoxon patient / MWU cell)
      - g2 NoCol vs g2 +Col  : g2 rescue       (Wilcoxon patient / MWU cell)
    Disease bracket is placed highest; rescue brackets stacked below it.
    """
    from scipy.stats import wilcoxon as _wilcoxon
    import matplotlib.colors as mc

    out = Path(out_dir) / 'per_feature_rescue'
    out.mkdir(parents=True, exist_ok=True)

    g1_label = GROUP_LABELS[g1]
    g2_label = GROUP_LABELS[g2]
    c1 = GROUP_COLORS[g1]
    c2 = GROUP_COLORS[g2]

    def lighten(hex_color, factor=0.45):
        rgb = mc.to_rgb(hex_color)
        return tuple(min(1.0, v + (1 - v) * factor) for v in rgb)

    c1_col = lighten(c1)
    c2_col = lighten(c2)

    # patient-level per Subject × Disease × Collagen_Status
    pt_agg = (df.groupby(['Subject', 'Disease', 'Collagen_Status'])[feat_cols]
              .mean().reset_index())

    rng = np.random.default_rng(42)

    # positions: gap between g1 pair and g2 pair
    positions  = [1, 2, 3.4, 4.4]
    colors_box = [c1, c1_col, c2, c2_col]
    hatches    = [None, '///', None, '///']

    def _rescue_p(is_cell, dis, v_nc, v_col):
        if is_cell:
            if len(v_nc) >= 3 and len(v_col) >= 3:
                try:
                    _, p = mannwhitneyu(v_nc, v_col, alternative='two-sided')
                    return p
                except Exception:
                    pass
        else:
            nc_s  = pt_agg[(pt_agg['Disease'] == dis) &
                            (pt_agg['Collagen_Status'] == 'NoCollagen')].set_index('Subject')[feat]
            col_s = pt_agg[(pt_agg['Disease'] == dis) &
                            (pt_agg['Collagen_Status'] == 'Collagen')].set_index('Subject')[feat]
            common = nc_s.index.intersection(col_s.index)
            if len(common) >= 4:
                try:
                    _, p = _wilcoxon(col_s.loc[common].values - nc_s.loc[common].values)
                    return p
                except Exception:
                    pass
        return np.nan

    for feat in feat_cols:
        _paper_rc()
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.5))
        feat_title = feat.split('_', 1)[1].replace('_', ' ') if '_' in feat else feat.replace('_', ' ')
        fig.suptitle(feat_title, fontsize=9, y=1.01)

        for ax, is_cell, level_label, res in [
            (axes[0], False, 'Patient', res_pt),
            (axes[1], True,  'Cell',   res_cell),
        ]:
            if is_cell:
                v_g1_nc  = df_nc[df_nc['Disease'] == g1][feat].dropna().values
                v_g1_col = (df[(df['Disease'] == g1) &
                               (df['Collagen_Status'] == 'Collagen')][feat].dropna().values)
                v_g2_nc  = df_nc[df_nc['Disease'] == g2][feat].dropna().values
                v_g2_col = (df[(df['Disease'] == g2) &
                               (df['Collagen_Status'] == 'Collagen')][feat].dropna().values)
            else:
                v_g1_nc  = df_pt[df_pt['Disease'] == g1][feat].dropna().values
                v_g1_col = (pt_agg[(pt_agg['Disease'] == g1) &
                                   (pt_agg['Collagen_Status'] == 'Collagen')][feat].dropna().values)
                v_g2_nc  = df_pt[df_pt['Disease'] == g2][feat].dropna().values
                v_g2_col = (pt_agg[(pt_agg['Disease'] == g2) &
                                   (pt_agg['Collagen_Status'] == 'Collagen')][feat].dropna().values)

            data    = [v_g1_nc, v_g1_col, v_g2_nc, v_g2_col]
            xlabels = [f'{g1_label}\nNoCol', f'{g1_label}\n+Col',
                       f'{g2_label}\nNoCol', f'{g2_label}\n+Col']

            bp = ax.boxplot(data, positions=positions, patch_artist=True,
                            medianprops=dict(color='black', lw=1.2),
                            whiskerprops=dict(lw=0.8),
                            capprops=dict(lw=0.8),
                            flierprops=dict(marker='o', markersize=2, alpha=0.4),
                            widths=0.55)
            for bx, color, hatch in zip(bp['boxes'], colors_box, hatches):
                bx.set_facecolor(color)
                bx.set_alpha(0.72)
                if hatch:
                    bx.set_hatch(hatch)

            for pos, vals, color in zip(positions, data, colors_box):
                jit = rng.uniform(-0.12, 0.12, len(vals))
                ax.scatter(np.full(len(vals), pos) + jit, vals,
                           color=color, alpha=0.55, s=18, zorder=4, edgecolors='none')

            ax.set_xticks(positions)
            ax.set_xticklabels(xlabels, fontsize=9, fontweight='bold')
            ax.set_title(level_label, fontsize=10, pad=3, fontweight='bold')
            ax.set_xlim(0.3, 4.9)
            _despine(ax)
            _grid(ax, 'y')

            dis_q = np.nan
            if res is not None and not res.empty and feat in res['Feature'].values:
                dis_q = res.loc[res['Feature'] == feat, 'BH_q'].iloc[0]

            p_g1_rescue = _rescue_p(is_cell, g1, v_g1_nc, v_g1_col)
            p_g2_rescue = _rescue_p(is_cell, g2, v_g2_nc, v_g2_col)

            rescue_brackets = []
            if not np.isnan(p_g1_rescue) and p_g1_rescue < ALPHA:
                rescue_brackets.append((positions[0], positions[1], sig_stars(p_g1_rescue)))
            if not np.isnan(p_g2_rescue) and p_g2_rescue < ALPHA:
                rescue_brackets.append((positions[2], positions[3], sig_stars(p_g2_rescue)))

            dis_bracket = []
            if not np.isnan(dis_q) and dis_q < ALPHA:
                dis_bracket = [(positions[0], positions[2], sig_stars(dis_q))]

            all_brackets = rescue_brackets + dis_bracket

            if all_brackets:
                ymin, ymax = ax.get_ylim()
                yr = ymax - ymin
                ax.set_ylim(ymin, ymax + yr * 0.15 * len(all_brackets))
                for k, (x1, x2, stars) in enumerate(all_brackets):
                    b_y = ymax + yr * (0.03 + 0.13 * k)
                    ax.plot([x1, x1, x2, x2],
                            [ymax + yr * 0.01, b_y, b_y, ymax + yr * 0.01],
                            'k-', lw=0.9)
                    ax.text((x1 + x2) / 2, b_y + yr * 0.01, stars,
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        patches = [
            mpatches.Patch(facecolor=c1,     label=f'{g1_label} NoCol'),
            mpatches.Patch(facecolor=c1_col, label=f'{g1_label} +Col', hatch='///'),
            mpatches.Patch(facecolor=c2,     label=f'{g2_label} NoCol'),
            mpatches.Patch(facecolor=c2_col, label=f'{g2_label} +Col', hatch='///'),
        ]
        fig.legend(handles=patches, loc='lower center', ncol=4,
                   frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fname = feat.replace('/', '_').replace(' ', '_')
        save(fig, out / f'{prefix}_{fname}.png')

    print(f"  Rescue comparison boxplots saved → {out}/")
