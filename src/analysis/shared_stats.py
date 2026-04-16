"""
Shared Statistical Engine for Thesis Analysis
==============================================
Two-group pairwise analysis (patient-level primary + cell-level exploratory).

Import this module in final_thesis_part1/run.py and final_thesis_part2/run.py.
"""

import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
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
    'Nucleus':      '#4C9BE8',
    'Actin':        '#E74C3C',
    'Mitochondria': '#2ECC71',
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
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def footer(fig, level='patient'):
    note = ('Patient-level (BH-FDR)' if level == 'patient'
            else 'Cell-level — exploratory (BH-FDR)')
    fig.text(0.98, 0.005,
             f'* p<0.05  ** p<0.01  *** p<0.001  ({note})',
             ha='right', fontsize=9, style='italic', color='#555')


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
    """
    For every feature: one figure with two panels.
      Left:  patient-level boxplot with individual points
      Right: cell-level boxplot with individual points
    Annotated with Cohen's d and BH-FDR q-value.
    Saves one PNG per feature into out_dir/per_feature/.
    """
    out = Path(out_dir) / 'per_feature'
    out.mkdir(parents=True, exist_ok=True)

    g1_label = GROUP_LABELS[g1]
    g2_label = GROUP_LABELS[g2]
    c1       = GROUP_COLORS[g1]
    c2       = GROUP_COLORS[g2]

    feat_cols = res_pt['Feature'].tolist() if not res_pt.empty else []
    if not feat_cols:
        feat_cols = res_cell['Feature'].tolist() if not res_cell.empty else []

    for feat in feat_cols:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(feat.replace('_', ' '), fontsize=13, fontweight='bold')

        for ax, df_use, res, level in [
            (axes[0], df_pt,  res_pt,   'Patient-level'),
            (axes[1], df_nc,  res_cell, 'Cell-level'),
        ]:
            vals_g1 = df_use[df_use['Disease'] == g1][feat].dropna().values
            vals_g2 = df_use[df_use['Disease'] == g2][feat].dropna().values

            # boxplot
            bp = ax.boxplot([vals_g1, vals_g2], patch_artist=True,
                            medianprops=dict(color='black', lw=2),
                            whiskerprops=dict(lw=1.2),
                            capprops=dict(lw=1.2),
                            flierprops=dict(marker='o', markersize=3, alpha=0.4))
            bp['boxes'][0].set_facecolor(c1); bp['boxes'][0].set_alpha(0.65)
            bp['boxes'][1].set_facecolor(c2); bp['boxes'][1].set_alpha(0.65)

            # jitter
            rng = np.random.default_rng(42)
            for xi, vals, c in [(1, vals_g1, c1), (2, vals_g2, c2)]:
                jit = rng.uniform(-0.15, 0.15, len(vals))
                ax.scatter(np.full(len(vals), xi) + jit, vals,
                           color=c, alpha=0.55, s=18, zorder=4, edgecolors='none')

            ax.set_xticks([1, 2])
            ax.set_xticklabels([g1_label, g2_label], fontsize=11)
            ax.set_ylabel(feat.replace('_', ' '), fontsize=10)
            ax.set_title(level, fontsize=11)
            ax.grid(axis='y', alpha=0.3)

            # annotation
            if not res.empty and feat in res['Feature'].values:
                row = res[res['Feature'] == feat].iloc[0]
                d   = row.get('Cohen_d', np.nan)
                q   = row.get('BH_q',    1.0)
                stars = sig_stars(q)
                ax.set_title(f"{level}\nd={d:.2f}  q={q:.3f}  {stars}", fontsize=10)

                # significance bar
                if q < ALPHA and len(vals_g1) > 0 and len(vals_g2) > 0:
                    y_top = max(np.percentile(vals_g1, 95) if len(vals_g1) else 0,
                                np.percentile(vals_g2, 95) if len(vals_g2) else 0) * 1.1
                    ax.plot([1, 1, 2, 2],
                            [y_top * 0.97, y_top, y_top, y_top * 0.97],
                            'k-', lw=1.2)
                    ax.text(1.5, y_top * 1.01, stars,
                            ha='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        fname = feat.replace('/', '_').replace(' ', '_')
        save(fig, out / f'{prefix}_{fname}.png')

    print(f"  Per-feature boxplots saved → {out}/")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: COHEN'S D BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
def plot_cohens_d_bars(res, path, title, g1_label, g2_label, level='patient'):
    """Horizontal bar chart of Cohen's d, coloured by organelle, stars on sig."""
    if res.empty:
        return
    sub = res.sort_values('Cohen_d', ascending=True).copy()
    colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in sub['Organelle']]

    fig, ax = plt.subplots(figsize=(10, max(6, len(sub) * 0.45)))
    bars = ax.barh(range(len(sub)), sub['Cohen_d'], color=colors,
                   edgecolor='white', height=0.7)
    ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=10)
    ax.set_xlabel(f"Cohen's d  ({g2_label} − {g1_label})", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    xlim = ax.get_xlim()
    pad  = (xlim[1] - xlim[0]) * 0.02
    for i, (_, row) in enumerate(sub.iterrows()):
        if row.get('Significant', False):
            x = row['Cohen_d']
            ax.text(x + (pad if x >= 0 else -pad), i, sig_stars(row['BH_q']),
                    va='center', ha='left' if x >= 0 else 'right',
                    fontsize=11, fontweight='bold')

    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, fontsize=10, loc='lower right')
    footer(fig, level)
    plt.tight_layout()
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: PATIENT vs CELL COMPARISON (dual -log10 q bars)
# ══════════════════════════════════════════════════════════════════════════════
def plot_level_comparison(res_pt, res_cell, path, title):
    """Side-by-side -log10(q) comparing patient-level and cell-level."""
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
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.45)))
    bw = 0.35
    ax.barh(y - bw/2, merged['-l10_pt'],   height=bw,
            color=LEVEL_COLORS['Patient'], label='Patient-level', alpha=0.85)
    ax.barh(y + bw/2, merged['-l10_cell'], height=bw,
            color=LEVEL_COLORS['Cell'],    label='Cell-level (exploratory)', alpha=0.85)
    ax.axvline(thr, color='red', ls='--', lw=1.5, label=f'FDR = {ALPHA}')
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace('_', ' ') for f in merged['Feature']], fontsize=10)
    ax.set_xlabel('-log₁₀ (BH-FDR q-value)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: LMM FOREST PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_lmm_forest(lmm, path, title='LMM — Forest Plot'):
    """Forest plot of LMM coefficients with 95% CI, one panel per term."""
    if lmm is None or lmm.empty:
        return
    terms = lmm['Term'].unique().tolist()
    max_feats = max((len(lmm[lmm['Term'] == t]) for t in terms), default=6)
    fig, axes = plt.subplots(1, len(terms),
                             figsize=(len(terms) * 5, max(6, max_feats * 0.55)))
    if len(terms) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for ax, term in zip(axes, terms):
        sub = lmm[lmm['Term'] == term].sort_values('Coef')
        colors_feat = ['#E74C3C' if s else '#AAA' for s in sub['Significant']]
        y = np.arange(len(sub))
        ax.scatter(sub['Coef'], y, color=colors_feat, zorder=4, s=50)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.plot([row['CI_low'], row['CI_high']], [i, i],
                    color='#E74C3C' if row['Significant'] else '#CCC', lw=1.5)
        ax.axvline(0, color='black', lw=0.9, ls='--')
        ax.set_yticks(y)
        ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=10)
        ax.set_xlabel('Coefficient (95% CI)', fontsize=11)
        n_sig = sub['Significant'].sum()
        ax.set_title(f'{term}\n{n_sig}/{len(sub)} sig', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: COLLAGEN RESCUE BARS
# ══════════════════════════════════════════════════════════════════════════════
def plot_collagen_bars(pt_res, cell_res, g1, g2, path, title='L4 — Collagen Rescue'):
    """2×2 grid: patient/cell × g1/g2, Cohen's d bars coloured by organelle."""
    groups  = [g1, g2]
    levels  = [('Patient-level (paired Wilcoxon)', pt_res),
               ('Cell-level (Mann-Whitney)',        cell_res)]

    if pt_res.empty and cell_res.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for row_i, (level_label, res) in enumerate(levels):
        for col_i, dis in enumerate(groups):
            ax  = axes[row_i][col_i]
            sub = res[res['Disease'] == dis].sort_values('Cohen_d', ascending=True)
            if sub.empty:
                ax.set_visible(False)
                continue
            org_colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in sub['Organelle']]
            ax.barh(range(len(sub)), sub['Cohen_d'], color=org_colors,
                    edgecolor='white', height=0.7)
            ax.axvline(0, color='black', lw=0.9)
            ax.set_yticks(range(len(sub)))
            ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=9)
            ax.set_xlabel("Cohen's d  (NoCollagen → +Collagen)", fontsize=11)
            n_sig = sub['Significant'].sum()
            ax.set_title(f'{GROUP_LABELS[dis]} — {level_label}\n'
                         f'{n_sig}/{len(sub)} sig (BH-FDR)',
                         fontsize=10, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            xlim = ax.get_xlim()
            pad  = (xlim[1] - xlim[0]) * 0.02
            for i, (_, row) in enumerate(sub.iterrows()):
                if row.get('Significant', False):
                    x = row['Cohen_d']
                    ax.text(x + (pad if x >= 0 else -pad), i, sig_stars(row['BH_q']),
                            va='center', ha='left' if x >= 0 else 'right', fontsize=10)

    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: SEX EFFECT BARS
# ══════════════════════════════════════════════════════════════════════════════
def plot_sex_bars(res_pt, res_cell, path, title='L5 — Sex Effect'):
    """Side-by-side Cohen's d bars for sex effect (patient vs cell)."""
    if res_pt.empty and res_cell.empty:
        return
    levels = [('Patient-level', res_pt, 'patient'),
              ('Cell-level',    res_cell, 'cell')]
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, 30 * 0.45)))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for ax, (label, res, lvl) in zip(axes, levels):
        if res.empty:
            ax.set_visible(False)
            continue
        sub    = res.sort_values('Cohen_d', ascending=True)
        colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in sub['Organelle']]
        ax.barh(range(len(sub)), sub['Cohen_d'], color=colors,
                edgecolor='white', height=0.7)
        ax.axvline(0, color='black', lw=0.9)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=10)
        ax.set_xlabel("Cohen's d  (Female → Male)", fontsize=12)
        n_sig = sub['Significant'].sum()
        ax.set_title(f'{label}\n{n_sig}/{len(sub)} sig (BH-FDR)', fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        footer(fig, lvl)
    plt.tight_layout()
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: SUMMARY HEATMAP + PCA
# ══════════════════════════════════════════════════════════════════════════════
def plot_summary(df_pt, feat_cols, res_pt, g1, g2, out_dir):
    """Z-score heatmap (top features) + PCA scatter at patient level."""
    out = Path(out_dir)
    sig_feats = (res_pt.loc[res_pt['Significant'], 'Feature'].tolist()
                 if not res_pt.empty else [])
    feats = sig_feats if sig_feats else feat_cols[:15]

    # Heatmap
    scaler = StandardScaler()
    X_sc   = pd.DataFrame(scaler.fit_transform(df_pt[feat_cols].fillna(0)),
                           columns=feat_cols, index=df_pt.index)
    hm = {GROUP_LABELS[g]: X_sc[df_pt['Disease'] == g][feats].mean()
          for g in [g1, g2]}
    hm_df = pd.DataFrame(hm, index=feats)
    row_labels = [f.replace('_', ' ') + (' *' if f in set(sig_feats) else '')
                  for f in feats]
    hm_df.index = row_labels

    fig, ax = plt.subplots(figsize=(8, max(6, len(feats) * 0.5)))
    sns.heatmap(hm_df, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Z-score (patient mean)'})
    ax.set_title(f'Summary Heatmap — {GROUP_LABELS[g1]} vs {GROUP_LABELS[g2]}\n'
                 '* = significant (BH-FDR, patient-level)',
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=15, labelsize=11)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)
    plt.tight_layout()
    save(fig, out / 'SUMMARY_heatmap.png')

    # PCA
    X  = df_pt[feat_cols].fillna(0).values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(2, Xs.shape[1]), random_state=42)
    Xp  = pca.fit_transform(Xs)
    ev  = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))
    for g in [g1, g2]:
        mask = df_pt['Disease'].values == g
        ax.scatter(Xp[mask, 0], Xp[mask, 1], c=GROUP_COLORS[g],
                   label=GROUP_LABELS[g], s=90, alpha=0.8,
                   edgecolors='white', lw=0.5)
        ctr = Xp[mask].mean(axis=0)
        ax.scatter(*ctr, c=GROUP_COLORS[g], s=300, marker='D',
                   edgecolors='black', lw=2, zorder=6)
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=12)
    ax.set_title(f'PCA — {GROUP_LABELS[g1]} vs {GROUP_LABELS[g2]}\n'
                 'Diamonds = group centroids', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, out / 'SUMMARY_pca.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: PER-FEATURE 4-GROUP BOXPLOTS (NoCollagen vs +Collagen × 2 groups)
# ══════════════════════════════════════════════════════════════════════════════
def plot_boxplots_collagen_per_feature(df, feat_cols, g1, g2, out_dir, prefix='col'):
    """
    For every feature: two panels (patient-level left, cell-level right).
    Each panel has 4 groups:
      g1 NoCollagen | g1 +Collagen | g2 NoCollagen | g2 +Collagen
    Collagen rescue p-value (paired Wilcoxon patient / MWU cell) annotated.
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

    for feat in feat_cols:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle(feat.replace('_', ' '), fontsize=13, fontweight='bold')

        for ax, df_src, level_label, is_cell in [
            (axes[0], pt_agg, 'Patient-level', False),
            (axes[1], df,     'Cell-level',    True),
        ]:

            data_groups = []
            for dis, col_status, xlbl, color, hatched in groups_def:
                if is_cell:
                    vals = (df_src[(df_src['Disease'] == dis) &
                                   (df_src['Collagen_Status'] == col_status)][feat]
                            .dropna().values)
                else:
                    vals = (df_src[(df_src['Disease'] == dis) &
                                   (df_src['Collagen_Status'] == col_status)][feat]
                            .dropna().values)
                data_groups.append((vals, xlbl, color, hatched))

            positions = [1, 2, 3.6, 4.6]
            bp = ax.boxplot([d[0] for d in data_groups],
                            positions=positions,
                            patch_artist=True,
                            medianprops=dict(color='black', lw=2),
                            whiskerprops=dict(lw=1.2),
                            capprops=dict(lw=1.2),
                            flierprops=dict(marker='o', markersize=3, alpha=0.4),
                            widths=0.65)

            rng = np.random.default_rng(42)
            for i, (bx, (vals, xlbl, color, hatched)) in enumerate(
                    zip(bp['boxes'], data_groups)):
                bx.set_facecolor(color)
                bx.set_alpha(0.72)
                if hatched:
                    bx.set_hatch('///')
                # jitter
                jit = rng.uniform(-0.18, 0.18, len(vals))
                ax.scatter(np.full(len(vals), positions[i]) + jit, vals,
                           color=color, alpha=0.55, s=16, zorder=4, edgecolors='none')

            ax.set_xticks(positions)
            ax.set_xticklabels([d[1] for d in data_groups], fontsize=10)
            ax.set_ylabel(feat.replace('_', ' '), fontsize=10)
            ax.set_title(level_label, fontsize=11)
            ax.grid(axis='y', alpha=0.3)

            # collagen rescue p-values (bracket between NoCol and +Col for each group)
            for gi, (dis, pos_nc, pos_col) in enumerate(
                    [(g1, positions[0], positions[1]),
                     (g2, positions[2], positions[3])]):
                if is_cell:
                    d1 = df_src[(df_src['Disease'] == dis) &
                                (df_src['Collagen_Status'] == 'NoCollagen')][feat].dropna().values
                    d2 = df_src[(df_src['Disease'] == dis) &
                                (df_src['Collagen_Status'] == 'Collagen')][feat].dropna().values
                    if len(d1) >= 3 and len(d2) >= 3:
                        try:
                            _, p = mannwhitneyu(d1, d2, alternative='two-sided')
                        except Exception:
                            p = np.nan
                    else:
                        p = np.nan
                else:
                    nc_s  = pt_agg[(pt_agg['Disease'] == dis) &
                                   (pt_agg['Collagen_Status'] == 'NoCollagen')].set_index('Subject')[feat]
                    col_s = pt_agg[(pt_agg['Disease'] == dis) &
                                   (pt_agg['Collagen_Status'] == 'Collagen')].set_index('Subject')[feat]
                    common = nc_s.index.intersection(col_s.index)
                    if len(common) >= 4:
                        diff = col_s.loc[common].values - nc_s.loc[common].values
                        try:
                            _, p = _wilcoxon(diff)
                        except Exception:
                            p = np.nan
                    else:
                        p = np.nan

                if not np.isnan(p):
                    all_vals = np.concatenate([
                        df_src[(df_src['Disease'] == dis)][feat].dropna().values
                        if is_cell else
                        pt_agg[(pt_agg['Disease'] == dis)][feat].dropna().values
                    ])
                    if len(all_vals):
                        y_top = np.percentile(all_vals, 97) * 1.08
                        ax.plot([pos_nc, pos_nc, pos_col, pos_col],
                                [y_top * 0.97, y_top, y_top, y_top * 0.97],
                                'k-', lw=1.1)
                        ax.text((pos_nc + pos_col) / 2, y_top * 1.01,
                                sig_stars(p), ha='center', fontsize=11,
                                fontweight='bold', color='black')

        # legend
        import matplotlib.patches as mpatches
        patches = [
            mpatches.Patch(facecolor=c1,     label=f'{g1_label} No Coll'),
            mpatches.Patch(facecolor=c1_col, label=f'{g1_label} +Coll',  hatch='///'),
            mpatches.Patch(facecolor=c2,     label=f'{g2_label} No Coll'),
            mpatches.Patch(facecolor=c2_col, label=f'{g2_label} +Coll',  hatch='///'),
        ]
        fig.legend(handles=patches, loc='lower center', ncol=4,
                   fontsize=10, bbox_to_anchor=(0.5, -0.04))
        plt.tight_layout(rect=[0, 0.06, 1, 1])

        fname = feat.replace('/', '_').replace(' ', '_')
        save(fig, out / f'{prefix}_{fname}.png')

    print(f"  Collagen per-feature boxplots saved → {out}/")
