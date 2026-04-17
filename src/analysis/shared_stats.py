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
             ha='right', fontsize=56, style='italic', color='#555')


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
        fig, axes = plt.subplots(1, 2, figsize=(36, 18))
        fig.suptitle(feat.replace('_', ' '), fontsize=42, fontweight='bold')

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
            ax.set_xticklabels([g1_label, g2_label], fontsize=36)
            ax.set_ylabel(feat.replace('_', ' '), fontsize=32)
            ax.set_title(level, fontsize=34)
            ax.tick_params(axis='y', labelsize=28)
            ax.grid(axis='y', alpha=0.3)

            # annotation
            if not res.empty and feat in res['Feature'].values:
                row = res[res['Feature'] == feat].iloc[0]
                d   = row.get('Cohen_d', np.nan)
                q   = row.get('BH_q',    1.0)
                stars = sig_stars(q)
                ax.set_title(f"{level}\nd={d:.2f}  q={q:.3f}  {stars}", fontsize=32)

                # significance bar — anchored to matplotlib's current ylim
                if q < ALPHA and len(vals_g1) > 0 and len(vals_g2) > 0:
                    ymin, ymax = ax.get_ylim()
                    yr = ymax - ymin
                    bracket_y  = ymax + yr * 0.03
                    new_top    = ymax + yr * 0.18
                    ax.set_ylim(ymin, new_top)
                    ax.plot([1, 1, 2, 2],
                            [ymax + yr * 0.01, bracket_y, bracket_y, ymax + yr * 0.01],
                            'k-', lw=1.5)
                    ax.text(1.5, bracket_y + yr * 0.01, stars,
                            ha='center', va='bottom', fontsize=36, fontweight='bold')

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

    fig_side = max(40, len(sub) * 2.4)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))
    bars = ax.barh(range(len(sub)), sub['Cohen_d'], color=colors,
                   edgecolor='white', height=0.85)
    ax.axvline(0, color='black', lw=1.2)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=100)
    ax.set_xlabel(f"Cohen's d  ({g2_label} − {g1_label})", fontsize=80)
    ax.set_title(title, fontsize=90, fontweight='bold')
    ax.tick_params(axis='x', labelsize=70)
    ax.grid(axis='x', alpha=0.3)

    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]
    ax.set_xlim(xlim[0] - x_range * 0.08, xlim[1] + x_range * 0.08)
    xlim = ax.get_xlim()
    pad  = (xlim[1] - xlim[0]) * 0.03
    for i, (_, row) in enumerate(sub.iterrows()):
        if row.get('Significant', False):
            x = row['Cohen_d']
            ax.text(x + (pad if x >= 0 else -pad), i, sig_stars(row['BH_q']),
                    va='center', ha='left' if x >= 0 else 'right',
                    fontsize=76, fontweight='bold')

    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, fontsize=70, loc='lower right')
    footer(fig, level)
    plt.tight_layout()
    fig.subplots_adjust(left=0.40)
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
    fig, ax = plt.subplots(figsize=(50, max(24, n * 3.0)))
    bw = 0.35
    ax.barh(y - bw/2, merged['-l10_pt'],   height=bw,
            color=LEVEL_COLORS['Patient'], label='Patient-level', alpha=0.85)
    ax.barh(y + bw/2, merged['-l10_cell'], height=bw,
            color=LEVEL_COLORS['Cell'],    label='Cell-level (exploratory)', alpha=0.85)
    ax.axvline(thr, color='red', ls='--', lw=2.0, label=f'FDR = {ALPHA}')
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace('_', ' ') for f in merged['Feature']], fontsize=52)
    ax.set_xlabel('-log₁₀ (BH-FDR q-value)', fontsize=44)
    ax.set_title(title, fontsize=46, fontweight='bold')
    ax.tick_params(axis='x', labelsize=38)
    ax.legend(fontsize=38, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    fig.subplots_adjust(left=0.44, right=0.97, top=0.95, bottom=0.06)
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
                             figsize=(len(terms) * 18, max(20, max_feats * 1.6)))
    if len(terms) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=40, fontweight='bold')

    for ax, term in zip(axes, terms):
        sub = lmm[lmm['Term'] == term].sort_values('Coef')
        colors_feat = ['#E74C3C' if s else '#AAA' for s in sub['Significant']]
        y = np.arange(len(sub))
        ax.scatter(sub['Coef'], y, color=colors_feat, zorder=4, s=120)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.plot([row['CI_low'], row['CI_high']], [i, i],
                    color='#E74C3C' if row['Significant'] else '#CCC', lw=2.5)
        ax.axvline(0, color='black', lw=1.2, ls='--')
        ax.set_yticks(y)
        ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=42)
        ax.set_xlabel('Coefficient (95% CI)', fontsize=34)
        ax.tick_params(axis='x', labelsize=30)
        n_sig = sub['Significant'].sum()
        ax.set_title(f'{term}\n{n_sig}/{len(sub)} sig', fontsize=34, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

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


def plot_lmm_heatmap(lmm, path, title='LMM — All Effects'):
    """
    Heatmap of full LMM results.
    Color = -log10(BH_q) × sign(Coef)  (red = positive, blue = negative).
    * marks cells with BH_q < 0.05.
    Dashed lines on colorbar at ±log10(0.05).
    """
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

    y_labels = [_abbrev_feature(f) for f in feat_order]
    n_feat, n_term = len(feat_order), len(term_order)

    vmax = max(3.0, float(np.abs(pivot.values).max()))

    fig, ax = plt.subplots(figsize=(max(20, n_term * 5), max(24, n_feat * 0.9)))

    im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(n_term))
    ax.set_xticklabels(term_order, fontsize=34)
    ax.set_yticks(np.arange(n_feat))
    ax.set_yticklabels(y_labels, fontsize=28)
    ax.set_ylabel('Feature', fontsize=30)

    ax.set_xticks(np.arange(-0.5, n_term, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_feat, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.0)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(n_feat):
        for j in range(n_term):
            if sig_pv.iloc[i, j]:
                ax.text(j, i, '*', ha='center', va='center',
                        fontsize=30, fontweight='bold', color='black')

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label(r'$-\log_{10}(p_\mathrm{FDR})\times\mathrm{sign(coef)}$',
                   fontsize=26)
    cbar.ax.tick_params(labelsize=22)
    thr = -np.log10(ALPHA)
    for sgn in [1, -1]:
        cbar.ax.axhline(sgn * thr, color='black', ls='--', lw=1.5)
        cbar.ax.text(1.1, sgn * thr, 'p=0.05', va='center', fontsize=20,
                     transform=cbar.ax.get_yaxis_transform())

    ax.set_title(f'{title}\nColor = -log10(p_FDR) × sign(coef)  |  * = FDR < 0.05',
                 fontsize=30, fontweight='bold')
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

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(28, max(16, n_subj * 1.6) * n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax, (vals, ses, panel_title, xlabel) in zip(axes, panels):
        order = np.argsort(vals)           # ascending → highest subject ends up at top
        for rank, idx in enumerate(order):
            ax.errorbar(vals[idx], rank,
                        xerr=1.96 * ses[idx],
                        fmt='o', color='black',
                        markerfacecolor='white', markeredgecolor='black',
                        markeredgewidth=2.2, markersize=16,
                        elinewidth=2.5, capsize=0, zorder=3)
        ax.axvline(0, color='red', ls='--', lw=2.5)
        ax.set_yticks(range(n_subj))
        ax.set_yticklabels([str(subjects[i]) for i in order], fontsize=34)
        ax.set_xlabel(xlabel, fontsize=36)
        ax.set_ylabel('Subject ID', fontsize=36)
        ax.set_title(panel_title, fontsize=40, fontweight='bold')
        ax.tick_params(axis='x', labelsize=30)
        ax.grid(axis='x', alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=44, fontweight='bold')

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

    # actual features per panel (each disease × level combination)
    n_per_panel = max(
        *(len(res[res['Disease'] == d]) for res in [pt_res, cell_res]
          if not res.empty for d in groups),
        15
    )
    fig_h = max(30, n_per_panel * 1.5 * 2 + 8)   # 1.5 in/feature × 2 rows + margins
    fig, axes = plt.subplots(2, 2, figsize=(44, fig_h))
    fig.suptitle(title, fontsize=32, fontweight='bold', y=0.995)

    for row_i, (level_label, res) in enumerate(levels):
        for col_i, dis in enumerate(groups):
            ax  = axes[row_i][col_i]
            sub = res[res['Disease'] == dis].sort_values('Cohen_d', ascending=True)
            if sub.empty:
                ax.set_visible(False)
                continue
            org_colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in sub['Organelle']]
            ax.barh(range(len(sub)), sub['Cohen_d'], color=org_colors,
                    edgecolor='white', height=0.85)
            ax.axvline(0, color='black', lw=1.2)
            ax.set_yticks(range(len(sub)))
            if col_i == 0:
                ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=36)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("Cohen's d  (NoCollagen → +Collagen)", fontsize=32)
            ax.tick_params(axis='x', labelsize=28)
            n_sig = sub['Significant'].sum()
            ax.set_title(f'{GROUP_LABELS[dis]} — {level_label}\n'
                         f'{n_sig}/{len(sub)} sig (BH-FDR)',
                         fontsize=30, fontweight='bold', pad=10)
            ax.grid(axis='x', alpha=0.3)

            xlim = ax.get_xlim()
            pad  = (xlim[1] - xlim[0]) * 0.02
            for i, (_, row) in enumerate(sub.iterrows()):
                if row.get('Significant', False):
                    x = row['Cohen_d']
                    ax.text(x + (pad if x >= 0 else -pad), i, sig_stars(row['BH_q']),
                            va='center', ha='left' if x >= 0 else 'right', fontsize=30)

    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=30, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.38, right=0.97, top=0.93, bottom=0.06,
                        hspace=0.55, wspace=0.30)
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
    n_feats = len(res_pt) if not res_pt.empty else len(res_cell) if not res_cell.empty else 30
    fig, axes = plt.subplots(1, 2, figsize=(42, max(20, n_feats * 2.4)))
    fig.suptitle(title, fontsize=30, fontweight='bold')
    for ax, (label, res, lvl) in zip(axes, levels):
        if res.empty:
            ax.set_visible(False)
            continue
        sub    = res.sort_values('Cohen_d', ascending=True)
        colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in sub['Organelle']]
        ax.barh(range(len(sub)), sub['Cohen_d'], color=colors,
                edgecolor='white', height=0.85)
        ax.axvline(0, color='black', lw=1.2)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=34)
        ax.set_xlabel("Cohen's d  (Female → Male)", fontsize=28)
        ax.tick_params(axis='x', labelsize=24)
        n_sig = sub['Significant'].sum()
        ax.set_title(f'{label}\n{n_sig}/{len(sub)} sig (BH-FDR)', fontsize=26)
        ax.grid(axis='x', alpha=0.3)
        footer(fig, lvl)
    plt.tight_layout()
    fig.subplots_adjust(left=0.38)
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

    fig, ax = plt.subplots(figsize=(14, max(10, len(feats) * 1.0)))
    sns.heatmap(hm_df, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Z-score (patient mean)'})
    ax.set_title(f'Summary Heatmap — {GROUP_LABELS[g1]} vs {GROUP_LABELS[g2]}\n'
                 '* = significant (BH-FDR, patient-level)',
                 fontsize=24, fontweight='bold')
    ax.tick_params(axis='x', rotation=15, labelsize=22)
    ax.tick_params(axis='y', rotation=0,  labelsize=18)
    plt.tight_layout()
    save(fig, out / 'SUMMARY_heatmap.png')

    # PCA
    X  = df_pt[feat_cols].fillna(0).values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(2, Xs.shape[1]), random_state=42)
    Xp  = pca.fit_transform(Xs)
    ev  = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 9))
    for g in [g1, g2]:
        mask = df_pt['Disease'].values == g
        ax.scatter(Xp[mask, 0], Xp[mask, 1], c=GROUP_COLORS[g],
                   label=GROUP_LABELS[g], s=90, alpha=0.8,
                   edgecolors='white', lw=0.5)
        ctr = Xp[mask].mean(axis=0)
        ax.scatter(*ctr, c=GROUP_COLORS[g], s=300, marker='D',
                   edgecolors='black', lw=2, zorder=6)
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=24)
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=24)
    ax.set_title(f'PCA — {GROUP_LABELS[g1]} vs {GROUP_LABELS[g2]}\n'
                 'Diamonds = group centroids', fontsize=24, fontweight='bold')
    ax.legend(fontsize=22)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, out / 'SUMMARY_pca.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: AORTA DIAMETER — STYLED SPEARMAN BARS
# ══════════════════════════════════════════════════════════════════════════════
def plot_diameter_spearman(res, path, title='L7 — Aorta Diameter'):
    """
    Styled Spearman ρ bar chart (reference style).
    Red = negative ρ, blue = positive ρ.
    Solid = FDR < 0.05, faded (alpha=0.22) = ns.
    Features sorted by organelle group then name.
    """
    if res.empty:
        return

    org_rank = {'Actin': 0, 'Mitochondria': 1, 'Nucleus': 2, 'Other': 3}
    sub = res.copy()
    sub['_org_rank'] = sub['Organelle'].map(org_rank).fillna(3)
    sub = sub.sort_values(['_org_rank', 'Feature']).reset_index(drop=True)
    n = len(sub)

    fig, ax = plt.subplots(figsize=(32, max(22, n * 1.4)))
    ax.set_facecolor('#F4F6F9')
    fig.patch.set_facecolor('white')

    for i, row in sub.iterrows():
        rho = row['Spearman_r']
        sig = row.get('Significant', False)
        alp = 0.88 if sig else 0.22
        col = '#C0392B' if rho < 0 else '#2471A3'
        ax.barh(i, rho, color=col, alpha=alp, height=0.72, edgecolor='none')

    ax.axvline(0, color='#777', ls='--', lw=1.4)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks(np.arange(-1.0, 1.25, 0.25))
    ax.set_yticks(range(n))
    ax.set_yticklabels(sub['Feature'], fontsize=34)
    ax.set_xlabel('Spearman ρ  (feature vs. aortic diameter)', fontsize=30)
    ax.set_title(title, fontsize=34, fontweight='bold')
    ax.tick_params(axis='x', labelsize=28)
    ax.grid(axis='x', color='white', linewidth=1.8, zorder=0)

    # organelle separator lines
    prev_org = None
    for i, row in sub.iterrows():
        if prev_org is not None and row['Organelle'] != prev_org:
            ax.axhline(i - 0.5, color='#BBBBBB', lw=1.4)
        prev_org = row['Organelle']

    # significance stars in matching colour
    for i, row in sub.iterrows():
        if row.get('Significant', False):
            x = row['Spearman_r']
            col = '#C0392B' if x < 0 else '#2471A3'
            ax.text(x + (0.03 if x >= 0 else -0.03), i, sig_stars(row['BH_q']),
                    va='center', ha='left' if x >= 0 else 'right',
                    fontsize=30, fontweight='bold', color=col)

    solid_p = mpatches.Patch(color='#777', alpha=0.88, label='FDR < 0.05  (solid)')
    faded_p = mpatches.Patch(color='#777', alpha=0.22, label='ns  (faded)')
    ax.legend(handles=[solid_p, faded_p], fontsize=26, loc='lower right', framealpha=0.9)

    fig.subplots_adjust(left=0.42, right=0.97, top=0.93, bottom=0.07)
    save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: SEX EFFECT HEATMAP (Cohen's d per group)
# ══════════════════════════════════════════════════════════════════════════════
def plot_sex_heatmap(df_pt, feat_cols, g1, g2, path, title='Sex Effect'):
    """
    Heatmap of Cohen's d (Male − Female) within each group.
    One column per group; features sorted by mean |Cohen's d|.
    Stars for BH-FDR < 0.05 within each group.
    """
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

    d_df  = pd.DataFrame(np.column_stack(d_cols), index=feat_cols, columns=col_labels)
    q_df  = pd.DataFrame(np.column_stack(q_cols), index=feat_cols, columns=col_labels)
    sig_df = q_df < ALPHA

    order = d_df.abs().mean(axis=1).sort_values().index
    d_df  = d_df.loc[order]
    q_df  = q_df.loc[order]
    sig_df = sig_df.loc[order]

    n_feat = len(d_df)
    n_col  = len(groups)
    vmax   = max(4.0, float(d_df.abs().values.max()))

    fig, ax = plt.subplots(figsize=(max(16, n_col * 7), max(24, n_feat * 0.9)))

    im = ax.imshow(d_df.values, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_col))
    ax.set_xticklabels(col_labels, fontsize=32)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(d_df.index, fontsize=28)

    ax.set_xticks(np.arange(-0.5, n_col, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_feat, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.8)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(n_feat):
        for j in range(n_col):
            if sig_df.iloc[i, j]:
                ax.text(j, i, sig_stars(q_df.iloc[i, j]),
                        ha='center', va='center',
                        fontsize=26, fontweight='bold', color='black')

    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
    cbar.set_label("Cohen's d", fontsize=30)
    cbar.ax.tick_params(labelsize=26)

    ax.set_title(f"{title}\nCohen's d  (Male − Female)\n* FDR<0.05  ** <0.01  *** <0.001",
                 fontsize=32, fontweight='bold')

    plt.tight_layout()
    save(fig, path)


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
        fig, axes = plt.subplots(1, 2, figsize=(40, 18))
        fig.suptitle(feat.replace('_', ' '), fontsize=42, fontweight='bold')

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
            ax.set_xticklabels([d[1] for d in data_groups], fontsize=34)
            ax.set_ylabel(feat.replace('_', ' '), fontsize=32)
            ax.tick_params(axis='y', labelsize=28)
            ax.set_title(level_label, fontsize=36)
            ax.grid(axis='y', alpha=0.3)

            # collagen rescue p-values — compute all first, then draw with stable ylim
            p_vals = {}
            for dis, pos_nc, pos_col in [(g1, positions[0], positions[1]),
                                          (g2, positions[2], positions[3])]:
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
                p_vals[(dis, pos_nc, pos_col)] = p

            # only draw brackets for significant results; extend ylim once
            sig_pairs = [(dis, pos_nc, pos_col, p)
                         for (dis, pos_nc, pos_col), p in p_vals.items()
                         if not np.isnan(p) and p < ALPHA]
            if sig_pairs:
                ymin, ymax = ax.get_ylim()
                yr = ymax - ymin
                # reserve enough space for all brackets (stack them if both sig)
                n_brackets = len(sig_pairs)
                ax.set_ylim(ymin, ymax + yr * 0.15 * n_brackets)
                for k, (dis, pos_nc, pos_col, p) in enumerate(sig_pairs):
                    b_y = ymax + yr * (0.03 + 0.13 * k)
                    ax.plot([pos_nc, pos_nc, pos_col, pos_col],
                            [ymax + yr * 0.01, b_y, b_y, ymax + yr * 0.01],
                            'k-', lw=1.1)
                    ax.text((pos_nc + pos_col) / 2, b_y + yr * 0.01,
                            sig_stars(p), ha='center', va='bottom', fontsize=34,
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
                   fontsize=30, bbox_to_anchor=(0.5, -0.04))
        plt.tight_layout(rect=[0, 0.06, 1, 1])

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
        fig, axes = plt.subplots(1, 2, figsize=(40, 18))
        fig.suptitle(feat.replace('_', ' '), fontsize=42, fontweight='bold')

        for ax, is_cell, level_label, res in [
            (axes[0], False, 'Patient-level', res_pt),
            (axes[1], True,  'Cell-level',    res_cell),
        ]:
            # ── data ──────────────────────────────────────────────────────────
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
                            medianprops=dict(color='black', lw=2),
                            whiskerprops=dict(lw=1.2),
                            capprops=dict(lw=1.2),
                            flierprops=dict(marker='o', markersize=3, alpha=0.4),
                            widths=0.65)
            for bx, color, hatch in zip(bp['boxes'], colors_box, hatches):
                bx.set_facecolor(color)
                bx.set_alpha(0.72)
                if hatch:
                    bx.set_hatch(hatch)

            for pos, vals, color in zip(positions, data, colors_box):
                jit = rng.uniform(-0.18, 0.18, len(vals))
                ax.scatter(np.full(len(vals), pos) + jit, vals,
                           color=color, alpha=0.55, s=18, zorder=4, edgecolors='none')

            ax.set_xticks(positions)
            ax.set_xticklabels(xlabels, fontsize=34)
            ax.set_ylabel(feat.replace('_', ' '), fontsize=32)
            ax.tick_params(axis='y', labelsize=28)
            ax.set_title(level_label, fontsize=36)
            ax.grid(axis='y', alpha=0.3)
            ax.set_xlim(0.3, 5.1)

            # ── p-values ───────────────────────────────────────────────────────
            dis_q = np.nan
            if res is not None and not res.empty and feat in res['Feature'].values:
                dis_q = res.loc[res['Feature'] == feat, 'BH_q'].iloc[0]

            p_g1_rescue = _rescue_p(is_cell, g1, v_g1_nc, v_g1_col)
            p_g2_rescue = _rescue_p(is_cell, g2, v_g2_nc, v_g2_col)

            # build significant bracket list, disease bracket placed last (highest)
            rescue_brackets = []
            if not np.isnan(p_g1_rescue) and p_g1_rescue < ALPHA:
                rescue_brackets.append((positions[0], positions[1], sig_stars(p_g1_rescue)))
            if not np.isnan(p_g2_rescue) and p_g2_rescue < ALPHA:
                rescue_brackets.append((positions[2], positions[3], sig_stars(p_g2_rescue)))

            dis_bracket = []
            if not np.isnan(dis_q) and dis_q < ALPHA:
                dis_bracket = [(positions[0], positions[2], sig_stars(dis_q))]

            all_brackets = rescue_brackets + dis_bracket  # rescue lower, disease higher

            if all_brackets:
                ymin, ymax = ax.get_ylim()
                yr = ymax - ymin
                ax.set_ylim(ymin, ymax + yr * 0.15 * len(all_brackets))
                for k, (x1, x2, stars) in enumerate(all_brackets):
                    b_y = ymax + yr * (0.03 + 0.13 * k)
                    ax.plot([x1, x1, x2, x2],
                            [ymax + yr * 0.01, b_y, b_y, ymax + yr * 0.01],
                            'k-', lw=1.5)
                    ax.text((x1 + x2) / 2, b_y + yr * 0.01, stars,
                            ha='center', va='bottom', fontsize=36, fontweight='bold')

        patches = [
            mpatches.Patch(facecolor=c1,    label=f'{g1_label} NoCol'),
            mpatches.Patch(facecolor=c1_col, label=f'{g1_label} +Col', hatch='///'),
            mpatches.Patch(facecolor=c2,    label=f'{g2_label} NoCol'),
            mpatches.Patch(facecolor=c2_col, label=f'{g2_label} +Col', hatch='///'),
        ]
        fig.legend(handles=patches, loc='lower center', ncol=4,
                   fontsize=30, bbox_to_anchor=(0.5, -0.04))
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        fname = feat.replace('/', '_').replace(' ', '_')
        save(fig, out / f'{prefix}_{fname}.png')

    print(f"  Rescue comparison boxplots saved → {out}/")
