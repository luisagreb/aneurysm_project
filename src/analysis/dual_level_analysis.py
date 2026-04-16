"""
Dual-Level Statistical Analysis — Complete Framework
=====================================================
Thesis: Computational 3D Phenotyping of Mitochondrial Morphology in Aortic Aneurysm

TWO COMPLEMENTARY LEVELS OF ANALYSIS:

  LEVEL A — PATIENT-LEVEL (Primary, pseudoreplication-corrected)
    Unit: one value per donor = mean across NoCollagen cells
    n:    Healthy ~10 | TAA ~11 | BAV ~5
    Tests: KW + BH-FDR + Dunn's post-hoc + Cohen's d
    This is your PRIMARY result. Report these numbers in the main text.

  LEVEL B — CELL-LEVEL (Exploratory / Hypothesis-generating)
    Unit: individual cells (pseudoreplicated)
    n:    Healthy ~110 | TAA ~111 | BAV ~50
    Tests: KW + BH-FDR + Dunn's post-hoc + Cohen's d
    Report in Supplementary or as supporting evidence.

  LEVEL C — LMM CONFIRMATORY (Bridges both levels)
    Unit: cells, clustered by patient (random intercept)
    Tests: LMM with Disease + Age + Sex as fixed effects
    Accounts for within-patient correlation without losing cell-level resolution.

LAYERS TESTED AT BOTH LEVELS:
  L1  Disease effect        (Healthy vs TAA vs BAV)
  L2  Age-corrected disease (residualize age, repeat L1)
  L3  LMM confirmatory      (cell-level, subject random intercept)
  L4  Collagen rescue        (NoCollagen vs Collagen, per disease group)
  L5  Sex effect             (Male vs Female, overall + per disease)
  L6  Hypertension effect    (Yes vs No)
  L7  Aorta diameter         (Spearman correlation)

HOW TO REPORT IN THESIS:
  Main text   → Level A (patient-level) numbers
  Main text   → Level C (LMM) for confirmation
  Supplement  → Level B (cell-level) as supporting evidence
  Framing     → "Cell-level analysis (Supplementary) revealed broader trends
                 consistent with patient-level findings..."

Outputs: outputs/dual_level_analysis/
  patient/   — Level A results
  cell/      — Level B results
  lmm/       — Level C results
  comparison/ — Side-by-side patient vs cell comparison plots
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu, spearmanr, shapiro
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print("WARNING: scikit_posthocs not installed → pip install scikit-posthocs")

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed → pip install statsmodels")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
FEATURES_FILE = 'outputs/Advanced_Features_Raw_Final.csv'
METADATA_FILE = 'Book1 (2).xlsx'
BASE_DIR      = Path('outputs/dual_level_analysis')
PT_DIR        = BASE_DIR / 'patient'
CELL_DIR      = BASE_DIR / 'cell'
LMM_DIR       = BASE_DIR / 'lmm'
COMP_DIR      = BASE_DIR / 'comparison'
ALPHA         = 0.05

DISEASE_ORDER  = ['Healthy', 'TAA', 'BAV']
DISEASE_LABELS = {'Healthy': 'Non-Aneurysmal', 'TAA': 'TAA', 'BAV': 'BAV'}
DISEASE_COLORS = {'Healthy': '#4C9BE8', 'TAA': '#E74C3C', 'BAV': '#F39C12'}
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
    note = ('Patient-level' if level == 'patient'
            else 'Cell-level (pseudoreplicated — exploratory)')
    fig.text(0.98, 0.005,
             f'* p<0.05  ** p<0.01  *** p<0.001  (BH-FDR, {note})',
             ha='right', fontsize=9, style='italic', color='#555')

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(FEATURES_FILE)
    df = df[df['Disease'].isin(DISEASE_ORDER)].copy()
    df['Subject'] = df['CellName'].apply(norm_id)

    feat_cols = sorted([c for c in df.columns
                        if c.startswith(('Actin_', 'Mito_', 'Nucleus_'))])
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)

    # Load metadata
    try:
        meta = pd.read_excel(METADATA_FILE, sheet_name='specimen information')
        meta['Subject']      = meta['Aortic Specimen ID'].apply(norm_id)
        meta['Hypertn']      = meta['hypertn'].map({'Yes': 'Yes', 'No': 'No'})
        meta['AortaDiam_mm'] = pd.to_numeric(meta['trtznlrgdiammeas'], errors='coerce')
        df = df.merge(meta[['Subject', 'Hypertn', 'AortaDiam_mm']].dropna(subset=['Subject']),
                      on='Subject', how='left')
        print(f"  Hypertension data matched: {df['Hypertn'].notna().sum()} cells")
        print(f"  Aorta diameter matched:    {df['AortaDiam_mm'].notna().sum()} cells")
    except Exception as e:
        print(f"  Metadata not loaded: {e}")
        df['Hypertn']      = np.nan
        df['AortaDiam_mm'] = np.nan

    # Patient-level aggregation (NoCollagen only for disease comparison)
    df_nc = df[df['Collagen_Status'] == 'NoCollagen'].copy()

    # FIX 1: added .reset_index() so 'Subject' becomes a column and the
    # subsequent merge(on='Subject') can find it in both DataFrames.
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
    for d in DISEASE_ORDER:
        sp = df_pt[df_pt['Disease'] == d]
        sc = df_nc[df_nc['Disease'] == d]
        ages = sp['Age'].dropna()
        print(f"    {DISEASE_LABELS[d]:18s}: {len(sp)} patients ({len(sc)} cells) | "
              f"Age {ages.mean():.1f}±{ages.std():.1f}yr | "
              f"M={(sp['Gender'] == 'Male').sum()} F={(sp['Gender'] == 'Female').sum()}")

    df_pt.to_csv(PT_DIR / 'patient_means.csv', index=False)
    return df, df_nc, df_pt, feat_cols

# ══════════════════════════════════════════════════════════════════════════════
# CORE STATISTICAL ENGINE
# Runs the same KW + Dunn + Cohen's d pipeline on any dataframe
# ══════════════════════════════════════════════════════════════════════════════
def run_kw_analysis(df_input, feat_cols, label=''):
    """
    Kruskal-Wallis + BH-FDR + pairwise Mann-Whitney + Cohen's d.
    Works identically for patient-level or cell-level data.
    """
    rows = []
    for feat in feat_cols:
        groups = [df_input.loc[df_input['Disease'] == d, feat].dropna().values
                  for d in DISEASE_ORDER]
        if any(len(g) < 3 for g in groups):
            continue
        try:   stat, p = kruskal(*groups)
        except: stat, p = np.nan, np.nan

        pairs = [('Healthy', 'TAA'), ('Healthy', 'BAV'), ('TAA', 'BAV')]
        pair_stats = {}
        for a, b in pairs:
            ga = df_input.loc[df_input['Disease'] == a, feat].dropna().values
            gb = df_input.loc[df_input['Disease'] == b, feat].dropna().values
            pair_stats[f'd_{a}_{b}'] = cohens_d(ga, gb) if len(ga) > 1 and len(gb) > 1 else np.nan
            try:
                _, pp = mannwhitneyu(ga, gb, alternative='two-sided')
            except:
                pp = np.nan
            pair_stats[f'p_{a}_{b}'] = pp
        rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                     'KW_stat': stat, 'KW_p': p, **pair_stats})

    kw = pd.DataFrame(rows)
    kw['BH_q']        = bh_fdr(kw['KW_p'].values)
    kw['Significant'] = kw['BH_q'] < ALPHA
    for a, b in [('Healthy', 'TAA'), ('Healthy', 'BAV'), ('TAA', 'BAV')]:
        pkey = f'p_{a}_{b}'
        if pkey in kw.columns:
            kw[f'q_{a}_{b}'] = bh_fdr(kw[pkey].fillna(1).values)

    n_sig = kw['Significant'].sum()
    print(f"    {label}: {n_sig}/{len(kw)} significant (KW BH-FDR)")
    return kw


def run_dunn(df_input, kw, label=''):
    """Dunn's post-hoc on significant features."""
    dunn = {}
    if not HAS_POSTHOCS:
        return dunn
    for feat in kw.loc[kw['Significant'], 'Feature']:
        data = [df_input.loc[df_input['Disease'] == d, feat].dropna().tolist()
                for d in DISEASE_ORDER]
        try:
            pm = sp.posthoc_dunn(data, p_adjust='bonferroni')
            pm.index = pm.columns = DISEASE_ORDER
            dunn[feat] = pm
        except:
            pass
    print(f"    {label}: Dunn's post-hoc computed for {len(dunn)} features")
    return dunn


def run_age_correction(df_input, feat_cols, label=''):
    """Regress age out, return residualized dataframe."""
    df_r = df_input.copy()
    n = 0
    for feat in feat_cols:
        sub = df_input[['Age', feat]].dropna()
        if len(sub) < 5:
            continue
        mdl = LinearRegression().fit(sub[['Age']], sub[feat])
        age_f = df_input['Age'].fillna(df_input['Age'].mean()).values.reshape(-1, 1)
        df_r[feat] = df_input[feat] - mdl.predict(age_f)
        n += 1
    print(f"    {label}: age regressed out of {n} features")
    return df_r

# ══════════════════════════════════════════════════════════════════════════════
# L1 — DISEASE EFFECT
# ══════════════════════════════════════════════════════════════════════════════
def layer1(df_nc, df_pt, feat_cols):
    print("\n" + "=" * 70)
    print("L1 — DISEASE EFFECT")
    print("=" * 70)

    kw_pt   = run_kw_analysis(df_pt, feat_cols, 'Patient-level')
    kw_cell = run_kw_analysis(df_nc, feat_cols, 'Cell-level')
    dunn_pt   = run_dunn(df_pt, kw_pt,   'Patient-level')
    dunn_cell = run_dunn(df_nc, kw_cell, 'Cell-level')

    kw_pt.to_csv(PT_DIR   / 'L1_kruskal_wallis.csv', index=False)
    kw_cell.to_csv(CELL_DIR / 'L1_kruskal_wallis.csv', index=False)

    # Plots — patient level
    plot_disease_bars(kw_pt,   PT_DIR   / 'L1_disease_effect.png',   'Patient-level', 'patient')
    plot_cohens_d_3panel(kw_pt, PT_DIR  / 'L1_cohens_d.png',         'Patient-level', 'patient')
    if dunn_pt:
        plot_dunn_heatmap(dunn_pt, kw_pt, PT_DIR / 'L1_dunn.png',    'Patient-level')
        save_dunn_csv(dunn_pt, PT_DIR / 'L1_dunn_posthoc.csv')

    # Plots — cell level
    plot_disease_bars(kw_cell,   CELL_DIR / 'L1_disease_effect.png', 'Cell-level (exploratory)', 'cell')
    plot_cohens_d_3panel(kw_cell, CELL_DIR / 'L1_cohens_d.png',      'Cell-level (exploratory)', 'cell')
    if dunn_cell:
        plot_dunn_heatmap(dunn_cell, kw_cell, CELL_DIR / 'L1_dunn.png', 'Cell-level (exploratory)')
        save_dunn_csv(dunn_cell, CELL_DIR / 'L1_dunn_posthoc.csv')

    # Comparison plot
    plot_level_comparison(kw_pt, kw_cell, COMP_DIR / 'L1_patient_vs_cell.png',
                          'L1 — Disease Effect: Patient vs Cell Level')

    return kw_pt, kw_cell, dunn_pt, dunn_cell

# ══════════════════════════════════════════════════════════════════════════════
# L2 — AGE-CORRECTED DISEASE EFFECT
# ══════════════════════════════════════════════════════════════════════════════
def layer2(df_nc, df_pt, feat_cols, kw_pt_l1, kw_cell_l1):
    print("\n" + "=" * 70)
    print("L2 — AGE-CORRECTED DISEASE EFFECT")
    print("=" * 70)

    df_pt_r   = run_age_correction(df_pt,  feat_cols, 'Patient-level')
    df_cell_r = run_age_correction(df_nc,  feat_cols, 'Cell-level')

    kw_pt2   = run_kw_analysis(df_pt_r,   feat_cols, 'Patient-level (age-corrected)')
    kw_cell2 = run_kw_analysis(df_cell_r, feat_cols, 'Cell-level (age-corrected)')

    kw_pt2.to_csv(PT_DIR   / 'L2_age_corrected_kw.csv', index=False)
    kw_cell2.to_csv(CELL_DIR / 'L2_age_corrected_kw.csv', index=False)

    # FIX 3: removed redundant local import — spearmanr already imported at top
    age_rows = []
    for feat in feat_cols:
        sub = df_pt[['Age', feat]].dropna()
        if len(sub) < 5: continue
        r, p = spearmanr(sub['Age'], sub[feat])
        age_rows.append({'Feature': feat, 'Spearman_r': r, 'p': p})
    age_df = pd.DataFrame(age_rows)
    if not age_df.empty:
        age_df['BH_q'] = bh_fdr(age_df['p'].values)
        age_df['Significant'] = age_df['BH_q'] < ALPHA
        age_df.to_csv(PT_DIR / 'L2_age_correlations.csv', index=False)

    # Print comparison
    for label, kw1, kw2 in [('Patient', kw_pt_l1, kw_pt2),
                              ('Cell',    kw_cell_l1, kw_cell2)]:
        s1 = set(kw1.loc[kw1['Significant'], 'Feature'])
        s2 = set(kw2.loc[kw2['Significant'], 'Feature'])
        print(f"  {label}: {len(s1)} → {len(s2)} sig  "
              f"(kept={len(s1 & s2)}, lost={len(s1 - s2)}, gained={len(s2 - s1)})")

    # Plots
    plot_disease_bars(kw_pt2,   PT_DIR   / 'L2_age_corrected.png',
                      'Patient-level (age-corrected)', 'patient')
    plot_disease_bars(kw_cell2, CELL_DIR / 'L2_age_corrected.png',
                      'Cell-level age-corrected (exploratory)', 'cell')
    plot_l1_vs_l2(kw_pt_l1, kw_pt2,     PT_DIR   / 'L2_L1vsL2_scatter.png', 'Patient-level')
    plot_l1_vs_l2(kw_cell_l1, kw_cell2, CELL_DIR / 'L2_L1vsL2_scatter.png', 'Cell-level')
    plot_level_comparison(kw_pt2, kw_cell2, COMP_DIR / 'L2_patient_vs_cell.png',
                          'L2 — Age-Corrected Disease Effect: Patient vs Cell Level')

    return kw_pt2, kw_cell2

# ══════════════════════════════════════════════════════════════════════════════
# L3 — LMM CONFIRMATORY
# ══════════════════════════════════════════════════════════════════════════════
def layer3_lmm(df, feat_cols, kw_pt_l1):
    print("\n" + "=" * 70)
    print("L3 — LMM CONFIRMATORY (cell-level, Subject = random intercept)")
    print("=" * 70)

    if not HAS_STATSMODELS:
        print("  SKIPPED — pip install statsmodels")
        return None

    target = kw_pt_l1.loc[kw_pt_l1['Significant'], 'Feature'].tolist() or feat_cols
    print(f"  Running LMM on {len(target)} features "
          f"(patient-level significant + all if none)")

    df = df.copy()
    df['D_TAA']   = (df['Disease'] == 'TAA').astype(float)
    df['D_BAV']   = (df['Disease'] == 'BAV').astype(float)
    df['Sex_bin'] = (df['Gender']  == 'Male').astype(float)

    records = []
    for feat in target:
        needed = ['Subject', 'D_TAA', 'D_BAV', 'Age', 'Sex_bin', feat]
        sub = df[needed].rename(columns={feat: 'Y'}).dropna()
        if sub['Subject'].nunique() < 5: continue
        try:
            res = smf.mixedlm('Y ~ D_TAA + D_BAV + Age + Sex_bin',
                               data=sub, groups=sub['Subject']
                               ).fit(reml=True, method='lbfgs')
            for label, param in [('TAA vs Healthy', 'D_TAA'),
                                  ('BAV vs Healthy', 'D_BAV'),
                                  ('Age',            'Age'),
                                  ('Sex (Male)',      'Sex_bin')]:
                if param in res.params:
                    ci = res.conf_int().loc[param]
                    records.append({
                        'Feature': feat, 'Organelle': get_organelle(feat),
                        'Term': label,
                        'Coef': res.params[param], 'SE': res.bse[param],
                        'p': res.pvalues[param],
                        'CI_low': ci[0], 'CI_high': ci[1],
                        'N_cells': len(sub), 'N_patients': sub['Subject'].nunique(),
                    })
        except:
            pass

    if not records:
        print("  No LMM results.")
        return None

    lmm = pd.DataFrame(records)
    for term in lmm['Term'].unique():
        mask = lmm['Term'] == term
        lmm.loc[mask, 'BH_q'] = bh_fdr(lmm.loc[mask, 'p'].values)
    lmm['Significant'] = lmm['BH_q'] < ALPHA
    lmm.to_csv(LMM_DIR / 'L3_lmm_results.csv', index=False)

    print("  Significant (BH-FDR):")
    for term in lmm['Term'].unique():
        sig = lmm[(lmm['Term'] == term) & lmm['Significant']]['Feature'].tolist()
        print(f"    {term:20s}: {len(sig)} — {sig}")

    plot_lmm_forest(lmm, LMM_DIR / 'L3_lmm_forest.png')
    return lmm

# ══════════════════════════════════════════════════════════════════════════════
# L4 — COLLAGEN RESCUE
# ══════════════════════════════════════════════════════════════════════════════
# FIX 4: removed unused df_pt_full parameter — the function builds its own
# patient-level aggregation directly from df.
def layer4_collagen(df, feat_cols):
    """
    Patient-level: paired Wilcoxon on patient means (NoCollagen vs Collagen).
    Cell-level: Mann-Whitney on all cells.
    """
    print("\n" + "=" * 70)
    print("L4 — COLLAGEN RESCUE EFFECT")
    print("=" * 70)

    from scipy.stats import wilcoxon as wilcoxon_test

    # ── Patient-level: paired Wilcoxon ────────────────────────────────────────
    print("  Patient-level: paired Wilcoxon (NoCollagen vs Collagen means)")
    df_full = df.copy()
    pt_agg = df_full.groupby(['Subject', 'Disease', 'Collagen_Status'])[feat_cols].mean().reset_index()

    pt_rows = []
    for dis in DISEASE_ORDER:
        for feat in feat_cols:
            nc  = pt_agg[(pt_agg['Disease'] == dis) & (pt_agg['Collagen_Status'] == 'NoCollagen')]\
                      .set_index('Subject')[feat]
            col = pt_agg[(pt_agg['Disease'] == dis) & (pt_agg['Collagen_Status'] == 'Collagen')]\
                      .set_index('Subject')[feat]
            common = nc.index.intersection(col.index)
            if len(common) < 4: continue
            diff = col.loc[common].values - nc.loc[common].values
            try:
                _, p = wilcoxon_test(diff)
            except:
                p = np.nan
            d = cohens_d(nc.loc[common].values, col.loc[common].values)
            pt_rows.append({'Disease': dis, 'Feature': feat,
                            'Organelle': get_organelle(feat),
                            'Cohen_d': d, 'p': p,
                            'N_pairs': len(common)})

    pt_coll = pd.DataFrame(pt_rows)
    for dis in DISEASE_ORDER:
        mask = pt_coll['Disease'] == dis
        pt_coll.loc[mask, 'BH_q'] = bh_fdr(pt_coll.loc[mask, 'p'].fillna(1).values)
    pt_coll['Significant'] = pt_coll['BH_q'] < ALPHA
    pt_coll.to_csv(PT_DIR / 'L4_collagen_wilcoxon.csv', index=False)

    for dis in DISEASE_ORDER:
        sub = pt_coll[pt_coll['Disease'] == dis]
        n_pairs = int(sub['N_pairs'].iloc[0]) if len(sub) else 0
        print(f"    {DISEASE_LABELS[dis]:18s}: "
              f"{sub['Significant'].sum()}/{len(sub)} sig "
              f"(paired Wilcoxon, n={n_pairs} pairs)")

    # ── Cell-level: Mann-Whitney ───────────────────────────────────────────────
    print("  Cell-level: Mann-Whitney (NoCollagen vs Collagen cells)")
    cell_rows = []
    for dis in DISEASE_ORDER:
        sub = df[df['Disease'] == dis]
        nc_cells  = sub[sub['Collagen_Status'] == 'NoCollagen']
        col_cells = sub[sub['Collagen_Status'] == 'Collagen']
        for feat in feat_cols:
            d1 = nc_cells[feat].dropna().values
            d2 = col_cells[feat].dropna().values
            if len(d1) < 3 or len(d2) < 3: continue
            try:
                _, p = mannwhitneyu(d1, d2, alternative='two-sided')
            except:
                p = np.nan
            cell_rows.append({'Disease': dis, 'Feature': feat,
                               'Organelle': get_organelle(feat),
                               'Cohen_d': cohens_d(d1, d2), 'p': p})

    cell_coll = pd.DataFrame(cell_rows)
    for dis in DISEASE_ORDER:
        mask = cell_coll['Disease'] == dis
        cell_coll.loc[mask, 'BH_q'] = bh_fdr(cell_coll.loc[mask, 'p'].fillna(1).values)
    cell_coll['Significant'] = cell_coll['BH_q'] < ALPHA
    cell_coll.to_csv(CELL_DIR / 'L4_collagen_mw.csv', index=False)

    for dis in DISEASE_ORDER:
        sub = cell_coll[cell_coll['Disease'] == dis]
        print(f"    {DISEASE_LABELS[dis]:18s}: "
              f"{sub['Significant'].sum()}/{len(sub)} sig "
              f"(Mann-Whitney, cell-level)")

    # ── Distance to healthy centroid ──────────────────────────────────────────
    # FIX 2: fillna(0) before StandardScaler to avoid crash on NaN values
    scaler   = StandardScaler()
    X_sc     = pd.DataFrame(scaler.fit_transform(df[feat_cols].fillna(0)),
                             columns=feat_cols, index=df.index)
    centroid = X_sc[df['Disease'] == 'Healthy'].mean().values

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('L4 — Distance to Healthy Centroid (Rescue Effect)\n'
                 'Cell-level  |  Mann-Whitney p-value shown',
                 fontsize=13, fontweight='bold')
    for ax, dis in zip(axes, DISEASE_ORDER):
        idx_nc  = df.index[(df['Disease'] == dis) & (df['Collagen_Status'] == 'NoCollagen')]
        idx_col = df.index[(df['Disease'] == dis) & (df['Collagen_Status'] == 'Collagen')]
        d_nc  = np.linalg.norm(X_sc.loc[idx_nc].values  - centroid, axis=1)
        d_col = np.linalg.norm(X_sc.loc[idx_col].values - centroid, axis=1)
        bp = ax.boxplot([d_nc, d_col], patch_artist=True,
                        medianprops=dict(color='black', lw=2))
        bp['boxes'][0].set_facecolor('#AAAAAA'); bp['boxes'][0].set_alpha(0.8)
        bp['boxes'][1].set_facecolor('#9B59B6'); bp['boxes'][1].set_alpha(0.8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['No Collagen', '+Collagen'], fontsize=11)
        ax.set_ylabel('Distance to Healthy centroid', fontsize=11)
        if len(d_nc) >= 3 and len(d_col) >= 3:
            try:
                _, p = mannwhitneyu(d_nc, d_col, alternative='two-sided')
                y_top = max(d_nc.max(), d_col.max()) * 1.08
                ax.plot([1, 1, 2, 2], [y_top * 0.97, y_top, y_top, y_top * 0.97], 'k-', lw=1.2)
                ax.text(1.5, y_top * 1.01, f'p={p:.3f}', ha='center', fontsize=11)
            except: pass
        direction = 'RESCUE' if (len(d_col) > 0 and d_col.mean() < d_nc.mean()) else 'NO RESCUE'
        ax.set_title(f'{DISEASE_LABELS[dis]}\n{direction}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig, CELL_DIR / 'L4_rescue_distance.png')

    plot_collagen_effect(pt_coll, cell_coll, COMP_DIR / 'L4_collagen_comparison.png')

# ══════════════════════════════════════════════════════════════════════════════
# L5 — SEX EFFECT
# ══════════════════════════════════════════════════════════════════════════════
def layer5_sex(df_nc, df_pt, feat_cols):
    print("\n" + "=" * 70)
    print("L5 — SEX EFFECT")
    print("=" * 70)

    def sex_kw(df_input, label):
        rows = []
        for feat in feat_cols:
            m = df_input[df_input['Gender'] == 'Male'][feat].dropna().values
            f = df_input[df_input['Gender'] == 'Female'][feat].dropna().values
            if len(m) < 3 or len(f) < 3: continue
            try:
                _, p = mannwhitneyu(m, f, alternative='two-sided')
            except:
                p = np.nan
            rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                         'Cohen_d': cohens_d(f, m), 'p': p})
        res = pd.DataFrame(rows)
        res['BH_q'] = bh_fdr(res['p'].values)
        res['Significant'] = res['BH_q'] < ALPHA
        print(f"    {label}: {res['Significant'].sum()}/{len(res)} sig")
        return res

    sex_pt   = sex_kw(df_pt, 'Patient-level (overall)')
    sex_cell = sex_kw(df_nc, 'Cell-level (overall)')
    sex_pt.to_csv(PT_DIR   / 'L5_sex_overall.csv', index=False)
    sex_cell.to_csv(CELL_DIR / 'L5_sex_overall.csv', index=False)

    # Per-disease
    for dis in DISEASE_ORDER:
        sex_kw(df_pt[df_pt['Disease'] == dis], f'Patient-level ({dis})')
        sex_kw(df_nc[df_nc['Disease'] == dis], f'Cell-level ({dis})')

    plot_sex_bars(sex_pt, sex_cell, COMP_DIR / 'L5_sex_comparison.png')

# ══════════════════════════════════════════════════════════════════════════════
# L6 — HYPERTENSION EFFECT
# ══════════════════════════════════════════════════════════════════════════════
def layer6_hypertension(df_nc, df_pt, feat_cols):
    print("\n" + "=" * 70)
    print("L6 — HYPERTENSION EFFECT")
    print("=" * 70)

    def hyp_test(df_input, label):
        df_h = df_input[df_input['Hypertn'].isin(['Yes', 'No'])].copy()
        if len(df_h) < 6:
            print(f"    {label}: insufficient data (n={len(df_h)})")
            return pd.DataFrame()
        n_y = (df_h['Hypertn'] == 'Yes').sum()
        n_n = (df_h['Hypertn'] == 'No').sum()
        rows = []
        for feat in feat_cols:
            hy = df_h[df_h['Hypertn'] == 'Yes'][feat].dropna().values
            no = df_h[df_h['Hypertn'] == 'No'][feat].dropna().values
            if len(hy) < 3 or len(no) < 3: continue
            try:
                _, p = mannwhitneyu(hy, no, alternative='two-sided')
            except:
                p = np.nan
            rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                         'Cohen_d': cohens_d(no, hy), 'p': p})
        if not rows: return pd.DataFrame()
        res = pd.DataFrame(rows)
        res['BH_q'] = bh_fdr(res['p'].values)
        res['Significant'] = res['BH_q'] < ALPHA
        print(f"    {label} (HT={n_y}, No-HT={n_n}): "
              f"{res['Significant'].sum()}/{len(res)} sig")
        return res

    hyp_pt   = hyp_test(df_pt, 'Patient-level')
    hyp_cell = hyp_test(df_nc, 'Cell-level')
    if not hyp_pt.empty:   hyp_pt.to_csv(PT_DIR   / 'L6_hypertension.csv', index=False)
    if not hyp_cell.empty: hyp_cell.to_csv(CELL_DIR / 'L6_hypertension.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# L7 — AORTA DIAMETER
# ══════════════════════════════════════════════════════════════════════════════
def layer7_aorta(df_nc, df_pt, feat_cols):
    print("\n" + "=" * 70)
    print("L7 — AORTA DIAMETER CORRELATIONS (Spearman)")
    print("=" * 70)

    def diam_corr(df_input, label):
        df_d = df_input[df_input['AortaDiam_mm'].notna()].copy()
        if len(df_d) < 5:
            print(f"    {label}: insufficient data")
            return pd.DataFrame()
        rows = []
        for feat in feat_cols:
            sub = df_d[['AortaDiam_mm', feat]].dropna()
            if len(sub) < 5: continue
            r, p = spearmanr(sub['AortaDiam_mm'], sub[feat])
            rows.append({'Feature': feat, 'Organelle': get_organelle(feat),
                         'Spearman_r': r, 'p': p})
        res = pd.DataFrame(rows)
        res['BH_q'] = bh_fdr(res['p'].values)
        res['Significant'] = res['BH_q'] < ALPHA
        print(f"    {label} (n={len(df_d)}): {res['Significant'].sum()}/{len(res)} sig")
        return res

    corr_pt   = diam_corr(df_pt, 'Patient-level')
    corr_cell = diam_corr(df_nc, 'Cell-level')
    if not corr_pt.empty:   corr_pt.to_csv(PT_DIR   / 'L7_aorta_diameter.csv', index=False)
    if not corr_cell.empty: corr_cell.to_csv(CELL_DIR / 'L7_aorta_diameter.csv', index=False)
    if not corr_pt.empty and not corr_cell.empty:
        plot_aorta_comparison(corr_pt, corr_cell, COMP_DIR / 'L7_aorta_comparison.png')

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def summary(df_pt, feat_cols, kw_pt):
    print("\n  Generating summary heatmap...")
    sig   = kw_pt.loc[kw_pt['Significant'], 'Feature'].tolist()
    near  = kw_pt[~kw_pt['Significant']].sort_values('BH_q').head(6)['Feature'].tolist()
    feats = sig + [f for f in near if f not in sig]
    if not feats:
        feats = kw_pt.sort_values('KW_stat', ascending=False).head(15)['Feature'].tolist()

    # FIX 2: fillna(0) before StandardScaler to avoid crash on NaN values
    scaler = StandardScaler()
    X_sc = pd.DataFrame(scaler.fit_transform(df_pt[feat_cols].fillna(0)),
                        columns=feat_cols, index=df_pt.index)
    hm = {DISEASE_LABELS[d]: X_sc[df_pt['Disease'] == d][feats].mean()
          for d in DISEASE_ORDER}
    hm_df = pd.DataFrame(hm, index=feats)

    row_labels = [f.replace('_', ' ') + (' *' if f in set(sig) else '')
                  for f in feats]
    hm_df.index = row_labels

    fig, ax = plt.subplots(figsize=(9, max(7, len(feats) * 0.5)))
    sns.heatmap(hm_df, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Z-score (patient mean)'})
    ax.set_title('Summary — Standardised Group Means\n'
                 '* = significant (KW BH-FDR, patient-level)',
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=15, labelsize=11)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)
    plt.tight_layout()
    save(fig, BASE_DIR / 'SUMMARY_heatmap.png')

    # PCA — FIX 2: fillna(0) before StandardScaler
    X  = df_pt[feat_cols].fillna(0).values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    ev  = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 7))
    for dis in DISEASE_ORDER:
        mask = df_pt['Disease'].values == dis
        ax.scatter(Xp[mask, 0], Xp[mask, 1], c=DISEASE_COLORS[dis],
                   label=DISEASE_LABELS[dis], s=90, alpha=0.8,
                   edgecolors='white', lw=0.5)
        ctr = Xp[mask].mean(axis=0)
        ax.scatter(*ctr, c=DISEASE_COLORS[dis], s=300, marker='D',
                   edgecolors='black', lw=2, zorder=6)
    ax.set_xlabel(f'PC1 ({ev[0] * 100:.1f}%)', fontsize=13)
    ax.set_ylabel(f'PC2 ({ev[1] * 100:.1f}%)', fontsize=13)
    ax.set_title('PCA — Patient-Level Morphological Space\n'
                 'Diamonds = group centroids',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, BASE_DIR / 'SUMMARY_pca.png')

# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def plot_disease_bars(kw, path, title_suffix, level):
    sub = kw.copy()
    sub['-l10q'] = -np.log10(sub['BH_q'].clip(1e-10))
    sub = sub.sort_values('-l10q')
    colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in sub['Organelle']]
    thr = -np.log10(ALPHA)
    n_sig = sub['Significant'].sum()

    fig, ax = plt.subplots(figsize=(11, max(6, len(sub) * 0.5)))
    ax.barh(range(len(sub)), sub['-l10q'], color=colors, edgecolor='white', height=0.7)
    ax.axvline(thr, color='red', ls='--', lw=1.5)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=11)
    ax.set_xlabel('-log₁₀ (BH-FDR q-value)', fontsize=13)
    ax.set_title(f'L1 Disease Effect — {title_suffix}\n{n_sig}/{len(sub)} significant',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    xlim_max = max(sub['-l10q'].max() * 1.2, thr * 1.5)
    ax.set_xlim(0, xlim_max)
    pad = xlim_max * 0.02
    for i, (_, row) in enumerate(sub.iterrows()):
        if row['Significant']:
            ax.text(row['-l10q'] + pad, i, sig_stars(row['BH_q']),
                    va='center', ha='left', fontsize=11, fontweight='bold')
    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, fontsize=10, loc='lower right')
    footer(fig, level)
    plt.tight_layout()
    save(fig, path)


def plot_cohens_d_3panel(kw, path, title_suffix, level):
    pairs = [('d_Healthy_TAA', 'q_Healthy_TAA', 'Non-A vs TAA', '#E74C3C'),
             ('d_Healthy_BAV', 'q_Healthy_BAV', 'Non-A vs BAV', '#F39C12'),
             ('d_TAA_BAV',     'q_TAA_BAV',     'TAA vs BAV',   '#9B59B6')]
    feat_order = (kw[['Feature', 'd_Healthy_TAA']].dropna()
                  .sort_values('d_Healthy_TAA')['Feature'].tolist())

    fig, axes = plt.subplots(1, 3, figsize=(22, max(8, len(feat_order) * 0.55)))
    fig.suptitle(f"Cohen's d — {title_suffix}", fontsize=15, fontweight='bold')

    for ax, (dc, qc, label, color) in zip(axes, pairs):
        sub = (kw.set_index('Feature').reindex(feat_order).reset_index()
               [['Feature', dc, qc, 'Significant']])
        sub[dc]  = sub[dc].fillna(0)
        sub[qc]  = sub[qc].fillna(1)
        sub['pair_sig'] = sub[qc] < ALPHA
        oc = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA') for f in sub['Feature']]
        ax.barh(range(len(sub)), sub[dc], color=oc, edgecolor='white', height=0.7)
        ax.axvline(0, color='black', lw=0.9)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=11)
        ax.set_xlabel("Cohen's d", fontsize=13)
        ax.set_title(label, fontsize=13, fontweight='bold', color=color)
        ax.grid(axis='x', alpha=0.3)
        xlim = ax.get_xlim()
        pad = (xlim[1] - xlim[0]) * 0.025
        for i, (_, row) in enumerate(sub.iterrows()):
            if row['pair_sig']:
                x = row[dc]
                ax.text(x + (pad if x >= 0 else -pad), i, sig_stars(row[qc]),
                        va='center', ha='left' if x >= 0 else 'right',
                        fontsize=11, fontweight='bold')
        cur = ax.get_xlim()
        ax.set_xlim(cur[0] - abs(cur[0]) * 0.15, cur[1] + abs(cur[1]) * 0.15)

    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=12, bbox_to_anchor=(0.5, -0.01))
    footer(fig, level)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, path)


def plot_dunn_heatmap(dunn, kw, path, title_suffix):
    feats = list(dunn.keys())
    if not feats: return
    feat_order = (kw[kw['Feature'].isin(feats)]
                  .sort_values('KW_stat', ascending=False)['Feature'].tolist())
    ncols = min(3, len(feat_order))
    nrows = int(np.ceil(len(feat_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = np.array(axes).flatten() if len(feat_order) > 1 else [axes]
    fig.suptitle(f"Dunn's Post-hoc (Bonferroni) — {title_suffix}\n"
                 "-log₁₀(p)  |  * = p < 0.05",
                 fontsize=13, fontweight='bold')
    for i, feat in enumerate(feat_order):
        ax = axes[i]
        pm  = dunn[feat]
        lpm = -np.log10(pm.values.clip(1e-10))
        np.fill_diagonal(lpm, 0)
        short = [DISEASE_LABELS[g][:7] for g in DISEASE_ORDER]
        sns.heatmap(lpm, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=short, yticklabels=short, ax=ax,
                    annot_kws={'size': 10}, linewidths=0.5)
        for r in range(3):
            for c in range(3):
                if r != c and pm.values[r, c] < ALPHA:
                    ax.text(c + 0.5, r + 0.8, '*', ha='center', fontsize=14, fontweight='bold')
        kw_row = kw[kw['Feature'] == feat]
        q_str = f"q={kw_row['BH_q'].values[0]:.3f}" if len(kw_row) else ''
        ax.set_title(f"{feat.replace('_', ' ')}  ({q_str})", fontsize=9, fontweight='bold')
    for j in range(len(feat_order), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    save(fig, path)


def plot_level_comparison(kw_pt, kw_cell, path, title):
    """Side-by-side -log10(q) bars, patient (top) and cell (bottom)."""
    merged = kw_pt[['Feature', 'BH_q', 'Organelle']].merge(
        kw_cell[['Feature', 'BH_q']], on='Feature', suffixes=('_pt', '_cell'))
    merged['-l10_pt']   = -np.log10(merged['BH_q_pt'].clip(1e-10))
    merged['-l10_cell'] = -np.log10(merged['BH_q_cell'].clip(1e-10))
    merged = merged.sort_values('-l10_pt', ascending=True)
    thr = -np.log10(ALPHA)

    y = np.arange(len(merged))
    h = 0.38

    fig, ax = plt.subplots(figsize=(12, max(7, len(merged) * 0.52)))
    ax.barh(y + h / 2, merged['-l10_pt'],   height=h, color=LEVEL_COLORS['Patient'],
            alpha=0.85, label='Patient-level', edgecolor='white')
    ax.barh(y - h / 2, merged['-l10_cell'], height=h, color=LEVEL_COLORS['Cell'],
            alpha=0.55, label='Cell-level (exploratory)', edgecolor='white')
    ax.axvline(thr, color='red', ls='--', lw=1.5, label=f'q={ALPHA}')
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace('_', ' ') for f in merged['Feature']], fontsize=11)
    ax.set_xlabel('-log₁₀ (BH-FDR q-value)', fontsize=13)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save(fig, path)


def plot_l1_vs_l2(kw1, kw2, path, level_label):
    merged = kw1[['Feature', 'BH_q', 'Organelle']].merge(
        kw2[['Feature', 'BH_q']], on='Feature', suffixes=('_L1', '_L2'))
    merged['-l10_L1'] = -np.log10(merged['BH_q_L1'].clip(1e-10))
    merged['-l10_L2'] = -np.log10(merged['BH_q_L2'].clip(1e-10))
    colors = [ORGANELLE_COLORS.get(o, '#AAA') for o in merged['Organelle']]
    thr = -np.log10(ALPHA)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(merged['-l10_L1'], merged['-l10_L2'],
               c=colors, alpha=0.75, s=70, edgecolors='white', lw=0.5)
    lim = max(merged[['-l10_L1', '-l10_L2']].max()) * 1.08
    ax.plot([0, lim], [0, lim], 'k--', lw=0.9, alpha=0.4)
    ax.axhline(thr, color='red', lw=0.8, ls=':')
    ax.axvline(thr, color='red', lw=0.8, ls=':')
    ax.set_xlabel(f'-log₁₀(q)  L1 raw ({level_label})', fontsize=12)
    ax.set_ylabel(f'-log₁₀(q)  L2 age-corrected ({level_label})', fontsize=12)
    ax.set_title(f'L2 — Age Correction Impact ({level_label})\n'
                 'Above diagonal = signal strengthened after correction',
                 fontsize=12, fontweight='bold')
    for _, row in merged.iterrows():
        if row['-l10_L1'] > thr or row['-l10_L2'] > thr:
            ax.annotate(row['Feature'].replace('_', ' '),
                        (row['-l10_L1'], row['-l10_L2']),
                        fontsize=7, alpha=0.8,
                        xytext=(4, 4), textcoords='offset points')
    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    ax.legend(handles=patches, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, path)


def plot_lmm_forest(lmm, path):
    terms       = ['TAA vs Healthy', 'BAV vs Healthy', 'Age', 'Sex (Male)']
    term_colors = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB']

    # FIX 5: dynamic figure height based on actual max features per term
    max_feats = max((len(lmm[lmm['Term'] == t]) for t in terms), default=6)
    fig, axes = plt.subplots(1, len(terms),
                             figsize=(len(terms) * 5, max(6, max_feats * 0.55)))
    fig.suptitle('L3 — LMM Fixed Effects  (Subject = random intercept)\n'
                 'BH-FDR corrected  |  * = q < 0.05',
                 fontsize=13, fontweight='bold')
    for ax, term, color in zip(axes, terms, term_colors):
        sub = (lmm[lmm['Term'] == term]
               .assign(absC=lambda d: d['Coef'].abs())
               .sort_values('absC', ascending=False))
        if sub.empty:
            ax.set_visible(False); continue
        y = range(len(sub))
        ax.errorbar(sub['Coef'], list(y),
                    xerr=[sub['Coef'] - sub['CI_low'], sub['CI_high'] - sub['Coef']],
                    fmt='o', color=color, ecolor='#555', capsize=4, ms=7, lw=1.5)
        ax.axvline(0, color='black', lw=0.9, ls='--')
        ax.set_yticks(list(y))
        ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=10)
        ax.set_xlabel('Coefficient', fontsize=11)
        ax.set_title(term, fontsize=11, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3, axis='x')
        xlim = ax.get_xlim()
        for j, (_, row) in enumerate(sub.iterrows()):
            if row.get('Significant', False):
                ax.text(xlim[1], j, ' *', va='center', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save(fig, path)


def plot_collagen_effect(pt_coll, cell_coll, path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("L4 — Collagen Effect: Cohen's d  (+Collagen vs NoCollagen)\n"
                 "Top row: Patient-level (Paired Wilcoxon)  |  "
                 "Bottom row: Cell-level (Mann-Whitney, exploratory)",
                 fontsize=13, fontweight='bold')
    for row_idx, (res, level) in enumerate([(pt_coll, 'Patient'), (cell_coll, 'Cell')]):
        for col_idx, dis in enumerate(DISEASE_ORDER):
            ax = axes[row_idx][col_idx]
            sub = res[res['Disease'] == dis].sort_values('Cohen_d')
            if sub.empty:
                ax.set_visible(False); continue
            oc = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA') for f in sub['Feature']]
            ax.barh(range(len(sub)), sub['Cohen_d'], color=oc, edgecolor='white', height=0.7)
            ax.axvline(0, color='black', lw=0.9)
            ax.set_yticks(range(len(sub)))
            ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=8)
            ax.set_xlabel("Cohen's d", fontsize=10)
            ax.set_title(f"{DISEASE_LABELS[dis]}  ({level})\n"
                         f"{sub['Significant'].sum()}/{len(sub)} sig",
                         fontsize=10, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save(fig, path)


def plot_sex_bars(sex_pt, sex_cell, path):
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(sex_pt) * 0.5)))
    fig.suptitle("L5 — Sex Effect: Cohen's d  (Male vs Female)\n"
                 "Left: Patient-level  |  Right: Cell-level (exploratory)",
                 fontsize=13, fontweight='bold')
    for ax, (res, level) in zip(axes, [(sex_pt, 'Patient'), (sex_cell, 'Cell')]):
        sub = res.sort_values('Cohen_d')
        oc = [ORGANELLE_COLORS.get(get_organelle(f), '#AAA') for f in sub['Feature']]
        ax.barh(range(len(sub)), sub['Cohen_d'], color=oc, edgecolor='white', height=0.7)
        ax.axvline(0, color='black', lw=0.9)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f.replace('_', ' ') for f in sub['Feature']], fontsize=11)
        ax.set_xlabel("Cohen's d  (+d = higher in Males)", fontsize=12)
        n_sig = sub['Significant'].sum()
        ax.set_title(f'{level}-level  |  {n_sig}/{len(sub)} significant',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        xlim = ax.get_xlim()
        pad = (xlim[1] - xlim[0]) * 0.025
        for i, (_, row) in enumerate(sub.iterrows()):
            if row['Significant']:
                x = row['Cohen_d']
                ax.text(x + (pad if x >= 0 else -pad), i, sig_stars(row['BH_q']),
                        va='center', ha='left' if x >= 0 else 'right',
                        fontsize=11, fontweight='bold')
        cur = ax.get_xlim()
        ax.set_xlim(cur[0] - abs(cur[0]) * 0.12, cur[1] + abs(cur[1]) * 0.12)
    patches = [mpatches.Patch(color=c, label=o) for o, c in ORGANELLE_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, path)


def plot_aorta_comparison(corr_pt, corr_cell, path):
    merged = corr_pt[['Feature', 'Spearman_r', 'Organelle']].merge(
        corr_cell[['Feature', 'Spearman_r']], on='Feature', suffixes=('_pt', '_cell'))
    merged = merged.sort_values('Spearman_r_pt')
    y = np.arange(len(merged)); h = 0.38

    fig, ax = plt.subplots(figsize=(12, max(7, len(merged) * 0.52)))
    ax.barh(y + h / 2, merged['Spearman_r_pt'],   height=h,
            color=LEVEL_COLORS['Patient'], alpha=0.85, label='Patient-level', edgecolor='white')
    ax.barh(y - h / 2, merged['Spearman_r_cell'], height=h,
            color=LEVEL_COLORS['Cell'], alpha=0.55, label='Cell-level (exploratory)', edgecolor='white')
    ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace('_', ' ') for f in merged['Feature']], fontsize=11)
    ax.set_xlabel('Spearman r  (vs Aorta Diameter mm)', fontsize=13)
    ax.set_title('L7 — Aorta Diameter Correlations: Patient vs Cell Level',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save(fig, path)


def save_dunn_csv(dunn, path):
    rows = []
    for feat, pm in dunn.items():
        for a in DISEASE_ORDER:
            for b in DISEASE_ORDER:
                if a < b:
                    rows.append({'Feature': feat, 'Group_A': a, 'Group_B': b,
                                 'Dunn_p': pm.loc[a, b],
                                 'Significant': pm.loc[a, b] < ALPHA})
    pd.DataFrame(rows).to_csv(path, index=False)

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def print_final_summary(results):
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — SIGNIFICANT FEATURES PER LAYER × LEVEL")
    print("=" * 70)
    header = f"{'Layer':<35} {'Patient-level':>14} {'Cell-level':>12}"
    print(header)
    print("-" * 65)
    for label, pt_n, pt_total, cell_n, cell_total in results:
        print(f"  {label:<33} {pt_n:>5}/{pt_total:<8} {cell_n:>5}/{cell_total}")
    print("\n  Patient-level = PRIMARY (report in main text)")
    print("  Cell-level    = EXPLORATORY (report in supplementary)")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    for d in [BASE_DIR, PT_DIR, CELL_DIR, LMM_DIR, COMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DUAL-LEVEL STATISTICAL ANALYSIS")
    print("Thesis: 3D Phenotyping of Mitochondrial Morphology — Aortic Aneurysm")
    print("=" * 70)

    df, df_nc, df_pt, feat_cols = load_data()

    kw_pt_l1, kw_cell_l1, dunn_pt, dunn_cell = layer1(df_nc, df_pt, feat_cols)
    kw_pt_l2, kw_cell_l2                     = layer2(df_nc, df_pt, feat_cols, kw_pt_l1, kw_cell_l1)
    lmm                                       = layer3_lmm(df, feat_cols, kw_pt_l1)
    layer4_collagen(df, feat_cols)   # FIX 4: removed unused df_pt argument
    layer5_sex(df_nc, df_pt, feat_cols)
    layer6_hypertension(df_nc, df_pt, feat_cols)
    layer7_aorta(df_nc, df_pt, feat_cols)
    summary(df_pt, feat_cols, kw_pt_l1)

    # Summary table
    lmm_disease_sig = (lmm[(lmm['Term'].isin(['TAA vs Healthy', 'BAV vs Healthy']))
                            & lmm['Significant']]['Feature'].nunique()
                       if lmm is not None else 0)
    print_final_summary([
        ('L1 Disease effect',
         kw_pt_l1['Significant'].sum(), len(kw_pt_l1),
         kw_cell_l1['Significant'].sum(), len(kw_cell_l1)),
        ('L2 Age-corrected disease effect',
         kw_pt_l2['Significant'].sum(), len(kw_pt_l2),
         kw_cell_l2['Significant'].sum(), len(kw_cell_l2)),
        ('L3 LMM (disease terms)',
         lmm_disease_sig, len(kw_pt_l1),
         'N/A', '(cell-level, clustered)'),
    ])

    print(f"\nAll outputs in: {BASE_DIR}/")
    for sub in [PT_DIR, CELL_DIR, LMM_DIR, COMP_DIR]:
        files = sorted(sub.glob('*'))
        if files:
            print(f"  {sub.name}/  ({len(files)} files)")


if __name__ == '__main__':
    main()
