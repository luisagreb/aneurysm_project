"""
Pairwise Boxplot Analysis with Significance Brackets
=====================================================
For each feature:
  - Boxplot with 4 groups: Healthy No Coll, Healthy +Coll, TAA No Coll, TAA +Coll
  - Significance brackets for 4 comparisons (FDR-corrected across all features)
  - Age correlation annotation (Spearman ρ + FDR p-value)
  - Mean ± SD diamond marker
  - One plot per feature saved as PNG

Output: classification_results/pairwise_boxplots/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import os
import warnings
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, spearmanr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_FILE = 'outputs/Advanced_Features_Raw_200.csv'
OUT_DIR       = 'classification_results/smc_analysis/pairwise_boxplots'
ALPHA         = 0.05
os.makedirs(OUT_DIR, exist_ok=True)

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
    'Nucleus_Elongation','Nucleus_Flatness','Nucleus_Solidity'
]

GROUP_ORDER  = ['Healthy_NoCollagen', 'Healthy_Collagen',
                'TAA_NoCollagen',     'TAA_Collagen']
GROUP_LABELS = ['Healthy\nNo Coll', 'Healthy\n+Coll',
                'TAA\nNo Coll',     'TAA\n+Coll']

PALETTE = {
    'Healthy_NoCollagen': '#A8D5B5',
    'Healthy_Collagen':   '#2E8B57',
    'TAA_NoCollagen':     '#FFAB9F',
    'TAA_Collagen':       '#D73B2F',
}

COMPARISONS = [
    ('Healthy_NoCollagen', 'TAA_NoCollagen',   'Disease\n(−Col)'),
    ('Healthy_Collagen',   'TAA_Collagen',     'Disease\n(+Col)'),
    ('TAA_NoCollagen',     'TAA_Collagen',     'Rescue\n(TAA)'),
    ('Healthy_NoCollagen', 'Healthy_Collagen', 'Collagen\n(Healthy)'),
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

def shorten(feat):
    return (feat.replace('_µm³','').replace('_µm²','').replace('_µm','')
               .replace('_ratio','').replace('_n','').replace('_',' '))

def stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df_all = pd.read_csv(FEATURES_FILE)
df_all['SpecimenID'] = df_all['CellName'].apply(extract_id).apply(lambda x: normalize(x))

na_norm = [normalize(x) for x in NON_ANEURYSMAL]
an_norm = [normalize(x) for x in ANEURYSMAL]

df = df_all[df_all['SpecimenID'].isin(na_norm + an_norm)].copy()
df['SMC_Group']   = df['SpecimenID'].apply(lambda x: 'Healthy' if x in na_norm else 'TAA')
df['Group_Label'] = df['SMC_Group'] + '_' + df['Collagen_Status']
feat_cols = [c for c in FEATURE_COLS if c in df.columns]

print(f"Dataset: {len(df)} cells, {df['SpecimenID'].nunique()} specimens")
print(df.groupby(['SMC_Group','Collagen_Status']).size().to_string())

# ── Step 1: Run all pairwise tests with Shapiro-Wilk normality check ──────────
# For each feature × comparison:
#   1. Test normality of each group with Shapiro-Wilk (p > 0.05 = normal)
#   2. If BOTH groups are normal → T-test (parametric)
#   3. If EITHER group is non-normal → Mann-Whitney U (non-parametric)
# Then apply FDR correction globally across all features × comparisons.
print("\nRunning pairwise tests (Shapiro-Wilk → T-test or Mann-Whitney)...")

def is_normal(data, alpha=0.05):
    """
    Test normality with Shapiro-Wilk.
    Returns True if data is normally distributed (p > alpha).
    Shapiro-Wilk requires between 3 and 5000 samples.
    """
    if len(data) < 3:
        return False
    if len(data) > 5000:
        # Shapiro-Wilk unreliable for very large samples — assume non-normal
        return False
    _, p = shapiro(data)
    return p > alpha

def adaptive_test(d1, d2):
    """
    Choose between T-test and Mann-Whitney based on normality of both groups.
    Returns (p_value, test_used).
    """
    normal1 = is_normal(d1)
    normal2 = is_normal(d2)
    if normal1 and normal2:
        # Both normal — use independent samples T-test
        _, p = ttest_ind(d1, d2, equal_var=False)  # Welch's t-test (unequal variance)
        return p, 'T-test'
    else:
        # At least one non-normal — use Mann-Whitney
        _, p = mannwhitneyu(d1, d2, alternative='two-sided')
        return p, 'Mann-Whitney'

raw_results = []
test_choices = {}  # track which test was used for each feature × comparison

for feat in feat_cols:
    for g1, g2, label in COMPARISONS:
        d1 = df[df['Group_Label']==g1][feat].dropna()
        d2 = df[df['Group_Label']==g2][feat].dropna()
        if len(d1) < 3 or len(d2) < 3:
            raw_results.append({
                'Feature': feat, 'Comparison': label,
                'G1': g1, 'G2': g2,
                'Mean1': np.nan, 'Mean2': np.nan,
                'Median1': np.nan, 'Median2': np.nan,
                'SD1': np.nan, 'SD2': np.nan,
                'Cohens_d': np.nan, 'p_value': 1.0,
                'Test_used': 'N/A',
                'Normal_g1': False, 'Normal_g2': False,
                'N1': len(d1), 'N2': len(d2)
            })
            continue

        normal1 = is_normal(d1)
        normal2 = is_normal(d2)
        p, test_used = adaptive_test(d1, d2)
        cohens_d = (d2.mean() - d1.mean()) / np.sqrt((d1.std()**2 + d2.std()**2) / 2)

        raw_results.append({
            'Feature': feat, 'Comparison': label,
            'G1': g1, 'G2': g2,
            'Mean1': d1.mean(), 'Mean2': d2.mean(),
            'Median1': d1.median(), 'Median2': d2.median(),
            'SD1': d1.std(), 'SD2': d2.std(),
            'Cohens_d': cohens_d, 'p_value': p,
            'Test_used': test_used,
            'Normal_g1': normal1, 'Normal_g2': normal2,
            'N1': len(d1), 'N2': len(d2)
        })
        test_choices[(feat, label)] = test_used

# Print test choice summary
n_ttest = sum(1 for v in test_choices.values() if v == 'T-test')
n_mw    = sum(1 for v in test_choices.values() if v == 'Mann-Whitney')
print(f"  T-test used:       {n_ttest} comparisons (both groups normal)")
print(f"  Mann-Whitney used: {n_mw} comparisons (at least one non-normal)")

pw_df = pd.DataFrame(raw_results)

# FDR correction across ALL tests simultaneously
_, fdr, _, _ = multipletests(pw_df['p_value'].fillna(1), method='fdr_bh')
pw_df['p_FDR'] = fdr
pw_df['Significant'] = fdr < ALPHA
pw_df.to_csv(f'{OUT_DIR}/pairwise_results.csv', index=False)

print("Significant comparisons (FDR < 0.05):")
for _, (g1, g2, label) in enumerate(COMPARISONS):
    n = pw_df[(pw_df['Comparison']==label) & pw_df['Significant']]['Feature'].nunique()
    print(f"  {label.replace(chr(10),' '):25s}: {n} features")

# ── Step 2: Age correlation for each feature ──────────────────────────────────
print("\nComputing age correlations...")
age_results = []
for feat in feat_cols:
    sub = df[['Age', feat]].dropna()
    if len(sub) < 10:
        age_results.append({'Feature': feat, 'Spearman_r': np.nan, 'p_value': 1.0})
        continue
    r, p = spearmanr(sub[feat], sub['Age'])
    age_results.append({'Feature': feat, 'Spearman_r': r, 'p_value': p})

age_df = pd.DataFrame(age_results)
_, fdr_age, _, _ = multipletests(age_df['p_value'].fillna(1), method='fdr_bh')
age_df['p_FDR'] = fdr_age
age_df['Significant'] = fdr_age < ALPHA
age_df = age_df.set_index('Feature')

# ── Step 3: Sex effect for each feature ──────────────────────────────────────
print("\nComputing sex effects (Male vs Female)...")
sex_results = []
for feat in feat_cols:
    d_male   = df[df['Gender']=='Male'][feat].dropna()
    d_female = df[df['Gender']=='Female'][feat].dropna()
    if len(d_male) < 3 or len(d_female) < 3:
        sex_results.append({'Feature': feat, 'Cohens_d': np.nan,
                            'p_value': 1.0, 'Test_used': 'N/A'})
        continue
    p, test_used = adaptive_test(d_male, d_female)
    cohens_d = (d_male.mean() - d_female.mean()) / np.sqrt(
        (d_male.std()**2 + d_female.std()**2) / 2)
    sex_results.append({
        'Feature': feat,
        'Mean_Male': d_male.mean(), 'Mean_Female': d_female.mean(),
        'Cohens_d': cohens_d,
        'Test_used': test_used,
        'p_value': p
    })

sex_df = pd.DataFrame(sex_results)
_, fdr_sex, _, _ = multipletests(sex_df['p_value'].fillna(1), method='fdr_bh')
sex_df['p_FDR'] = fdr_sex
sex_df['Significant'] = fdr_sex < ALPHA
sex_df = sex_df.set_index('Feature')

n_sex_sig = sex_df['Significant'].sum()
print(f"  Significant sex effects (FDR < 0.05): {n_sex_sig} features")
print([shorten(f) for f in sex_df[sex_df['Significant']].index.tolist()])
# ── Step 4: Plot one figure per feature ───────────────────────────────────────

def add_significance_bracket(ax, x1, x2, y, h, p_fdr, fontsize=9):
    """Draw a bracket only for significant comparisons."""
    s = stars(p_fdr)
    if s == 'ns':
        return
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='#333333')
    ax.text((x1+x2)/2, y+h, s, ha='center', va='bottom',
            fontsize=fontsize, color='#333333', fontweight='bold')

for feat in feat_cols:
    fig, ax = plt.subplots(figsize=(7, 5))

    # ── Draw boxplots ──────────────────────────────────────────────────────
    data_by_group = []
    positions = []
    for i, grp in enumerate(GROUP_ORDER):
        vals = df[df['Group_Label']==grp][feat].dropna().values
        data_by_group.append(vals)
        positions.append(i)

    bp = ax.boxplot(data_by_group,
                    positions=positions,
                    widths=0.5,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=4,
                                    markerfacecolor='none', alpha=0.5),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    boxprops=dict(linewidth=1.2))

    for patch, grp in zip(bp['boxes'], GROUP_ORDER):
        patch.set_facecolor(PALETTE[grp])
        patch.set_alpha(0.85)

    # ── Mean ± SD diamond marker ───────────────────────────────────────────
    for i, (grp, vals) in enumerate(zip(GROUP_ORDER, data_by_group)):
        if len(vals) > 0:
            mean_val = np.mean(vals)
            ax.plot(i, mean_val, marker='D', color='black',
                    markersize=6, zorder=5, label='Mean ± SD' if i == 0 else '')
            sd_val = np.std(vals)
            ax.plot([i, i], [mean_val - sd_val, mean_val + sd_val],
                    color='black', linewidth=1.5, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_LABELS, fontsize=10)
    ax.set_ylabel(feat, fontsize=10)
    ax.set_title(shorten(feat), fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linewidth=0.8)

    # ── Significance brackets ──────────────────────────────────────────────
    all_vals = np.concatenate([v for v in data_by_group if len(v) > 0])
    y_max = np.percentile(all_vals, 97) if len(all_vals) > 0 else 1
    y_range = y_max - np.min(all_vals) if len(all_vals) > 0 else 1
    h   = y_range * 0.04
    gap = y_range * 0.08

    pos_map = {grp: i for i, grp in enumerate(GROUP_ORDER)}
    bracket_levels = {
        ('Healthy_NoCollagen', 'TAA_NoCollagen'):   y_max + gap,
        ('Healthy_Collagen',   'TAA_Collagen'):     y_max + gap * 2,
        ('TAA_NoCollagen',     'TAA_Collagen'):     y_max + gap * 3,
        ('Healthy_NoCollagen', 'Healthy_Collagen'): y_max + gap * 4,
    }

    feat_pw = pw_df[pw_df['Feature'] == feat]
    for g1, g2, label in COMPARISONS:
        row = feat_pw[(feat_pw['G1']==g1) & (feat_pw['G2']==g2)]
        if len(row) == 0:
            continue
        p_fdr = row['p_FDR'].values[0]
        add_significance_bracket(ax, pos_map[g1], pos_map[g2],
                                 bracket_levels[(g1, g2)], h, p_fdr)

    ax.set_ylim(bottom=ax.get_ylim()[0], top=y_max + gap * 5 + h * 2)

    # ── Mean ± SD legend only ──────────────────────────────────────────────
    ax.legend(handles=[plt.Line2D([0], [0], marker='D', color='black',
                                  markersize=6, linewidth=0, label='Mean ± SD')],
              loc='upper right', fontsize=8, framealpha=0.9)

    # ── Sex effect annotation (bottom-right, only if significant) ──────────
    if feat in sex_df.index:
        sex_row = sex_df.loc[feat]
        if sex_row['Significant'] and not np.isnan(sex_row['Cohens_d']):
            direction = '♂ > ♀' if sex_row['Cohens_d'] > 0 else '♀ > ♂'
            s = stars(sex_row['p_FDR'])
            ax.annotate(f'Sex: {direction} {s}',
                        xy=(0.98, 0.02), xycoords='axes fraction',
                        ha='right', va='bottom', fontsize=9,
                        color='#7B68EE', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='#7B68EE',
                                  alpha=0.9, linewidth=1))

    plt.tight_layout()

    safe_feat = feat.replace('/', '_')
    plt.savefig(f'{OUT_DIR}/{safe_feat}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {safe_feat}.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\nGenerating summary table...")
summary_rows = []
for feat in feat_cols:
    row = {'Feature': feat, 'Short': shorten(feat)}
    feat_pw = pw_df[pw_df['Feature'] == feat]
    for g1, g2, label in COMPARISONS:
        r = feat_pw[(feat_pw['G1']==g1) & (feat_pw['G2']==g2)]
        lab = label.replace('\n',' ')
        if len(r):
            row[f'{lab}_p_FDR'] = round(r['p_FDR'].values[0], 4)
            row[f'{lab}_sig']   = r['Significant'].values[0]
            row[f'{lab}_d']     = round(r['Cohens_d'].values[0], 3)
    if feat in age_df.index:
        row['Age_rho']  = round(age_df.loc[feat, 'Spearman_r'], 3)
        row['Age_FDR']  = round(age_df.loc[feat, 'p_FDR'], 4)
        row['Age_sig']  = age_df.loc[feat, 'Significant']
    if feat in sex_df.index and not np.isnan(sex_df.loc[feat, 'Cohens_d']):
        row['Sex_d']    = round(sex_df.loc[feat, 'Cohens_d'], 3)
        row['Sex_FDR']  = round(sex_df.loc[feat, 'p_FDR'], 4)
        row['Sex_sig']  = sex_df.loc[feat, 'Significant']
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'{OUT_DIR}/summary_table.csv', index=False)

print("\n" + "="*60)
print("DONE")
print("="*60)
print(f"Plots saved to: {OUT_DIR}/")
print(f"Summary table: {OUT_DIR}/summary_table.csv")
print(f"Full results:  {OUT_DIR}/pairwise_results.csv")

# Print quick summary
print("\nSignificant features per comparison (FDR < 0.05):")
for _, (g1, g2, label) in enumerate(COMPARISONS):
    sig_feats = pw_df[(pw_df['Comparison']==label) & pw_df['Significant']]['Feature'].tolist()
    print(f"\n  {label.replace(chr(10),' ')}:")
    for f in sig_feats:
        d = pw_df[(pw_df['Feature']==f) & (pw_df['Comparison']==label)]['Cohens_d'].values[0]
        print(f"    {shorten(f):35s} d={d:.2f}")