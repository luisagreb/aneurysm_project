"""
5-Layer Statistical Analysis of SMC Morphology
===============================================
Layer 1 — Raw adaptive test (Shapiro-Wilk → T-test or Mann-Whitney)
Layer 2 — Age-residualized adaptive test
Layer 3 — Linear Mixed Model (age, sex, collagen, patient clustering)
Layer 4 — Collagen rescue (within TAA only)
Layer 5 — Sex effect (Male vs Female, overall + stratified by disease group)

Outputs: classification_results/4layer_analysis/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import re
import os
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, spearmanr
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_FILE = 'outputs/Advanced_Features_Raw_200.csv'
OUT_DIR       = 'classification_results/smc_analysis/5layer'
ALPHA         = 0.05

os.makedirs(OUT_DIR, exist_ok=True)
for sub in ['layer1','layer2','layer3','layer4','layer5_sex','summary']:
    os.makedirs(f'{OUT_DIR}/{sub}', exist_ok=True)

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

GROUP_ORDER  = ['Non-Aneurysmal_NoCollagen','Non-Aneurysmal_Collagen',
                'Aneurysmal_NoCollagen','Aneurysmal_Collagen']
GROUP_LABELS = ['Non-A\n-Col','Non-A\n+Col','TAA\n-Col','TAA\n+Col']
PALETTE = {
    'Non-Aneurysmal_NoCollagen': '#B2DFDB',
    'Non-Aneurysmal_Collagen':   '#00796B',
    'Aneurysmal_NoCollagen':     '#FFCCBC',
    'Aneurysmal_Collagen':       '#E64A19',
}

def shorten(feat):
    return feat.replace('_', ' ')

def normalize(sid):
    return re.sub(r'-0*(\d+)$', r'-\1', sid.lower()) if sid else None

def extract_id(cell_name):
    m = re.search(r'(0[123][A-Za-z]+-\d+)', str(cell_name))
    if not m:
        m = re.search(r'(0[123][A-Za-z]+)(\d+)', str(cell_name))
        if not m: return None
        return (m.group(1) + '-' + m.group(2)).lower()
    return m.group(1).lower()

def is_normal(data, alpha=0.05):
    """Shapiro-Wilk normality test. Returns True if p > alpha (normal)."""
    if len(data) < 3 or len(data) > 5000:
        return False
    _, p = shapiro(data)
    return p > alpha

def adaptive_test(d1, d2):
    """
    Test normality of both groups with Shapiro-Wilk.
    Both normal → Welch T-test. Either non-normal → Mann-Whitney U.
    Returns (p_value, test_name).
    """
    if is_normal(d1) and is_normal(d2):
        _, p = ttest_ind(d1, d2, equal_var=False)
        return p, 'T-test'
    else:
        _, p = mannwhitneyu(d1, d2, alternative='two-sided')
        return p, 'Mann-Whitney'

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df_all = pd.read_csv(FEATURES_FILE)
df_all['SpecimenID'] = df_all['CellName'].apply(extract_id).apply(lambda x: normalize(x))

na_norm = [normalize(x) for x in NON_ANEURYSMAL]
an_norm = [normalize(x) for x in ANEURYSMAL]

df = df_all[df_all['SpecimenID'].isin(na_norm + an_norm)].copy()
df['SMC_Group']    = df['SpecimenID'].apply(lambda x: 'Non-Aneurysmal' if x in na_norm else 'Aneurysmal')
df['Group_Label']  = df['SMC_Group'] + '_' + df['Collagen_Status']
df['Subject']      = df['SpecimenID']
df['Disease_bin']  = (df['SMC_Group'] == 'Aneurysmal').astype(int)
df['Collagen_bin'] = (df['Collagen_Status'] == 'Collagen').astype(int)
df['Gender_bin']   = (df['Gender'] == 'Male').astype(int)

feat_cols = [c for c in FEATURE_COLS if c in df.columns]

print(f"Dataset: {len(df)} cells, {df['SpecimenID'].nunique()} specimens")
print(df.groupby(['SMC_Group','Collagen_Status']).size().to_string())
print(f"\nAge by group:")
print(df.groupby('SMC_Group')['Age'].describe()[['mean','std','min','max']].round(1).to_string())
print(f"\nSex distribution:")
print(df.groupby(['SMC_Group','Gender']).size().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Raw adaptive test
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("LAYER 1 — Raw adaptive test (Shapiro-Wilk -> T-test or Mann-Whitney)")
print("="*60)

def run_adaptive(df, feat_cols, g1_label, g2_label, group_col='Group_Label'):
    results = []
    for feat in feat_cols:
        d1 = df[df[group_col]==g1_label][feat].dropna()
        d2 = df[df[group_col]==g2_label][feat].dropna()
        if len(d1) < 3 or len(d2) < 3:
            continue
        p, test_used = adaptive_test(d1, d2)
        cohens_d = (d2.mean() - d1.mean()) / np.sqrt((d1.std()**2 + d2.std()**2) / 2)
        results.append({
            'Feature': feat,
            'Mean_g1': d1.mean(), 'Mean_g2': d2.mean(),
            'Cohens_d': cohens_d, 'Test_used': test_used, 'p_value': p
        })
    res_df = pd.DataFrame(results)
    _, fdr, _, _ = multipletests(res_df['p_value'], method='fdr_bh')
    res_df['p_FDR'] = fdr
    res_df['Significant'] = fdr < ALPHA
    return res_df.sort_values('p_value')

df['Disease_Group'] = df['SMC_Group']
l1_disease = run_adaptive(df, feat_cols, 'Non-Aneurysmal', 'Aneurysmal', group_col='Disease_Group')
l1_sig = l1_disease[l1_disease['Significant']]['Feature'].tolist()
n_ttest = (l1_disease['Test_used'] == 'T-test').sum()
n_mw    = (l1_disease['Test_used'] == 'Mann-Whitney').sum()
print(f"Significant features: {len(l1_sig)} | T-test: {n_ttest} | Mann-Whitney: {n_mw}")
print([shorten(f) for f in l1_sig])
l1_disease.to_csv(f'{OUT_DIR}/layer1/layer1_raw_disease.csv', index=False)

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

fig, ax = plt.subplots(figsize=(11, 6))
plot_df = l1_disease.sort_values('Cohens_d').reset_index(drop=True)
ax.barh([shorten(f) for f in plot_df['Feature']], plot_df['Cohens_d'],
        color=['#E64A19' if s else '#AAAAAA' for s in plot_df['Significant']],
        edgecolor='white', height=0.7)
ax.axvline(0, color='black', linewidth=1)

# Significance stars next to each bar
for i, row in plot_df.iterrows():
    s = sig_stars(row['p_FDR'])
    if s:
        x = row['Cohens_d']
        offset = 0.02 if x >= 0 else -0.02
        ha = 'left' if x >= 0 else 'right'
        ax.text(x + offset, i, s, va='center', ha=ha,
                fontsize=10, color='#333333', fontweight='bold')

ax.set_xlabel("Cohen's d (TAA - Non-Aneurysmal)", fontsize=11)
ax.set_title("Layer 1 - Raw Disease Effect (Shapiro-Wilk -> T-test or Mann-Whitney)\nPositive = higher in TAA",
             fontsize=12, fontweight='bold')
ax.legend(handles=[mpatches.Patch(color='#E64A19', label='FDR < 0.05'),
                   mpatches.Patch(color='#AAAAAA', label='FDR >= 0.05')])
ax.annotate('* p<0.05  ** p<0.01  *** p<0.001  (FDR-corrected)',
            xy=(0.99, 0.01), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=8, color='#555555')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/layer1/layer1_disease_effect.png', dpi=300, bbox_inches='tight')
plt.close()
print("Layer 1 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Age-residualized adaptive test
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("LAYER 2 — Age-residualized test")
print("="*60)

def residualize_age(df, feat_cols):
    df_resid = df.copy()
    age_effects = {}
    for feat in feat_cols:
        sub = df[['Age', feat]].dropna()
        if len(sub) < 10:
            continue
        model = LinearRegression()
        model.fit(sub[['Age']], sub[feat])
        predicted = model.predict(df[['Age']].fillna(df['Age'].mean()))
        df_resid[feat] = df[feat] - predicted
        age_effects[feat] = {'slope': model.coef_[0], 'intercept': model.intercept_}
    return df_resid, age_effects

df_resid, age_effects = residualize_age(df, feat_cols)
age_eff_df = pd.DataFrame(age_effects).T.reset_index()
age_eff_df.columns = ['Feature','Slope_per_year','Intercept']

l2_disease = run_adaptive(df_resid, feat_cols, 'Non-Aneurysmal', 'Aneurysmal', group_col='Disease_Group')
l2_sig = l2_disease[l2_disease['Significant']]['Feature'].tolist()
lost = set(l1_sig) - set(l2_sig)
kept = set(l1_sig) & set(l2_sig)
new  = set(l2_sig) - set(l1_sig)
print(f"Significant after age correction: {len(l2_sig)}")
print(f"KEPT (robust): {len(kept)} | LOST (age-driven): {len(lost)} | NEW: {len(new)}")
print(f"Kept: {[shorten(f) for f in kept]}")
print(f"Lost: {[shorten(f) for f in lost]}")
l2_disease.to_csv(f'{OUT_DIR}/layer2/layer2_age_corrected.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
for ax, res_df, title in zip(axes, [l1_disease, l2_disease],
    ['Layer 1: Raw', 'Layer 2: Age-corrected residuals']):
    plot_df = res_df.set_index('Feature').reindex(feat_cols).reset_index()
    ax.barh([shorten(f) for f in plot_df['Feature']], plot_df['Cohens_d'],
            color=['#E64A19' if s else '#AAAAAA' for s in plot_df['Significant']],
            edgecolor='white', height=0.7)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Cohen's d (TAA - Non-Aneurysmal)", fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    for i, row in plot_df.iterrows():
        s = sig_stars(row['p_FDR'])
        if s:
            x = row['Cohens_d']
            offset = 0.02 if x >= 0 else -0.02
            ha = 'left' if x >= 0 else 'right'
            ax.text(x + offset, i, s, va='center', ha=ha,
                    fontsize=9, color='#333333', fontweight='bold')
axes[1].legend(handles=[mpatches.Patch(color='#E64A19', label='FDR < 0.05'),
                         mpatches.Patch(color='#AAAAAA', label='FDR >= 0.05')], loc='lower right')
fig.suptitle("Layer 1 vs Layer 2: Impact of Age Correction", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/layer2/layer2_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
age_eff_sorted = age_eff_df.sort_values('Slope_per_year')
ax.barh([shorten(f) for f in age_eff_sorted['Feature']], age_eff_sorted['Slope_per_year'],
        color=['#E64A19' if f in l1_sig else '#AAAAAA' for f in age_eff_sorted['Feature']],
        edgecolor='white', height=0.7)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Change per year of age', fontsize=11)
ax.set_title('Age Effect on Each Feature\n(red = significant in Layer 1)', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/layer2/layer2_age_slopes.png', dpi=300, bbox_inches='tight')
plt.close()
print("Layer 2 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Full Linear Mixed Model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("LAYER 3 — Linear Mixed Model")
print("="*60)

lmm_results = []
for feat in feat_cols:
    sub = df[['Subject','SMC_Group','Collagen_Status','Age','Gender',
              'Disease_bin','Collagen_bin','Gender_bin', feat]].dropna()
    if len(sub) < 20 or sub['Subject'].nunique() < 4:
        continue
    sub = sub.copy()
    sub['feat'] = sub[feat]
    try:
        md = smf.mixedlm(
            "feat ~ Disease_bin + Collagen_bin + Disease_bin:Collagen_bin + Age + Gender_bin",
            sub, groups=sub["Subject"])
        res = md.fit(reml=True, method='powell', maxiter=300)
        lmm_results.append({
            'Feature':          feat,
            'Disease_coef':     res.params.get('Disease_bin', np.nan),
            'Disease_pval':     res.pvalues.get('Disease_bin', np.nan),
            'Collagen_coef':    res.params.get('Collagen_bin', np.nan),
            'Collagen_pval':    res.pvalues.get('Collagen_bin', np.nan),
            'Interaction_coef': res.params.get('Disease_bin:Collagen_bin', np.nan),
            'Interaction_pval': res.pvalues.get('Disease_bin:Collagen_bin', np.nan),
            'Age_coef':         res.params.get('Age', np.nan),
            'Age_pval':         res.pvalues.get('Age', np.nan),
            'Gender_coef':      res.params.get('Gender_bin', np.nan),
            'Gender_pval':      res.pvalues.get('Gender_bin', np.nan),
            'ICC':              res.cov_re.iloc[0,0] / (res.cov_re.iloc[0,0] + res.scale)
                                if hasattr(res.cov_re, 'iloc') else np.nan,
            'Converged':        res.converged,
            'N_cells':          len(sub),
            'N_subjects':       sub['Subject'].nunique(),
        })
    except Exception as e:
        print(f"  LMM failed for {feat}: {e}")

lmm_df = pd.DataFrame(lmm_results)
for effect in ['Disease','Collagen','Interaction','Age','Gender']:
    pvals = lmm_df[f'{effect}_pval'].fillna(1)
    _, fdr, _, _ = multipletests(pvals, method='fdr_bh')
    lmm_df[f'{effect}_FDR'] = fdr
    lmm_df[f'{effect}_sig'] = fdr < ALPHA

lmm_df.to_csv(f'{OUT_DIR}/layer3/layer3_lmm_results.csv', index=False)
print("\nSignificant features (FDR < 0.05):")
for effect in ['Disease','Collagen','Interaction','Age','Gender']:
    sig = lmm_df[lmm_df[f'{effect}_sig']==True]['Feature'].tolist()
    print(f"  {effect:12s}: {len(sig)} -- {[shorten(f) for f in sig]}")

fig, ax = plt.subplots(figsize=(10, 7))
forest_df = lmm_df.sort_values('Disease_coef')
ax.barh([shorten(f) for f in forest_df['Feature']], forest_df['Disease_coef'],
        color=['#E64A19' if s else '#AAAAAA' for s in forest_df['Disease_sig']],
        edgecolor='white', height=0.7)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('LMM coefficient (TAA effect)', fontsize=10)
ax.set_title('LMM Disease Effect\n(controlling for age, sex, collagen, patient clustering)',
             fontsize=12, fontweight='bold')
ax.legend(handles=[mpatches.Patch(color='#E64A19', label='FDR < 0.05'),
                   mpatches.Patch(color='#AAAAAA', label='FDR >= 0.05')])
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/layer3/layer3_lmm_forest.png', dpi=300, bbox_inches='tight')
plt.close()

effects = ['Disease','Collagen','Interaction','Age','Gender']
coef_matrix = pd.DataFrame(index=lmm_df['Feature'], columns=effects, dtype=float)
pval_matrix = pd.DataFrame(index=lmm_df['Feature'], columns=effects, dtype=float)
sig_matrix  = pd.DataFrame(index=lmm_df['Feature'], columns=effects, dtype=bool)
for e in effects:
    lmm_indexed = lmm_df.set_index('Feature')
    coef_matrix[e] = lmm_indexed[f'{e}_coef'].values
    pval_matrix[e] = lmm_indexed[f'{e}_FDR'].values
    sig_matrix[e]  = lmm_indexed[f'{e}_sig'].values

# Use -log10(p_FDR) * sign(coef): intensity = significance, direction = effect sign
# Cap at -log10(0.001)=3 to avoid extreme values; anything below p=0.05 gets value < 1.3
sign_matrix = np.sign(coef_matrix.astype(float))
logp_matrix = -np.log10(pval_matrix.astype(float).clip(lower=1e-10))
score_matrix = sign_matrix * logp_matrix
# Symmetric color scale: 0=no effect, ±log10(0.05)≈1.3 = significance threshold, cap at ±3
vmax_score = 3.0

fig, ax = plt.subplots(figsize=(8, max(6, len(feat_cols)*0.45)))
sns.heatmap(score_matrix.astype(float), ax=ax, cmap='RdBu_r', center=0,
            vmin=-vmax_score, vmax=vmax_score,
            xticklabels=effects, yticklabels=[shorten(f) for f in lmm_df['Feature']],
            linewidths=0.4, linecolor='#cccccc',
            cbar_kws={'label': '−log₁₀(p_FDR) × sign(coef)'})

# Add vertical reference lines at ±log10(0.05) on colorbar to mark FDR threshold
sig_thresh = -np.log10(0.05)  # ≈1.301
cbar = ax.collections[0].colorbar
cbar.ax.axhline(sig_thresh,  color='black', linewidth=1.2, linestyle='--')
cbar.ax.axhline(-sig_thresh, color='black', linewidth=1.2, linestyle='--')
cbar.ax.text(2.5, sig_thresh,  'p=0.05', va='bottom', ha='right', fontsize=7, color='black')
cbar.ax.text(2.5, -sig_thresh, 'p=0.05', va='top',    ha='right', fontsize=7, color='black')

for i, feat in enumerate(lmm_df['Feature']):
    for j, e in enumerate(effects):
        if sig_matrix.loc[feat, e]:
            ax.text(j+0.5, i+0.5, '*', ha='center', va='center',
                    fontsize=14, color='black', fontweight='bold')
ax.set_title('Layer 3 - LMM All Effects\nColor = −log₁₀(p_FDR) × sign(coef)  |  * = FDR < 0.05',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/layer3/layer3_lmm_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Layer 3 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Collagen rescue (within TAA only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("LAYER 4 -- Collagen rescue (within TAA only)")
print("="*60)

df_taa = df[df['SMC_Group'] == 'Aneurysmal'].copy()
print(f"TAA cells: {len(df_taa)} ({df_taa['SpecimenID'].nunique()} specimens)")

l4_results = []
for feat in feat_cols:
    d_nocol = df_taa[df_taa['Collagen_Status']=='NoCollagen'][feat].dropna()
    d_col   = df_taa[df_taa['Collagen_Status']=='Collagen'][feat].dropna()
    if len(d_nocol) < 3 or len(d_col) < 3:
        continue
    p, test_used = adaptive_test(d_nocol, d_col)
    cohens_d = (d_col.mean() - d_nocol.mean()) / np.sqrt((d_nocol.std()**2 + d_col.std()**2) / 2)
    mean_na = df[df['SMC_Group']=='Non-Aneurysmal'][feat].mean()
    rescue  = abs(d_col.mean() - mean_na) < abs(d_nocol.mean() - mean_na)
    l4_results.append({
        'Feature': feat,
        'Mean_TAA_NoCollagen': d_nocol.mean(),
        'Mean_TAA_Collagen':   d_col.mean(),
        'Mean_NonAneurysmal':  mean_na,
        'Cohens_d': cohens_d, 'Test_used': test_used,
        'p_value': p, 'Moves_toward_healthy': rescue
    })

l4_df = pd.DataFrame(l4_results)
_, fdr, _, _ = multipletests(l4_df['p_value'], method='fdr_bh')
l4_df['p_FDR'] = fdr
l4_df['Significant'] = fdr < ALPHA
l4_df = l4_df.sort_values('p_value')
l4_df.to_csv(f'{OUT_DIR}/layer4/layer4_collagen_rescue.csv', index=False)

l4_sig    = l4_df[l4_df['Significant']]['Feature'].tolist()
l4_rescue = l4_df[l4_df['Significant'] & l4_df['Moves_toward_healthy']]['Feature'].tolist()
print(f"Significant collagen effect: {len(l4_sig)} | Toward healthy: {len(l4_rescue)}")
print([shorten(f) for f in l4_rescue])

fig, ax = plt.subplots(figsize=(10, 7))
plot_df = l4_df.sort_values('Cohens_d').reset_index(drop=True)

# Color logic: direction is primary (teal/red), saturation encodes nominal significance (p<0.05 raw)
# → 4 states: toward-healthy sig, toward-healthy ns, away-from-healthy sig, away-from-healthy ns
# FDR-significant features (if any) get a black edge highlight
def bar_color(row):
    toward  = row['Moves_toward_healthy']
    nom_sig = row['p_value'] < 0.05          # nominal, not FDR
    if toward and nom_sig:    return '#00796B'  # dark teal
    if toward and not nom_sig: return '#80CBC4' # light teal
    if not toward and nom_sig: return '#BF360C' # dark red
    return '#FFAB91'                            # light salmon

colors_l4    = [bar_color(r) for _, r in plot_df.iterrows()]
edge_colors  = ['black' if r['Significant'] else 'white' for _, r in plot_df.iterrows()]
edge_widths  = [1.5    if r['Significant'] else 0.5      for _, r in plot_df.iterrows()]

bars = ax.barh([shorten(f) for f in plot_df['Feature']], plot_df['Cohens_d'],
               color=colors_l4, edgecolor=edge_colors, linewidth=edge_widths, height=0.7)

# Add a star for FDR-significant features (if any)
for i, (_, row) in enumerate(plot_df.iterrows()):
    if row['Significant']:
        x = row['Cohens_d']
        offset = 0.02 if x >= 0 else -0.02
        ha = 'left' if x >= 0 else 'right'
        ax.text(x + offset, i, '*', ha=ha, va='center', fontsize=13,
                color='black', fontweight='bold')

ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel("Cohen's d  (TAA +Col − TAA NoCol)", fontsize=10)
ax.set_title('Collagen effect within TAA\n'
             'Direction relative to healthy mean  |  * = FDR < 0.05',
             fontsize=12, fontweight='bold')

legend_handles = [
    mpatches.Patch(color='#00796B', label='Toward healthy  (p < 0.05 raw)'),
    mpatches.Patch(color='#80CBC4', label='Toward healthy  (ns)'),
    mpatches.Patch(color='#BF360C', label='Away from healthy  (p < 0.05 raw)'),
    mpatches.Patch(color='#FFAB91', label='Away from healthy  (ns)'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9,
          framealpha=0.9, edgecolor='#cccccc')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/layer4/layer4_rescue_direction.png', dpi=300, bbox_inches='tight')
plt.close()

if len(l4_sig) > 0:
    top_feats = l4_sig[:min(6, len(l4_sig))]
    n = len(top_feats)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    df_plot = df[df['Group_Label'].isin([
        'Non-Aneurysmal_NoCollagen','Aneurysmal_NoCollagen','Aneurysmal_Collagen'])].copy()
    palette_r = {'Non-Aneurysmal_NoCollagen':'#00796B',
                 'Aneurysmal_NoCollagen':'#FFCCBC','Aneurysmal_Collagen':'#E64A19'}
    order_r  = ['Non-Aneurysmal_NoCollagen','Aneurysmal_NoCollagen','Aneurysmal_Collagen']
    labels_r = ['Non-A\n(reference)','TAA\n-Col','TAA\n+Col']
    for i, feat in enumerate(top_feats):
        ax = axes[i]
        sns.violinplot(data=df_plot, x='Group_Label', y=feat,
                       order=order_r, palette=palette_r, ax=ax, inner='quartile')
        sns.swarmplot(data=df_plot, x='Group_Label', y=feat,
                      order=order_r, color='black', size=2.5, alpha=0.4, ax=ax)
        ax.set_xticklabels(labels_r, fontsize=9)
        ax.set_xlabel('')
        ax.set_title(shorten(feat), fontsize=10, fontweight='bold')
        row = l4_df[l4_df['Feature']==feat].iloc[0]
        direction = 'toward healthy' if row['Moves_toward_healthy'] else 'away from healthy'
        ax.annotate(f"FDR={row['p_FDR']:.3f}, d={row['Cohens_d']:.2f}\n{direction}",
                    xy=(0.5,0.97), xycoords='axes fraction', ha='center', va='top', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Layer 4 -- Collagen Rescue Violins', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/layer4/layer4_rescue_violins.png', dpi=300, bbox_inches='tight')
    plt.close()
print("Layer 4 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — Sex effect
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("LAYER 5 -- Sex effect (Male vs Female)")
print("="*60)

print("\nSex distribution:")
print(df.groupby(['SMC_Group','Gender']).size().unstack(fill_value=0).to_string())
print(f"Male specimens: {df[df['Gender']=='Male']['SpecimenID'].nunique()} | "
      f"Female specimens: {df[df['Gender']=='Female']['SpecimenID'].nunique()}")

# ── 5a: Overall sex effect ─────────────────────────────────────────────────────
sex_results = []
for feat in feat_cols:
    male   = df[df['Gender']=='Male'][feat].dropna()
    female = df[df['Gender']=='Female'][feat].dropna()
    if len(male) < 3 or len(female) < 3:
        continue
    p, test_used = adaptive_test(male, female)
    cohens_d = (male.mean() - female.mean()) / np.sqrt((male.std()**2 + female.std()**2) / 2)
    sex_results.append({
        'Feature':     feat,
        'Mean_Male':   male.mean(), 'Mean_Female': female.mean(),
        'SD_Male':     male.std(),  'SD_Female':   female.std(),
        'N_Male':      len(male),   'N_Female':    len(female),
        'Cohens_d':    cohens_d,    'Test_used':   test_used,
        'p_value':     p
    })

sex_df = pd.DataFrame(sex_results)
_, fdr_sex, _, _ = multipletests(sex_df['p_value'], method='fdr_bh')
sex_df['p_FDR']       = fdr_sex
sex_df['Significant'] = fdr_sex < ALPHA
sex_df = sex_df.sort_values('p_value')
sex_df.to_csv(f'{OUT_DIR}/layer5_sex/sex_effect_overall.csv', index=False)

sex_sig = sex_df[sex_df['Significant']]['Feature'].tolist()
print(f"\nOverall sex effect (FDR < 0.05): {len(sex_sig)}")
print([shorten(f) for f in sex_sig])

# ── 5b: Sex effect within each disease group ──────────────────────────────────
print("\nSex effect within each disease group:")
sex_stratified = []
for group in ['Non-Aneurysmal', 'Aneurysmal']:
    df_grp = df[df['SMC_Group'] == group]
    for feat in feat_cols:
        male   = df_grp[df_grp['Gender']=='Male'][feat].dropna()
        female = df_grp[df_grp['Gender']=='Female'][feat].dropna()
        if len(male) < 3 or len(female) < 3:
            continue
        p, test_used = adaptive_test(male, female)
        cohens_d = (male.mean() - female.mean()) / np.sqrt((male.std()**2 + female.std()**2) / 2)
        sex_stratified.append({
            'Group': group, 'Feature': feat,
            'Mean_Male': male.mean(), 'Mean_Female': female.mean(),
            'Cohens_d': cohens_d, 'Test_used': test_used,
            'p_value': p, 'N_Male': len(male), 'N_Female': len(female)
        })

sex_strat_df = pd.DataFrame(sex_stratified)
for group in ['Non-Aneurysmal', 'Aneurysmal']:
    idx = sex_strat_df['Group'] == group
    if idx.sum() > 0:
        _, fdr_g, _, _ = multipletests(sex_strat_df.loc[idx, 'p_value'], method='fdr_bh')
        sex_strat_df.loc[idx, 'p_FDR']       = fdr_g
        sex_strat_df.loc[idx, 'Significant']  = fdr_g < ALPHA

sex_strat_df.to_csv(f'{OUT_DIR}/layer5_sex/sex_effect_by_group.csv', index=False)
for group in ['Non-Aneurysmal', 'Aneurysmal']:
    sig = sex_strat_df[(sex_strat_df['Group']==group) & sex_strat_df['Significant']]['Feature'].tolist()
    print(f"  {group}: {len(sig)} -- {[shorten(f) for f in sig]}")

# ── Plot 5a: Sex effect bar chart ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
plot_df = sex_df.sort_values('Cohens_d')
ax.barh([shorten(f) for f in plot_df['Feature']], plot_df['Cohens_d'],
        color=['#8E44AD' if s else '#AAAAAA' for s in plot_df['Significant']],
        edgecolor='white', height=0.7)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel("Cohen's d (Male - Female)\nPositive = higher in males", fontsize=11)
ax.set_title("Layer 5 -- Sex Effect on Cell Morphology\n(all cells, FDR-corrected)",
             fontsize=12, fontweight='bold')
ax.legend(handles=[mpatches.Patch(color='#8E44AD', label='FDR < 0.05'),
                   mpatches.Patch(color='#AAAAAA', label='FDR >= 0.05')])
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/layer5_sex/sex_effect_overall.png', dpi=300, bbox_inches='tight')
plt.close()

# ── Plot 5b: Sex effect heatmap by disease group ───────────────────────────────
groups_sex = ['Non-Aneurysmal', 'Aneurysmal']
d_matrix_sex   = pd.DataFrame(index=feat_cols, columns=groups_sex, dtype=float)
sig_matrix_sex = pd.DataFrame(index=feat_cols, columns=groups_sex, dtype=bool)
for group in groups_sex:
    for feat in feat_cols:
        row = sex_strat_df[(sex_strat_df['Group']==group) & (sex_strat_df['Feature']==feat)]
        if len(row):
            d_matrix_sex.loc[feat, group]   = row['Cohens_d'].values[0]
            sig_matrix_sex.loc[feat, group] = row['Significant'].values[0]

fig, ax = plt.subplots(figsize=(7, max(6, len(feat_cols)*0.45)))
sns.heatmap(d_matrix_sex.astype(float), ax=ax, cmap='PuOr', center=0,
            xticklabels=groups_sex, yticklabels=[shorten(f) for f in feat_cols],
            linewidths=0.4, linecolor='#cccccc',
            cbar_kws={"label": "Cohen's d (Male - Female)"})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=11)
for i, feat in enumerate(feat_cols):
    for j, group in enumerate(groups_sex):
        if sig_matrix_sex.loc[feat, group]:
            ax.text(j+0.5, i+0.5, '*', ha='center', va='center',
                    fontsize=14, color='black', fontweight='bold')
ax.set_title("Sex Effect by Disease Group\n* = FDR < 0.05", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/layer5_sex/sex_effect_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ── Plot 5c: Boxplots for significant sex features ─────────────────────────────
if len(sex_sig) > 0:
    top_sex = sex_sig[:min(6, len(sex_sig))]
    n = len(top_sex)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i, feat in enumerate(top_sex):
        ax = axes[i]
        sub_plot = df[df['Gender'].isin(['Male','Female'])][[feat,'Gender','SMC_Group']].dropna()
        sns.boxplot(data=sub_plot, x='Gender', y=feat, hue='SMC_Group',
                    order=['Male','Female'],
                    palette={'Non-Aneurysmal':'#B2DFDB','Aneurysmal':'#FFCCBC'},
                    ax=ax, width=0.5)
        sns.swarmplot(data=sub_plot, x='Gender', y=feat, hue='SMC_Group',
                      order=['Male','Female'],
                      palette={'Non-Aneurysmal':'#00796B','Aneurysmal':'#E64A19'},
                      ax=ax, size=3, alpha=0.6, dodge=True, legend=False)
        row = sex_df[sex_df['Feature']==feat].iloc[0]
        ax.set_title(shorten(feat), fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.annotate(f"FDR={row['p_FDR']:.3f}, d={row['Cohens_d']:.2f}",
                    xy=(0.5,0.97), xycoords='axes fraction', ha='center', va='top',
                    fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.legend(title='Group', fontsize=7, title_fontsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Layer 5 -- Significant Sex Effects (FDR < 0.05)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/layer5_sex/sex_significant_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("No significant sex effects to plot.")

print("Layer 5 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY -- Cross-layer feature robustness")
print("="*60)

summary = pd.DataFrame({'Feature': feat_cols})
summary['Layer1_sig']         = summary['Feature'].isin(l1_sig)
summary['Layer2_sig']         = summary['Feature'].isin(l2_sig)
summary['Layer3_Disease_sig'] = summary['Feature'].isin(lmm_df[lmm_df['Disease_sig']==True]['Feature'].tolist())
summary['Layer4_Rescue_sig']  = summary['Feature'].isin(l4_sig)
summary['Layer4_Rescue_dir']  = summary['Feature'].isin(l4_rescue)
summary['Layer5_Sex_sig']     = summary['Feature'].isin(sex_sig)
summary['N_layers_sig']       = (summary['Layer1_sig'].astype(int) +
                                  summary['Layer2_sig'].astype(int) +
                                  summary['Layer3_Disease_sig'].astype(int))
summary['Label'] = summary['Feature'].apply(shorten)
summary = summary.sort_values('N_layers_sig', ascending=False)
summary.to_csv(f'{OUT_DIR}/summary/cross_layer_summary.csv', index=False)

fig, ax = plt.subplots(figsize=(9, max(6, len(feat_cols)*0.4)))
heatmap_data = summary.set_index('Label')[[
    'Layer1_sig','Layer2_sig','Layer3_Disease_sig','Layer4_Rescue_sig','Layer5_Sex_sig'
]].astype(int)
heatmap_data.columns = ['L1: Raw','L2: Age-corr.','L3: LMM','L4: Rescue','L5: Sex']
sns.heatmap(heatmap_data, ax=ax, cmap='YlOrRd', vmin=0, vmax=1,
            linewidths=0.5, linecolor='#cccccc',
            cbar_kws={'label': 'Significant (FDR<0.05)'}, annot=False)
ax.set_title('Cross-layer Robustness\n(darker = significant)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/summary/cross_layer_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

counts = {
    'L1: Raw\n(disease)':        len(l1_sig),
    'L2: Age-corr.\n(disease)':  len(l2_sig),
    'L3: LMM\n(disease)':        int(lmm_df['Disease_sig'].sum()),
    'L3: LMM\n(collagen)':       int(lmm_df['Collagen_sig'].sum()),
    'L3: LMM\n(interaction)':    int(lmm_df['Interaction_sig'].sum()),
    'L3: LMM\n(age)':            int(lmm_df['Age_sig'].sum()),
    'L3: LMM\n(sex)':            int(lmm_df['Gender_sig'].sum()),
    'L4: Rescue\n(TAA)':         len(l4_sig),
    'L4: Rescue\n->healthy':     len(l4_rescue),
    'L5: Sex\n(overall)':        len(sex_sig),
}
bar_colors = ['#5C6BC0','#3949AB','#E64A19','#EF6C00','#FF8F00',
              '#2E7D32','#880E4F','#00796B','#004D40','#8E44AD']

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(counts))
bars = ax.bar(x, list(counts.values()), color=bar_colors, edgecolor='white', width=0.65)
ax.set_xticks(x)
ax.set_xticklabels(list(counts.keys()), fontsize=8.5)
ax.set_ylabel('Significant features (FDR < 0.05)', fontsize=10)
ax.set_title('5-Layer Analysis Summary', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(counts.values()) + 2)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
for bar, val in zip(bars, counts.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/summary/5layer_summary_bar.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Layer 1 -- Raw disease effect:             {len(l1_sig)}/{len(feat_cols)}")
print(f"Layer 2 -- Age-corrected disease effect:   {len(l2_sig)}/{len(feat_cols)}")
print(f"Layer 3 -- LMM disease:                    {int(lmm_df['Disease_sig'].sum())}/{len(feat_cols)}")
print(f"Layer 3 -- LMM collagen:                   {int(lmm_df['Collagen_sig'].sum())}/{len(feat_cols)}")
print(f"Layer 3 -- LMM interaction:                {int(lmm_df['Interaction_sig'].sum())}/{len(feat_cols)}")
print(f"Layer 3 -- LMM age:                        {int(lmm_df['Age_sig'].sum())}/{len(feat_cols)}")
print(f"Layer 3 -- LMM sex:                        {int(lmm_df['Gender_sig'].sum())}/{len(feat_cols)}")
print(f"Layer 4 -- Collagen rescue in TAA:         {len(l4_sig)}/{len(feat_cols)}")
print(f"Layer 4 -- Rescue toward healthy:          {len(l4_rescue)}/{len(feat_cols)}")
print(f"Layer 5 -- Sex effect overall:             {len(sex_sig)}/{len(feat_cols)}")
print(f"\nOutputs: {OUT_DIR}/")
print("  layer1/     -- Raw adaptive test")
print("  layer2/     -- Age-corrected + age slopes")
print("  layer3/     -- LMM forest + heatmap")
print("  layer4/     -- Collagen rescue + violins")
print("  layer5_sex/ -- Sex bar + heatmap + boxplots")
print("  summary/    -- Cross-layer heatmap + summary bar")