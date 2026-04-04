"""
SMC Statistical Analysis
Groups: Non-Aneurysmal vs Aneurysmal × +Collagen / -Collagen
Tests:  1. Disease effect (Non-Aneurysmal vs Aneurysmal)
        2. Collagen effect (+Col vs -Col)
        3. Age correlation
        4. Sex effect
Method: Linear Mixed Model (subject as random effect to account for
        multiple cells per patient — avoids pseudoreplication)
        + Mann-Whitney U for pairwise with FDR correction

Outputs are organised into subfolders:
  classification_results/smc_analysis/
    lmm/         — LMM results CSV + heatmap + forest plot
    pairwise/    — Mann-Whitney results CSV + effect-size heatmap
    age_sex/     — Age correlation + sex effect CSVs + plots
    plots/       — Violin plots + summary bar chart
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
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_FILE = 'outputs/Advanced_Features_Raw_200.csv'
BASE_DIR      = 'classification_results/smc_analysis'
LMM_DIR       = f'{BASE_DIR}/lmm'
PW_DIR        = f'{BASE_DIR}/pairwise'
AGE_SEX_DIR   = f'{BASE_DIR}/age_sex'
PLOTS_DIR     = f'{BASE_DIR}/plots'
ALPHA         = 0.05

for d in [LMM_DIR, PW_DIR, AGE_SEX_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

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

GROUP_ORDER = ['Non-Aneurysmal_NoCollagen','Non-Aneurysmal_Collagen',
               'Aneurysmal_NoCollagen','Aneurysmal_Collagen']
GROUP_LABELS = ['Non-A\n−Col','Non-A\n+Col','TAA\n−Col','TAA\n+Col']
PALETTE = {
    'Non-Aneurysmal_NoCollagen': '#B2DFDB',
    'Non-Aneurysmal_Collagen':   '#00796B',
    'Aneurysmal_NoCollagen':     '#FFCCBC',
    'Aneurysmal_Collagen':       '#E64A19',
}

def shorten(feat):
    return (feat.replace('_µm³','').replace('_µm²','').replace('_µm','')
               .replace('_ratio','').replace('_n','').replace('_',' '))

# ── Load & filter ─────────────────────────────────────────────────────────────
def extract_id(cell_name):
    m = re.search(r'(0[123][A-Za-z]+-\d+)', str(cell_name))
    if not m:
        m = re.search(r'(0[123][A-Za-z]+)(\d+)', str(cell_name))
        if not m: return None
        return (m.group(1) + '-' + m.group(2)).lower()
    return m.group(1).lower()

def normalize(sid):
    return re.sub(r'-0*(\d+)$', r'-\1', sid.lower()) if sid else None

df_all = pd.read_csv(FEATURES_FILE)
df_all['SpecimenID'] = df_all['CellName'].apply(extract_id).apply(lambda x: normalize(x))

na_norm = [normalize(x) for x in NON_ANEURYSMAL]
an_norm = [normalize(x) for x in ANEURYSMAL]
all_smc = na_norm + an_norm

df = df_all[df_all['SpecimenID'].isin(all_smc)].copy()
df['SMC_Group']   = df['SpecimenID'].apply(lambda x: 'Non-Aneurysmal' if x in na_norm else 'Aneurysmal')
df['Group_Label'] = df['SMC_Group'] + '_' + df['Collagen_Status']
df['Subject']     = df['SpecimenID']
feat_cols = [c for c in FEATURE_COLS if c in df.columns]

print(f"SMC dataset: {len(df)} cells, {df['SpecimenID'].nunique()} specimens")
print(df.groupby(['SMC_Group','Collagen_Status']).size().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 1. LINEAR MIXED MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- LINEAR MIXED MODELS ---")
lmm_results = []

for feat in feat_cols:
    sub = df[['Subject','SMC_Group','Collagen_Status','Age','Gender',feat]].dropna()
    if len(sub) < 20:
        continue
    sub = sub.copy()
    sub['Disease_bin']  = (sub['SMC_Group'] == 'Aneurysmal').astype(int)
    sub['Collagen_bin'] = (sub['Collagen_Status'] == 'Collagen').astype(int)
    sub['Gender_bin']   = (sub['Gender'] == 'Male').astype(int)
    sub['feat']         = sub[feat]
    try:
        md = smf.mixedlm(
            "feat ~ Disease_bin + Collagen_bin + Disease_bin:Collagen_bin + Age + Gender_bin",
            sub, groups=sub["Subject"])
        res = md.fit(reml=True)
        lmm_results.append({
            'Feature':            feat,
            'Disease_coef':       res.params.get('Disease_bin', np.nan),
            'Disease_pval':       res.pvalues.get('Disease_bin', np.nan),
            'Collagen_coef':      res.params.get('Collagen_bin', np.nan),
            'Collagen_pval':      res.pvalues.get('Collagen_bin', np.nan),
            'Interaction_coef':   res.params.get('Disease_bin:Collagen_bin', np.nan),
            'Interaction_pval':   res.pvalues.get('Disease_bin:Collagen_bin', np.nan),
            'Age_coef':           res.params.get('Age', np.nan),
            'Age_pval':           res.pvalues.get('Age', np.nan),
            'Gender_coef':        res.params.get('Gender_bin', np.nan),
            'Gender_pval':        res.pvalues.get('Gender_bin', np.nan),
            'Converged':          res.converged,
            'N_cells':            len(sub),
            'N_subjects':         sub['Subject'].nunique(),
        })
    except Exception as e:
        print(f"  LMM failed for {feat}: {e}")

lmm_df = pd.DataFrame(lmm_results)

for effect in ['Disease','Collagen','Interaction','Age','Gender']:
    pvals = lmm_df[f'{effect}_pval'].fillna(1)
    _, fdr, _, _ = multipletests(pvals, method='fdr_bh')
    lmm_df[f'{effect}_FDR'] = fdr
    lmm_df[f'{effect}_sig'] = fdr < ALPHA

lmm_df.to_csv(f'{LMM_DIR}/lmm_results.csv', index=False)

print("\nSignificant features (FDR < 0.05):")
for effect in ['Disease','Collagen','Interaction','Age','Gender']:
    sig = lmm_df[lmm_df[f'{effect}_sig']==True]['Feature'].tolist()
    print(f"  {effect:12s}: {len(sig)} — {[shorten(f) for f in sig]}")

# ── Plot 1: LMM coefficient heatmap ──────────────────────────────────────────
effects = ['Disease','Collagen','Interaction','Age','Gender']
coef_matrix = pd.DataFrame(index=lmm_df['Feature'], columns=effects)
sig_matrix  = pd.DataFrame(index=lmm_df['Feature'], columns=effects)
for e in effects:
    coef_matrix[e] = lmm_df.set_index('Feature')[f'{e}_coef'].values
    sig_matrix[e]  = lmm_df.set_index('Feature')[f'{e}_sig'].values

coef_norm = coef_matrix.apply(lambda col: col / col.abs().max() if col.abs().max() > 0 else col)
coef_norm = coef_norm.astype(float)
feat_labels = [shorten(f) for f in lmm_df['Feature'].tolist()]

fig, ax = plt.subplots(figsize=(8, max(6, len(feat_labels)*0.45)))
sns.heatmap(coef_norm, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=effects, yticklabels=feat_labels,
            linewidths=0.4, linecolor='#cccccc',
            cbar_kws={'label': 'Normalised coefficient'})

# Annotate significant cells with *
for i, feat in enumerate(lmm_df['Feature']):
    for j, e in enumerate(effects):
        if lmm_df.loc[lmm_df['Feature']==feat, f'{e}_sig'].values[0]:
            ax.text(j+0.5, i+0.5, '*', ha='center', va='center',
                    fontsize=14, color='black', fontweight='bold')

ax.set_title('LMM Coefficients (normalised)\n* = FDR < 0.05', fontsize=12, fontweight='bold')
ax.set_xlabel('Effect', fontsize=11)
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(f'{LMM_DIR}/lmm_coefficient_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ── Plot 2: LMM forest plot — Disease effect ──────────────────────────────────
forest_df = lmm_df[['Feature','Disease_coef','Disease_FDR','Disease_sig']].copy()
forest_df = forest_df.sort_values('Disease_coef')
forest_df['label'] = [shorten(f) for f in forest_df['Feature']]
colors_fp = ['#E64A19' if s else '#AAAAAA' for s in forest_df['Disease_sig']]

fig, ax = plt.subplots(figsize=(8, max(5, len(forest_df)*0.4)))
y_pos = np.arange(len(forest_df))
ax.barh(y_pos, forest_df['Disease_coef'], color=colors_fp, edgecolor='white', height=0.7)
ax.axvline(0, color='black', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(forest_df['label'], fontsize=9)
ax.set_xlabel('LMM coefficient (Aneurysmal − Non-Aneurysmal)', fontsize=10)
ax.set_title('Disease Effect on SMC Morphology\n(LMM, controlling for Collagen, Age, Sex)',
             fontsize=12, fontweight='bold')

sig_patch   = mpatches.Patch(color='#E64A19', label='FDR < 0.05')
insig_patch = mpatches.Patch(color='#AAAAAA', label='FDR ≥ 0.05')
ax.legend(handles=[sig_patch, insig_patch], loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{LMM_DIR}/lmm_forest_disease_effect.png', dpi=300, bbox_inches='tight')
plt.close()
print("LMM plots saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PAIRWISE MANN-WHITNEY
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- PAIRWISE MANN-WHITNEY ---")
comparisons = [
    ('Non-Aneurysmal_NoCollagen', 'Aneurysmal_NoCollagen',   'Disease (−Col)'),
    ('Non-Aneurysmal_Collagen',   'Aneurysmal_Collagen',     'Disease (+Col)'),
    ('Non-Aneurysmal_NoCollagen', 'Non-Aneurysmal_Collagen', 'Collagen (Non-A)'),
    ('Aneurysmal_NoCollagen',     'Aneurysmal_Collagen',     'Collagen (TAA)'),
]

pw_results = []
for feat in feat_cols:
    for g1, g2, label in comparisons:
        d1 = df[df['Group_Label']==g1][feat].dropna()
        d2 = df[df['Group_Label']==g2][feat].dropna()
        if len(d1) < 3 or len(d2) < 3:
            continue
        u, p = mannwhitneyu(d1, d2, alternative='two-sided')
        d = (d2.mean() - d1.mean()) / np.sqrt((d1.std()**2 + d2.std()**2) / 2)
        pw_results.append({'Feature':feat,'Comparison':label,
                           'Group1':g1,'Group2':g2,
                           'Mean1':d1.mean(),'Mean2':d2.mean(),
                           'Cohens_d':d,'U_stat':u,'p_value':p})

pw_df = pd.DataFrame(pw_results)
_, fdr, _, _ = multipletests(pw_df['p_value'], method='fdr_bh')
pw_df['p_FDR']       = fdr
pw_df['Significant'] = fdr < ALPHA
pw_df.to_csv(f'{PW_DIR}/pairwise_results.csv', index=False)

print("Significant pairwise comparisons (FDR < 0.05):")
for label in [c[2] for c in comparisons]:
    sub = pw_df[(pw_df['Comparison']==label) & (pw_df['Significant'])]
    print(f"  {label:30s}: {len(sub)} features")

# ── Plot 3: Effect-size heatmap (Cohen's d) ───────────────────────────────────
comp_labels = [c[2] for c in comparisons]
d_matrix = pd.DataFrame(index=feat_cols, columns=comp_labels, dtype=float)
sig_pw    = pd.DataFrame(index=feat_cols, columns=comp_labels, dtype=bool)

for feat in feat_cols:
    for label in comp_labels:
        row = pw_df[(pw_df['Feature']==feat) & (pw_df['Comparison']==label)]
        if len(row):
            d_matrix.loc[feat, label]  = row['Cohens_d'].values[0]
            sig_pw.loc[feat, label]    = row['Significant'].values[0]

d_matrix = d_matrix.astype(float)
feat_labels_pw = [shorten(f) for f in feat_cols]

fig, ax = plt.subplots(figsize=(8, max(6, len(feat_labels_pw)*0.45)))
sns.heatmap(d_matrix, ax=ax, cmap='RdBu_r', center=0,
            xticklabels=comp_labels, yticklabels=feat_labels_pw,
            linewidths=0.4, linecolor='#cccccc',
            cbar_kws={'label': "Cohen's d"})

for i, feat in enumerate(feat_cols):
    for j, label in enumerate(comp_labels):
        if sig_pw.loc[feat, label]:
            ax.text(j+0.5, i+0.5, '*', ha='center', va='center',
                    fontsize=14, color='black', fontweight='bold')

ax.set_title("Pairwise Effect Sizes (Cohen's d)\n* = FDR < 0.05", fontsize=12, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(f'{PW_DIR}/pairwise_effect_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Pairwise plots saved.")

# ── Plot 4: Violin plots — top disease-significant features ───────────────────
sig_disease  = lmm_df[lmm_df['Disease_sig']==True].sort_values('Disease_FDR')['Feature'].tolist()
sig_collagen = lmm_df[lmm_df['Collagen_sig']==True].sort_values('Collagen_FDR')['Feature'].tolist()
top_features = list(dict.fromkeys(sig_disease + sig_collagen))[:8]

if not top_features:
    # Fall back to top pairwise
    top_features = (pw_df[pw_df['Significant']]
                    .groupby('Feature').size()
                    .sort_values(ascending=False)
                    .head(8).index.tolist())

if top_features:
    n = len(top_features)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, feat in enumerate(top_features):
        ax = axes[i]
        sns.violinplot(data=df, x='Group_Label', y=feat,
                       order=GROUP_ORDER, palette=PALETTE, ax=ax, inner='quartile')
        sns.swarmplot(data=df, x='Group_Label', y=feat,
                      order=GROUP_ORDER, color='black', size=2.5, alpha=0.4, ax=ax)
        ax.set_xticklabels(GROUP_LABELS, fontsize=8)
        ax.set_xlabel('')
        title = shorten(feat)
        ax.set_title(title, fontsize=10, fontweight='bold')

        lmm_row = lmm_df[lmm_df['Feature']==feat]
        parts = []
        if len(lmm_row) and lmm_row.iloc[0]['Disease_sig']:
            parts.append(f"LMM Disease FDR={lmm_row.iloc[0]['Disease_FDR']:.3f}")
        if len(lmm_row) and lmm_row.iloc[0]['Collagen_sig']:
            parts.append(f"LMM Coll FDR={lmm_row.iloc[0]['Collagen_FDR']:.3f}")
        if parts:
            ax.annotate('\n'.join(parts), xy=(0.5,0.97), xycoords='axes fraction',
                        ha='center', va='top', fontsize=7,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('SMC Morphology: Non-Aneurysmal vs Aneurysmal × Collagen',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/violin_top_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Violin plots saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. AGE & SEX CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- AGE & SEX CORRELATIONS ---")

# Age Spearman
age_results = []
df_age = df.dropna(subset=['Age'])
for feat in feat_cols:
    sub = df_age[[feat,'Age']].dropna()
    if len(sub) < 10: continue
    r, p = spearmanr(sub[feat], sub['Age'])
    age_results.append({'Feature':feat,'Spearman_r':r,'p_value':p})

age_df = pd.DataFrame(age_results)
_, fdr, _, _ = multipletests(age_df['p_value'], method='fdr_bh')
age_df['p_FDR']     = fdr
age_df['Significant'] = fdr < ALPHA
age_df = age_df.sort_values('p_value')
age_df.to_csv(f'{AGE_SEX_DIR}/age_correlation.csv', index=False)
print(f"Features correlated with Age (FDR<0.05): {age_df['Significant'].sum()}")
print(age_df[age_df['Significant']][['Feature','Spearman_r','p_FDR']].to_string(index=False))

# Sex Mann-Whitney
sex_results = []
for feat in feat_cols:
    male   = df[df['Gender']=='Male'][feat].dropna()
    female = df[df['Gender']=='Female'][feat].dropna()
    if len(male) < 3 or len(female) < 3: continue
    u, p = mannwhitneyu(male, female, alternative='two-sided')
    d = (male.mean()-female.mean()) / np.sqrt((male.std()**2+female.std()**2)/2)
    sex_results.append({'Feature':feat,'Male_mean':male.mean(),'Female_mean':female.mean(),
                        'Cohens_d':d,'p_value':p})
sex_df = pd.DataFrame(sex_results)
_, fdr, _, _ = multipletests(sex_df['p_value'], method='fdr_bh')
sex_df['p_FDR']     = fdr
sex_df['Significant'] = fdr < ALPHA
sex_df.to_csv(f'{AGE_SEX_DIR}/sex_effect.csv', index=False)
print(f"Features with Sex effect (FDR<0.05): {sex_df['Significant'].sum()}")

# ── Plot 5: Age scatter plots for top correlated features ─────────────────────
top_age_feats = age_df[age_df['Significant']]['Feature'].tolist()[:6]
if not top_age_feats:
    top_age_feats = age_df.head(4)['Feature'].tolist()

if top_age_feats:
    n = len(top_age_feats)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    group_colors = {'Non-Aneurysmal': '#00796B', 'Aneurysmal': '#E64A19'}

    for i, feat in enumerate(top_age_feats):
        ax = axes[i]
        for grp, color in group_colors.items():
            sub = df_age[df_age['SMC_Group']==grp][[feat,'Age']].dropna()
            ax.scatter(sub['Age'], sub[feat], c=color, alpha=0.6, s=20, label=grp)
            if len(sub) > 2:
                m, b = np.polyfit(sub['Age'], sub[feat], 1)
                x_line = np.linspace(sub['Age'].min(), sub['Age'].max(), 50)
                ax.plot(x_line, m*x_line+b, color=color, linewidth=1.5)

        row = age_df[age_df['Feature']==feat].iloc[0]
        ax.set_title(shorten(feat), fontsize=10, fontweight='bold')
        ax.set_xlabel('Age (years)', fontsize=9)
        ax.annotate(f"ρ={row['Spearman_r']:.2f}, FDR={row['p_FDR']:.3f}",
                    xy=(0.05,0.95), xycoords='axes fraction',
                    ha='left', va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Age Correlations (Spearman)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{AGE_SEX_DIR}/age_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

# ── Plot 6: Sex effect boxplots ───────────────────────────────────────────────
top_sex_feats = sex_df[sex_df['Significant']]['Feature'].tolist()[:6]
if not top_sex_feats:
    top_sex_feats = sex_df.sort_values('p_value').head(4)['Feature'].tolist()

if top_sex_feats:
    n = len(top_sex_feats)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    sex_palette = {'Male': '#4FC3F7', 'Female': '#F48FB1'}

    for i, feat in enumerate(top_sex_feats):
        ax = axes[i]
        sns.boxplot(data=df.dropna(subset=[feat,'Gender']),
                    x='Gender', y=feat,
                    order=['Male','Female'], palette=sex_palette,
                    ax=ax, width=0.5)
        sns.swarmplot(data=df.dropna(subset=[feat,'Gender']),
                      x='Gender', y=feat,
                      order=['Male','Female'], color='black',
                      size=3, alpha=0.5, ax=ax)
        row = sex_df[sex_df['Feature']==feat].iloc[0]
        ax.set_title(shorten(feat), fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.annotate(f"FDR={row['p_FDR']:.3f}\nd={row['Cohens_d']:.2f}",
                    xy=(0.5,0.97), xycoords='axes fraction',
                    ha='center', va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Sex Effect on SMC Morphology', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{AGE_SEX_DIR}/sex_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
print("Age/Sex plots saved.")

# ── Plot 7: Summary bar chart — significant features per test ─────────────────
summary_counts = {
    'LMM Disease':   int(lmm_df['Disease_sig'].sum()),
    'LMM Collagen':  int(lmm_df['Collagen_sig'].sum()),
    'LMM Interact':  int(lmm_df['Interaction_sig'].sum()),
    'LMM Age':       int(lmm_df['Age_sig'].sum()),
    'LMM Sex':       int(lmm_df['Gender_sig'].sum()),
    'MW Disease\n(−Col)':  int(pw_df[(pw_df['Comparison']=='Disease (−Col)') & pw_df['Significant']]['Feature'].nunique()),
    'MW Disease\n(+Col)':  int(pw_df[(pw_df['Comparison']=='Disease (+Col)') & pw_df['Significant']]['Feature'].nunique()),
    'MW Collagen\n(Non-A)':int(pw_df[(pw_df['Comparison']=='Collagen (Non-A)') & pw_df['Significant']]['Feature'].nunique()),
    'MW Collagen\n(TAA)':  int(pw_df[(pw_df['Comparison']=='Collagen (TAA)') & pw_df['Significant']]['Feature'].nunique()),
    'Age corr':      int(age_df['Significant'].sum()),
    'Sex effect':    int(sex_df['Significant'].sum()),
}

bar_colors = (['#5C6BC0']*5 + ['#EF6C00']*4 + ['#2E7D32','#880E4F'])

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(summary_counts))
bars = ax.bar(x, list(summary_counts.values()), color=bar_colors, edgecolor='white', width=0.65)
ax.set_xticks(x)
ax.set_xticklabels(list(summary_counts.keys()), fontsize=9)
ax.set_ylabel('Number of significant features\n(FDR < 0.05)', fontsize=10)
ax.set_title('Summary: Significant Features per Statistical Test\nSMC Analysis — Non-Aneurysmal vs Aneurysmal × ±Collagen',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, max(summary_counts.values()) + 2)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

for bar, val in zip(bars, summary_counts.values()):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

lmm_patch = mpatches.Patch(color='#5C6BC0', label='LMM')
mw_patch  = mpatches.Patch(color='#EF6C00', label='Mann-Whitney')
age_patch = mpatches.Patch(color='#2E7D32', label='Age correlation')
sex_patch = mpatches.Patch(color='#880E4F', label='Sex effect')
ax.legend(handles=[lmm_patch, mw_patch, age_patch, sex_patch],
          loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/summary_significant_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("Summary bar chart saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Total cells analysed:      {len(df)}")
print(f"Total specimens:           {df['SpecimenID'].nunique()}")
print(f"Non-Aneurysmal cells:      {len(df[df['SMC_Group']=='Non-Aneurysmal'])}")
print(f"Aneurysmal cells:          {len(df[df['SMC_Group']=='Aneurysmal'])}")
print(f"\nLMM Disease sig:           {lmm_df['Disease_sig'].sum()} / {len(lmm_df)} features")
print(f"LMM Collagen sig:          {lmm_df['Collagen_sig'].sum()} / {len(lmm_df)} features")
print(f"LMM Interaction sig:       {lmm_df['Interaction_sig'].sum()} / {len(lmm_df)} features")
print(f"LMM Age sig:               {lmm_df['Age_sig'].sum()} / {len(lmm_df)} features")
print(f"LMM Gender sig:            {lmm_df['Gender_sig'].sum()} / {len(lmm_df)} features")
print(f"\nOutputs organised in:")
print(f"  {LMM_DIR}/")
print(f"  {PW_DIR}/")
print(f"  {AGE_SEX_DIR}/")
print(f"  {PLOTS_DIR}/")
