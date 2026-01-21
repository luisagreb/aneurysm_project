"""
Linear Mixed Model Analysis - Accounting for Within-Subject Correlation

Goal: Properly account for the fact that multiple cells come from the same patient.
      This addresses pseudoreplication and provides rigorous statistical inference.

Model Structure:
    Feature ~ Disease * Collagen + (1 | Subject)
    
    - Disease: Fixed effect (Healthy vs TAA)
    - Collagen: Fixed effect (NoCollagen vs +Collagen)  
    - Disease:Collagen: Interaction term
    - (1 | Subject): Random intercept for each patient

This model accounts for:
    1. Within-subject correlation (cells from same patient are not independent)
    2. Main effect of Disease (after controlling for subject)
    3. Main effect of Collagen treatment
    4. Interaction between Disease and Collagen (rescue effect)

Outputs:
    - LMM results for each feature
    - Significance of Disease, Collagen, and Interaction effects
    - Visualization of random effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings
warnings.filterwarnings('ignore')

from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = 'src/analysis/src/analysis/outputs/linear_mixed_model'
ALPHA = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_subject_id(text):
    """
    Extract complete subject identifier with normalized format.
    
    CRITICAL: Normalizes case and removes leading zeros to ensure
    matching between metadata and features file.
    
    Examples:
        '01ASC-0180' -> '01ASC-180'
        '03Asc24' -> '03ASC-24'
        '01C-0096' -> '01C-96'
        '03Rt-45' -> '03RT-45'
    """
    if pd.isna(text):
        return None
    text = str(text).strip()
    first_part = text.split(' ')[0]
    
    # Match patterns like '01ASC-0180' or '01C-96' or '03Asc24'
    # Pattern: prefix (letters with optional leading digits) + optional hyphen + number
    match = re.match(r'^(\d{0,2}[A-Za-z]+)-?0*(\d+)', first_part)
    if match:
        prefix = match.group(1).upper()  # Normalize case (ASC, RT, C)
        number = match.group(2)  # Number without leading zeros
        return f"{prefix}-{number}"
    
    return None

def load_metadata(filepath):
    """Load metadata to classify Healthy vs TAA using normalized Subject IDs."""
    print(f"Loading metadata from {filepath}...")
    df = pd.read_excel(filepath, header=None, skiprows=18)
    
    # Use sets to store unique subject IDs
    healthy_ids = set()
    taa_ids = set()
    
    for x in df[0].dropna():
        sid = extract_subject_id(x)
        if sid:
            healthy_ids.add(sid)
            
    for x in df[1].dropna():
        sid = extract_subject_id(x)
        if sid:
            taa_ids.add(sid)
            
    print(f"  Healthy subjects ({len(healthy_ids)}): {sorted(list(healthy_ids))}")
    print(f"  TAA subjects ({len(taa_ids)}): {sorted(list(taa_ids))}")
    return healthy_ids, taa_ids

def extract_collagen_status(filename):
    """Extract collagen status from filename."""
    if pd.isna(filename):
        return None
    filename = str(filename).lower()
    if '+coll' in filename or '+col' in filename or 'plus coll' in filename:
        return 'Collagen'
    if '-coll' in filename or '-col' in filename or 'nocoll' in filename or 'no coll' in filename or 'dmso' in filename:
        return 'NoCollagen'
    return None

def run_lmm(df, feature, formula="Feature ~ Disease * Collagen"):
    """
    Run Linear Mixed Model for a single feature.
    
    Model: Feature ~ Disease * Collagen + (1|Subject)
    
    Returns dict with coefficients and p-values for each fixed effect.
    """
    # Prepare data
    model_df = df[['Subject', 'Disease', 'Collagen', feature]].dropna()
    model_df = model_df.rename(columns={feature: 'Feature'})
    
    # Need at least 2 subjects per group for random effects
    if model_df['Subject'].nunique() < 4:
        return None
    
    try:
        # Fit mixed model with random intercept for Subject
        model = mixedlm("Feature ~ Disease * Collagen", 
                       model_df, 
                       groups=model_df["Subject"])
        # Use REML=True for unbiased variance estimation (better for small N)
        result = model.fit(reml=True, method='powell', maxiter=200)
        
        # Extract results
        params = result.params
        pvalues = result.pvalues
        
        return {
            'Feature': feature,
            'Intercept': params.get('Intercept', np.nan),
            'Disease_TAA_coef': params.get('Disease[T.TAA]', np.nan),
            'Disease_TAA_pval': pvalues.get('Disease[T.TAA]', np.nan),
            'Collagen_coef': params.get('Collagen[T.NoCollagen]', np.nan),
            'Collagen_pval': pvalues.get('Collagen[T.NoCollagen]', np.nan),
            'Interaction_coef': params.get('Disease[T.TAA]:Collagen[T.NoCollagen]', np.nan),
            'Interaction_pval': pvalues.get('Disease[T.TAA]:Collagen[T.NoCollagen]', np.nan),
            'Random_Effect_Var': result.cov_re.iloc[0, 0] if hasattr(result.cov_re, 'iloc') else np.nan,
            'Residual_Var': result.scale,
            'ICC': result.cov_re.iloc[0, 0] / (result.cov_re.iloc[0, 0] + result.scale) if hasattr(result.cov_re, 'iloc') else np.nan,
            'Log_Likelihood': result.llf,
            'Converged': result.converged
        }
    except Exception as e:
        print(f"  Warning: LMM failed for {feature}: {e}")
        return None

def main():
    print("=" * 80)
    print("LINEAR MIXED MODEL ANALYSIS")
    print("Accounting for within-subject correlation (cells nested in patients)")
    print("=" * 80)
    
    # 1. Load and prepare data
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    df = pd.read_csv(FEATURES_FILE)
    
    df['Subject'] = df['CellName'].apply(extract_subject_id)
    df['Collagen'] = df['CellName'].apply(extract_collagen_status)
    
    def get_disease_label(nid):
        if nid in healthy_ids:
            return 'Healthy'
        if nid in taa_ids:
            return 'TAA'
        return None
    
    df['Disease'] = df['Subject'].apply(get_disease_label)
    df = df.dropna(subset=['Disease', 'Collagen', 'Subject'])
    
    # Convert Subject to string for grouping
    df['Subject'] = df['Subject'].astype(str)
    
    print(f"\nDataset: {len(df)} cells from {df['Subject'].nunique()} subjects")
    print(f"\nCells per condition:")
    print(df.groupby(['Disease', 'Collagen']).size().unstack(fill_value=0))
    
    print(f"\nSubjects per condition:")
    print(df.groupby(['Disease', 'Collagen'])['Subject'].nunique().unstack(fill_value=0))
    
    # 2. HYPOTHESIS-DRIVEN feature selection (reduces multiple testing burden)
    # Only test the top 5 biologically relevant features identified from exploratory analysis
    feature_cols = [
        'Actin_Skeleton_Length_µm',    # Top candidate from exploratory analysis
        'Actin_Solidity_ratio',         # Cell compactness
        'Mito_Sphericity_ratio',        # Mitochondrial shape
        'Mito_Fragment_Count_n',        # Mitochondrial fragmentation
        'Actin_Minor_Axis_µm'           # Cell thickness
    ]
    
    print(f"\nFitting Linear Mixed Models for {len(feature_cols)} features...")
    print("Model: Feature ~ Disease * Collagen + (1|Subject)")
    print("-" * 60)
    
    # 3. Run LMM for each feature
    results = []
    for i, feature in enumerate(feature_cols):
        print(f"  [{i+1}/{len(feature_cols)}] {feature}...", end=" ")
        result = run_lmm(df, feature)
        if result:
            results.append(result)
            print(f"Disease p={result['Disease_TAA_pval']:.4f}")
        else:
            print("Failed")
    
    results_df = pd.DataFrame(results)
    
    # 4. FDR correction for each effect
    for effect in ['Disease_TAA_pval', 'Collagen_pval', 'Interaction_pval']:
        col = effect.replace('_pval', '')
        valid_idx = ~results_df[effect].isna()
        if valid_idx.sum() > 0:
            _, p_adj, _, _ = multipletests(
                results_df.loc[valid_idx, effect], 
                alpha=ALPHA, method='fdr_bh'
            )
            results_df.loc[valid_idx, f'{col}_FDR'] = p_adj
            results_df[f'{col}_sig'] = results_df[f'{col}_FDR'] < ALPHA
    
    # 5. Sort by Disease effect p-value
    results_df = results_df.sort_values('Disease_TAA_pval')
    
    # 6. Save results
    results_df.to_csv(f'{OUTPUT_DIR}/lmm_results.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR}/lmm_results.csv")
    
    # 7. Print summary
    print(f"\n{'='*80}")
    print("SUMMARY: Linear Mixed Model Results")
    print(f"{'='*80}")
    
    n_disease_sig = results_df['Disease_TAA_sig'].sum() if 'Disease_TAA_sig' in results_df else 0
    n_collagen_sig = results_df['Collagen_sig'].sum() if 'Collagen_sig' in results_df else 0
    n_interaction_sig = results_df['Interaction_sig'].sum() if 'Interaction_sig' in results_df else 0
    
    print(f"\nSignificant effects (FDR < 0.05):")
    print(f"  Disease effect (TAA vs Healthy): {n_disease_sig}/{len(results_df)}")
    print(f"  Collagen effect: {n_collagen_sig}/{len(results_df)}")
    print(f"  Disease × Collagen interaction: {n_interaction_sig}/{len(results_df)}")
    
    print(f"\n--- Disease Effect: Top 10 Features ---")
    cols_disease = ['Feature', 'Disease_TAA_coef', 'Disease_TAA_pval', 'Disease_TAA_FDR', 'ICC']
    cols_disease = [c for c in cols_disease if c in results_df.columns]
    print(results_df[cols_disease].head(10).to_string(index=False))
    
    if n_interaction_sig > 0:
        print(f"\n--- Significant Disease × Collagen Interactions ---")
        interaction_sig = results_df[results_df['Interaction_sig'] == True]
        cols_int = ['Feature', 'Interaction_coef', 'Interaction_pval', 'Interaction_FDR']
        cols_int = [c for c in cols_int if c in results_df.columns]
        print(interaction_sig[cols_int].to_string(index=False))
    
    # 8. Visualizations
    
    # 8a. Forest plot of Disease effects (using STANDARDIZED coefficients)
    # Standardize by dividing coefficient by feature SD for comparability
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_df = results_df.copy()
    
    # Calculate standardized coefficients (coefficient / feature SD)
    std_coefs = []
    for _, row in plot_df.iterrows():
        feat = row['Feature']
        if feat in df.columns:
            feat_sd = df[feat].std()
            std_coef = row['Disease_TAA_coef'] / feat_sd if feat_sd > 0 else 0
        else:
            std_coef = 0
        std_coefs.append(std_coef)
    
    plot_df['Std_Coef'] = std_coefs
    plot_df = plot_df.sort_values('Std_Coef')
    
    y_pos = range(len(plot_df))
    colors = ['#E74C3C' if p < 0.05 else '#95A5A6' for p in plot_df['Disease_TAA_FDR']]
    
    ax.barh(y_pos, plot_df['Std_Coef'], color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['Feature'], fontsize=10)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Standardized Coefficient (TAA effect / SD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Linear Mixed Model: Disease Effect (TAA vs Healthy)\nRed = FDR < 0.05 | Standardized for comparability', 
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E74C3C', label='Significant (FDR < 0.05)'),
                      Patch(facecolor='#95A5A6', label='Not significant')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lmm_disease_effect.png', dpi=300)
    print(f"\nSaved: {OUTPUT_DIR}/lmm_disease_effect.png")
    
    # 8b. ICC plot (variance explained by subjects)
    if 'ICC' in results_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        icc_sorted = results_df.sort_values('ICC', ascending=False).head(20)
        
        ax.barh(range(len(icc_sorted)), icc_sorted['ICC'], color='#3498DB', edgecolor='black')
        ax.set_yticks(range(len(icc_sorted)))
        ax.set_yticklabels(icc_sorted['Feature'], fontsize=9)
        ax.set_xlabel('Intraclass Correlation (ICC)', fontsize=12, fontweight='bold')
        ax.set_title('Proportion of Variance Explained by Subject\nHigher ICC = More within-subject correlation', 
                     fontsize=12, fontweight='bold')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1, label='ICC = 0.5')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/lmm_icc_values.png', dpi=300)
        print(f"Saved: {OUTPUT_DIR}/lmm_icc_values.png")
    
    # 8c. Boxplots for significant features (4-group visualization)
    df['Group'] = df['Disease'] + '\n' + df['Collagen'].map({'Collagen': '+Coll', 'NoCollagen': 'No Coll'})
    
    palette = {'Healthy\n+Coll': '#85C1E9', 'Healthy\nNo Coll': '#3498DB', 
               'TAA\n+Coll': '#F1948A', 'TAA\nNo Coll': '#E74C3C'}
    order = ['Healthy\nNo Coll', 'Healthy\n+Coll', 'TAA\nNo Coll', 'TAA\n+Coll']
    
    sig_features = results_df[results_df['Disease_TAA_FDR'] < 0.05]['Feature'].tolist()
    
    if len(sig_features) > 0:
        n_plots = len(sig_features)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        for i, feat in enumerate(sig_features):
            ax = axes[i]
            sns.boxplot(data=df, x='Group', y=feat, ax=ax, order=order, palette=palette)
            
            # Get p-value for title
            fdr_p = results_df[results_df['Feature'] == feat]['Disease_TAA_FDR'].values[0]
            ax.set_title(f'{feat}\nFDR p = {fdr_p:.4f} ***', fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(feat, fontsize=10)
            ax.tick_params(axis='x', labelsize=9)
        
        plt.suptitle('LMM Significant Features: Disease Effect (FDR < 0.05)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/lmm_significant_boxplots.png', dpi=300)
        print(f"Saved: {OUTPUT_DIR}/lmm_significant_boxplots.png")
    
    # 8d. Interaction plot for Actin_Solidity (if significant)
    if 'Interaction_sig' in results_df.columns:
        int_sig = results_df[results_df['Interaction_sig'] == True]['Feature'].tolist()
        if len(int_sig) > 0:
            fig, axes = plt.subplots(1, len(int_sig), figsize=(6*len(int_sig), 5))
            if len(int_sig) == 1:
                axes = [axes]
            
            for i, feat in enumerate(int_sig):
                ax = axes[i]
                
                # Create interaction plot
                for disease in ['Healthy', 'TAA']:
                    subset = df[df['Disease'] == disease]
                    means = subset.groupby('Collagen')[feat].mean()
                    sems = subset.groupby('Collagen')[feat].sem()
                    
                    x = [0, 1] if disease == 'Healthy' else [0.1, 1.1]
                    color = '#3498DB' if disease == 'Healthy' else '#E74C3C'
                    
                    ax.errorbar(['No Coll', '+Coll'], 
                               [means.get('NoCollagen', 0), means.get('Collagen', 0)],
                               yerr=[sems.get('NoCollagen', 0), sems.get('Collagen', 0)],
                               marker='o', markersize=10, capsize=5, linewidth=2,
                               color=color, label=disease)
                
                fdr_p = results_df[results_df['Feature'] == feat]['Interaction_FDR'].values[0]
                ax.set_title(f'{feat}\nInteraction FDR p = {fdr_p:.4f} *', fontsize=11, fontweight='bold')
                ax.set_xlabel('Collagen Treatment', fontsize=11)
                ax.set_ylabel(f'Mean {feat}', fontsize=11)
                ax.legend(title='Disease', fontsize=10)
                ax.grid(alpha=0.3)
            
            plt.suptitle('Disease × Collagen Interaction', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/lmm_interaction_plot.png', dpi=300)
            print(f"Saved: {OUTPUT_DIR}/lmm_interaction_plot.png")
    
    # 8e. Save summary table for thesis
    summary_table = results_df[['Feature', 'Disease_TAA_coef', 'Disease_TAA_pval', 'Disease_TAA_FDR', 'ICC']].copy()
    summary_table.columns = ['Feature', 'Coefficient', 'p-value', 'FDR p-value', 'ICC']
    summary_table['Significant'] = summary_table['FDR p-value'] < 0.05
    summary_table = summary_table.round(4)
    summary_table.to_csv(f'{OUTPUT_DIR}/lmm_summary_table.csv', index=False)
    print(f"Saved: {OUTPUT_DIR}/lmm_summary_table.csv")
    
    # 8f. Comparison: LMM vs naive analysis
    print(f"\n{'='*80}")
    print("COMPARISON: LMM vs Naive Analysis")
    print(f"{'='*80}")
    print("\nLMM accounts for within-subject correlation (nested data structure).")
    print("Naive analysis treats each cell as independent (pseudoreplication).")
    print(f"\nMean ICC across features: {results_df['ICC'].mean():.3f}")
    print(f"  → {results_df['ICC'].mean()*100:.1f}% of variance is between subjects")
    print(f"  → {(1-results_df['ICC'].mean())*100:.1f}% of variance is within subjects")
    
    print(f"\n{'='*80}")
    print("Linear Mixed Model Analysis Complete")
    print(f"{'='*80}")
    
    # Thesis summary
    print(f"\n--- THESIS SUMMARY ---")
    print(f"Using Linear Mixed Models with subject as random effect:")
    print(f"  • Disease effect significant in {n_disease_sig} features (FDR < 0.05)")
    print(f"  • Collagen effect significant in {n_collagen_sig} features")
    print(f"  • Disease×Collagen interaction in {n_interaction_sig} features")
    print(f"  • Average ICC = {results_df['ICC'].mean():.2f} (subject explains {results_df['ICC'].mean()*100:.0f}% of variance)")

if __name__ == '__main__':
    main()

