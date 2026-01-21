"""
Phase 1: Statistical Discovery 

Goal: Prove that morphological features show real biological differences
      between Healthy and TAA cells, not random noise.

Outputs:
- Statistical test results with FDR correction
- Normality test results
- Boxplots/violin plots for top significant features
- Summary table for Thesis Section 3.1

Author: Antigravity AI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import shapiro, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import os

# Configuration - paths relative to project root (run from aneurysm_project/)
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = 'src/analysis/src/analysis/outputs/stats_TAAvsHealthyNoColl'
ALPHA = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_numeric_id(text):
    """
    Robustly extracts Subject ID from filename.
    Assumes standard format: "<SubjectID> <Conditions>..."
    From train_classifier.py
    """
    if pd.isna(text):
        return None
        
    text = str(text).strip()
    
    # Step 1: Get first token (Subject ID part)
    first_token = text.split(' ')[0]
    
    # Step 2: Remove trailing hyphens
    token = first_token.strip('-')
    
    # Step 3: Extract number
    # Priority A: Hyphen separator (common in 01C-XXX, 01ASC-XXX)
    parts = token.split('-')
    if len(parts) > 1:
        last_part = parts[-1]
        # Check if the part after hyphen is numeric
        if last_part.isdigit():
            return int(last_part)
        # If not purely digit, try extracting digits
        sub_digits = re.findall(r'\d+', last_part)
        if sub_digits:
            return int(sub_digits[0])
            
    # Priority B: No hyphen (e.g. 03Asc46), just take the last number group
    digits = re.findall(r'\d+', token)
    if digits:
        return int(digits[-1])
        
    return None

def load_metadata(filepath):
    """Load metadata to classify Healthy vs TAA."""
    print(f"Loading metadata from {filepath}...")
    df = pd.read_excel(filepath, header=None, skiprows=18)
    
    healthy_ids = set()
    taa_ids = set()
    
    for x in df[0].dropna():
        nid = extract_numeric_id(x)
        if nid is not None:
            healthy_ids.add(nid)
    
    for x in df[1].dropna():
        nid = extract_numeric_id(x)
        if nid is not None:
            taa_ids.add(nid)
    
    print(f"  Healthy IDs: {sorted(list(healthy_ids))}")
    print(f"  TAA IDs: {sorted(list(taa_ids))}")
    return healthy_ids, taa_ids

def extract_collagen_status(filename):
    """
    Extract collagen status from filename.
    From train_collagen_classifier.py
    +coll or +Coll -> 'Collagen'
    -coll or no +coll or DMSO -> 'NoCollagen'
    """
    if pd.isna(filename):
        return None
    
    filename = str(filename).lower()
    
    # Positive Collagen
    if '+coll' in filename or '+col' in filename or 'plus coll' in filename:
        return 'Collagen'
        
    # Negative Collagen (including DMSO control)
    if '-coll' in filename or '-col' in filename or 'nocoll' in filename or 'no coll' in filename or 'dmso' in filename:
        return 'NoCollagen'
        
    return None

def main():
    print("=" * 80)
    print("PHASE 1: STATISTICAL DISCOVERY - Healthy vs TAA (No Collagen)")
    print("=" * 80)
    
    # 1. Load data
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    df = pd.read_csv(FEATURES_FILE)
    
    # 2. Add labels
    df['Numeric_ID'] = df['CellName'].apply(extract_numeric_id)
    df['Collagen_Status'] = df['CellName'].apply(extract_collagen_status)
    
    def get_disease_label(nid):
        if nid in healthy_ids:
            return 'Healthy'
        if nid in taa_ids:
            return 'TAA'
        return None
    
    df['Disease'] = df['Numeric_ID'].apply(get_disease_label)
    
    # 3. Filter to No Collagen only
    df_nocol = df[df['Collagen_Status'] == 'NoCollagen'].copy()
    df_nocol = df_nocol.dropna(subset=['Disease'])
    
    print(f"\nDataset: {len(df_nocol)} cells (No Collagen)")
    print(f"  Healthy: {sum(df_nocol['Disease'] == 'Healthy')}")
    print(f"  TAA: {sum(df_nocol['Disease'] == 'TAA')}")
    
    # 4. Get feature columns
    feature_cols = [col for col in df_nocol.columns if 
                    col.endswith('_µm') or col.endswith('_ratio') or col.endswith('_µm³') or 
                    col.endswith('_µm²') or col.endswith('_n')]
    
    print(f"\nTesting {len(feature_cols)} features...")
    
    # 5. Run statistical tests
    results = []
    
    for feature in feature_cols:
        healthy = df_nocol[df_nocol['Disease'] == 'Healthy'][feature].dropna()
        taa = df_nocol[df_nocol['Disease'] == 'TAA'][feature].dropna()
        
        if len(healthy) < 3 or len(taa) < 3:
            continue
        
        # Normality tests
        _, p_norm_healthy = shapiro(healthy) if len(healthy) < 5000 else (None, 1.0)
        _, p_norm_taa = shapiro(taa) if len(taa) < 5000 else (None, 1.0)
        is_normal = (p_norm_healthy > 0.05) and (p_norm_taa > 0.05)
        
        # Choose test
        if is_normal:
            _, p_value = ttest_ind(healthy, taa, equal_var=False)
            test_used = 't-test'
        else:
            _, p_value = mannwhitneyu(healthy, taa, alternative='two-sided')
            test_used = 'Mann-Whitney U'
        
        # Effect size (Cohen's d for t-test, rank-biserial for Mann-Whitney)
        mean_diff = taa.mean() - healthy.mean()
        pooled_std = np.sqrt((healthy.std()**2 + taa.std()**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'Feature': feature,
            'Healthy_Mean': healthy.mean(),
            'Healthy_SD': healthy.std(),
            'Healthy_N': len(healthy),
            'TAA_Mean': taa.mean(),
            'TAA_SD': taa.std(),
            'TAA_N': len(taa),
            'Mean_Difference': mean_diff,
            'Effect_Size_Cohens_d': cohens_d,
            'Normal_Healthy': p_norm_healthy,
            'Normal_TAA': p_norm_taa,
            'Test': test_used,
            'p_value': p_value
        })
    
    results_df = pd.DataFrame(results)
    
    # 6. FDR correction (Benjamini-Hochberg)
    _, p_adjusted, _, _ = multipletests(results_df['p_value'], alpha=ALPHA, method='fdr_bh')
    results_df['p_adjusted_FDR'] = p_adjusted
    results_df['Significant_FDR'] = p_adjusted < ALPHA
    
    # Sort by p-value
    results_df = results_df.sort_values('p_value')
    
    # 7. Save results
    results_df.to_csv(f'{OUTPUT_DIR}/statistical_results.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR}/statistical_results.csv")
    
    # 8. Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total features tested: {len(results_df)}")
    print(f"Significant (p < 0.05): {sum(results_df['p_value'] < 0.05)}")
    print(f"Significant (FDR < 0.05): {sum(results_df['Significant_FDR'])}")
    
    print(f"\nMost Significant Features:")
    print(results_df[['Feature', 'Mean_Difference', 'Effect_Size_Cohens_d', 'p_value', 'p_adjusted_FDR']].head(10).to_string(index=False))
    
    # 9. Generate visualizations for top 6 features
    top_features = results_df.head(6)['Feature'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        
        # Prepare data
        plot_data = df_nocol[['Disease', feature]].dropna()
        
        # Violin plot
        sns.violinplot(data=plot_data, x='Disease', y=feature, ax=ax, palette={'Healthy': '#3498DB', 'TAA': '#E74C3C'})
        
        # Add boxplot overlay
        sns.boxplot(data=plot_data, x='Disease', y=feature, ax=ax, width=0.3, 
                    palette={'Healthy': '#3498DB', 'TAA': '#E74C3C'}, 
                    boxprops=dict(alpha=0.7))
        
        # Get stats for this feature
        feature_stats = results_df[results_df['Feature'] == feature].iloc[0]
        p_val = feature_stats['p_adjusted_FDR']
        
        # Add significance stars
        if p_val < 0.001:
            sig_label = '***'
        elif p_val < 0.01:
            sig_label = '**'
        elif p_val < 0.05:
            sig_label = '*'
        else:
            sig_label = 'ns'
        
        # Format p-value in scientific notation for very small values
        if p_val < 0.001:
            p_str = f"p = {p_val:.2e}"  # Scientific notation (e.g., 4.77e-07)
        else:
            p_str = f"p = {p_val:.4f}"  # Standard format
        
        ax.set_title(f"{feature}\n{p_str} {sig_label}", fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(feature, fontsize=9)
    
    plt.suptitle('Most Significant Features (Healthy vs TAA, No Collagen)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top_features_violin_plots.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/top_features_violin_plots.png")
    
    # 10. Cohen's d effect size bar plot (all features)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by absolute effect size
    results_sorted = results_df.sort_values('Effect_Size_Cohens_d', key=abs, ascending=True)
    
    # Color by direction: positive (TAA higher) = red, negative (TAA lower) = blue
    colors = ['#E74C3C' if d > 0 else '#3498DB' for d in results_sorted['Effect_Size_Cohens_d']]
    
    # Create horizontal bar plot
    y_pos = range(len(results_sorted))
    ax.barh(y_pos, results_sorted['Effect_Size_Cohens_d'], color=colors, edgecolor='black', alpha=0.8)
    
    # Add feature names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_sorted['Feature'], fontsize=9)
    
    # Add reference lines for effect size thresholds
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=-0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=-0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Labels
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title("Effect Size (Cohen's d) for All Features", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend in bottom right
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#E74C3C', edgecolor='black', label='TAA > Healthy'),
        Patch(facecolor='#3498DB', edgecolor='black', label='TAA < Healthy'),
        Line2D([0], [0], color='gray', linestyle='--', label='Large effect (|d|>0.8)'),
        Line2D([0], [0], color='gray', linestyle=':', label='Medium effect (|d|>0.5)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cohens_d_effect_sizes.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/cohens_d_effect_sizes.png")
    
    # 11. Subject-level aggregation (average across cells per subject)
    df_subject = df_nocol.groupby(['Numeric_ID', 'Disease'])[feature_cols].mean().reset_index()
    print(f"\nSubject-level analysis: {len(df_subject)} subjects")
    
    # 11. Histogram plots with KDE for top 6 features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        
        healthy = df_nocol[df_nocol['Disease'] == 'Healthy'][feature].dropna()
        taa = df_nocol[df_nocol['Disease'] == 'TAA'][feature].dropna()
        
        sns.histplot(healthy, kde=True, stat='density', label='Healthy', color='#3498DB', alpha=0.6, ax=ax)
        sns.histplot(taa, kde=True, stat='density', label='TAA', color='#E74C3C', alpha=0.6, ax=ax)
        ax.legend()
        ax.set_title(feature, fontsize=10, fontweight='bold')
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
    
    plt.suptitle('Distribution Histograms with KDE (Healthy vs TAA)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top_features_histograms.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/top_features_histograms.png")
    
    # 12. QQ plots for normality assessment (top 6 features)
    from scipy import stats as scipy_stats
    
    for feature in top_features:
        healthy = df_nocol[df_nocol['Disease'] == 'Healthy'][feature].dropna()
        taa = df_nocol[df_nocol['Disease'] == 'TAA'][feature].dropna()
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        scipy_stats.probplot(healthy, plot=axes[0])
        axes[0].set_title(f"{feature} – Healthy", fontsize=11, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        scipy_stats.probplot(taa, plot=axes[1])
        axes[1].set_title(f"{feature} – TAA", fontsize=11, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.suptitle(f'QQ Plot: {feature}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Clean filename
        safe_feature = feature.replace('/', '_').replace('²', '2').replace('³', '3').replace('µ', 'u')
        plt.savefig(f'{OUTPUT_DIR}/qq_plot_{safe_feature}.png', dpi=300)
        plt.close() 
    
    print(f"Saved: {OUTPUT_DIR}/qq_plot_*.png (6 QQ plots)")
    
    print(f"\n{'='*80}")
    print("3D Morphometric Analysis Complete")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
