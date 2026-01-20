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
from scipy.stats import shapiro, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import os

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = 'src/analysis/outputs/phase1'
ALPHA = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_numeric_id(text):
    """Extract subject ID from filename (e.g., '01ASC-0180' -> 180)."""
    import re
    if pd.isna(text):
        return None
    text = str(text).strip()
    
    # Look for pattern like "ASC-####" or "asc-####"
    match = re.search(r'ASC-0?(\d{2,4})', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Fallback: split on "-" and get number after last dash
    parts = text.split('-')
    if len(parts) > 1:
        # Get the part after the last dash, before any space
        last_part = parts[-1].split()[0]
        # Remove leading zeros and convert
        digits = re.findall(r'\d+', last_part)
        if digits:
            return int(digits[0])
    
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
    """Extract collagen status (+coll or no coll)."""
    if pd.isna(filename):
        return None
    # Remove spaces and convert to lowercase for robust matching
    filename_clean = str(filename).replace(' ', '').lower()
    if '+coll' in filename_clean or '+col' in filename_clean:
        return 'Collagen'
    elif '-coll' in filename_clean or '-col' in filename_clean or 'nocoll' in filename_clean:
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
    
    print(f"\nTop 10 Most Significant Features (FDR-corrected):")
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
        
        ax.set_title(f"{feature}\np = {p_val:.4f} {sig_label}", fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(feature, fontsize=9)
    
    plt.suptitle('Phase 1: Top Significant Features (Healthy vs TAA, No Collagen)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top_features_violin_plots.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/top_features_violin_plots.png")
    
    print(f"\n{'='*80}")
    print("3D Morphometric Analysis Complete")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
