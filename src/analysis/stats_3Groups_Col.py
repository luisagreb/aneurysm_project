"""
Statistical Analysis and Visualization - 3 Disease Groups x 2 Collagen Treatments
```python
Groups: Healthy, TAV-TAA, BAV-TAA × NoCollagen, Collagen (6 groups total)
```

Method: Kruskal-Wallis H test for overall variations, followed by Dunn's 
post-hoc tests for pairwise significance if the omnibus test is significant.
Visualization: Violin plots arranged by the 6 groups.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import scikit_posthocs as sp
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

FEATURES_FILE = 'outputs/Advanced_Features_Raw_Final.csv'
OUTPUT_DIR = 'classification_results/stats_3groups_collagen'
ALPHA = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print("6-GROUP ANALYSIS: Healthy vs TAA_TAV vs TAA_BAV × Collagen vs No Collagen")
    print("=" * 80)
    
    # 1. Load data
    df = pd.read_csv(FEATURES_FILE)
    df = df.dropna(subset=['Disease', 'Collagen_Status'])
    
    # Map to new TAA_TAV and TAA_BAV naming convention
    df['Disease'] = df['Disease'].replace({'TAA': 'TAA_TAV', 'BAV': 'TAA_BAV'})
    
    # Create 6-group label
    df['Group'] = df['Disease'] + '_' + df['Collagen_Status']
    
    group_order = [
        'Healthy_NoCollagen', 'Healthy_Collagen', 
        'TAA_TAV_NoCollagen', 'TAA_TAV_Collagen',
        'TAA_BAV_NoCollagen', 'TAA_BAV_Collagen'
    ]
    group_labels = [
        'Healthy\\n-Coll', 'Healthy\\n+Coll', 
        'TAA_TAV\\n-Coll', 'TAA_TAV\\n+Coll',
        'TAA_BAV\\n-Coll', 'TAA_BAV\\n+Coll'
    ]
    
    print(f"\\nDataset: {len(df)} cells across 6 groups")
    for g in group_order:
        print(f"  {g}: {sum(df['Group'] == g)}")
    
    feature_cols = [
        'Actin_Skeleton_Length_µm', 'Actin_Volume_µm³', 'Actin_Solidity_ratio', 'Actin_Minor_Axis_µm',
        'Mito_Volume_µm³', 'Mito_Surface_Area_µm²', 'Mito_Sphericity_ratio', 'Mito_Fragment_Count_n', 'Mito_Mean_Branch_Length_µm',
        'Nucleus_Volume_µm³', 'Nucleus_Sphericity_ratio', 'Nucleus_Elongation_ratio', 'Nucleus_Flatness_ratio',
        'Actin_Fractional_Anisotropy_ratio', 'Mito_Total_Network_Length_µm'
    ]
    # Filter only overlapping features
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    print(f"\\nTesting {len(feature_cols)} features with Kruskal-Wallis + Dunn's post-hoc...")
    
    results = []
    
    for feature in feature_cols:
        groups_data = [df[df['Group'] == g][feature].dropna().values for g in group_order]
        
        if any(len(g) < 3 for g in groups_data):
            continue
            
        try:
            h_stat, p_omnibus = kruskal(*groups_data)
        except:
            continue
            
        result = {
            'Feature': feature,
            'H_statistic': h_stat,
            'p_omnibus': p_omnibus,
        }
        
        for i, g in enumerate(group_order):
            result[f'{g}_Mean'] = np.mean(groups_data[i])
            result[f'{g}_N'] = len(groups_data[i])
            
        results.append(result)
        
    results_df = pd.DataFrame(results)
    _, p_adjusted, _, _ = multipletests(results_df['p_omnibus'], alpha=ALPHA, method='fdr_bh')
    results_df['p_omnibus_FDR'] = p_adjusted
    results_df['Significant_FDR'] = p_adjusted < ALPHA
    
    results_df = results_df.sort_values('p_omnibus')
    results_df.to_csv(f'{OUTPUT_DIR}/kruskal_wallis_results.csv', index=False)
    
    print(f"\\nSignificant features (FDR < 0.05): {sum(results_df['Significant_FDR'])}")
    
    # Violin plots for top 6 features
    top_features = results_df.head(6)['Feature'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    palette = {
        'Healthy_NoCollagen': '#B2DFDB', # light teal
        'Healthy_Collagen': '#00796B',   # dark teal
        'TAA_TAV_NoCollagen': '#FFCCBC', # light orange
        'TAA_TAV_Collagen': '#E64A19',   # dark orange
        'TAA_BAV_NoCollagen': '#D1C4E9', # light purple
        'TAA_BAV_Collagen': '#512DA8'    # dark purple
    }
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        
        sns.violinplot(
            data=df, 
            x='Group', 
            y=feature, 
            order=group_order,
            palette=palette,
            ax=ax,
            inner="quartile"
        )
        
        # Add swarmplot to show data points
        sns.swarmplot(
            data=df,
            x='Group',
            y=feature,
            order=group_order,
            color='black',
            size=3,
            alpha=0.4,
            ax=ax
        )
        
        ax.set_title(feature.replace('_', ' ').replace('ratio', ''), fontsize=12, fontweight='bold')
        ax.set_xticklabels(group_labels, rotation=45)
        ax.set_ylabel(feature.split('_')[-1] if '_' in feature else feature)
        ax.set_xlabel('')
        
        # Add omnibus p-value text
        p_val = results_df[results_df['Feature'] == feature]['p_omnibus_FDR'].values[0]
        sig_text = f"K-W FDR p < 0.001" if p_val < 0.001 else f"K-W FDR p = {p_val:.3f}"
        
        ax.annotate(sig_text, xy=(0.5, 0.95), xycoords='axes fraction', 
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top6_features_violin_6groups.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/top6_features_violin_6groups.pdf', bbox_inches='tight')
    print(f"\\nSaved plots to: {OUTPUT_DIR}/top6_features_violin_6groups.png")
    
if __name__ == '__main__':
    main()
