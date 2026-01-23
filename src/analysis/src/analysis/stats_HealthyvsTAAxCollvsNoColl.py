"""
4-Group Statistical Analysis: Healthy vs TAA × Collagen vs No Collagen

Goal: Compare morphological features across 4 groups using:
  1. Kruskal-Wallis H test (omnibus non-parametric test)
  2. Dunn's post-hoc test with Bonferroni correction

Groups:
  1. Healthy - No Collagen
  2. Healthy + Collagen
  3. TAA - No Collagen
  4. TAA + Collagen

Outputs:
- Statistical results CSV with omnibus and post-hoc p-values
- Violin plots for top features across all 4 groups
- Cohen's d effect size plots

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import shapiro, kruskal
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import os

# Configuration - paths relative to project root (run from aneurysm_project/)
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = 'src/analysis/src/analysis/outputs/stats_HealthyvsTAAxCollvsNoColl'
ALPHA = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_numeric_id(text):
    """
    Robustly extracts Subject ID from filename.
    """
    if pd.isna(text):
        return None
        
    text = str(text).strip()
    first_token = text.split(' ')[0]
    token = first_token.strip('-')
    
    parts = token.split('-')
    if len(parts) > 1:
        last_part = parts[-1]
        if last_part.isdigit():
            return int(last_part)
        sub_digits = re.findall(r'\d+', last_part)
        if sub_digits:
            return int(sub_digits[0])
            
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
    """
    if pd.isna(filename):
        return None
    
    filename = str(filename).lower()
    
    if '+coll' in filename or '+col' in filename or 'plus coll' in filename:
        return 'Collagen'
        
    if '-coll' in filename or '-col' in filename or 'nocoll' in filename or 'no coll' in filename or 'dmso' in filename:
        return 'NoCollagen'
        
    return None

def main():
    print("=" * 80)
    print("4-GROUP ANALYSIS: Healthy vs TAA × Collagen vs No Collagen")
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
    
    # 3. Create 4-group label
    def get_group(row):
        if pd.isna(row['Disease']) or pd.isna(row['Collagen_Status']):
            return None
        return f"{row['Disease']}_{row['Collagen_Status']}"
    
    df['Group'] = df.apply(get_group, axis=1)
    df = df.dropna(subset=['Group'])
    
    # Define group order
    group_order = ['Healthy_NoCollagen', 'Healthy_Collagen', 'TAA_NoCollagen', 'TAA_Collagen']
    group_labels = ['Healthy\nNo Coll', 'Healthy\n+Coll', 'TAA\nNo Coll', 'TAA\n+Coll']
    
    print(f"\nDataset: {len(df)} cells across 4 groups")
    for g in group_order:
        print(f"  {g}: {sum(df['Group'] == g)}")
    
    # 4. Get feature columns
    feature_cols = [col for col in df.columns if 
                    col.endswith('_µm') or col.endswith('_ratio') or col.endswith('_µm³') or 
                    col.endswith('_µm²') or col.endswith('_n')]
    
    print(f"\nTesting {len(feature_cols)} features with Kruskal-Wallis + Dunn's post-hoc...")
    
    # 5. Run statistical tests
    results = []
    posthoc_results = []
    
    for feature in feature_cols:
        # Get data for each group
        groups_data = [df[df['Group'] == g][feature].dropna().values for g in group_order]
        
        # Skip if any group has < 3 samples
        if any(len(g) < 3 for g in groups_data):
            continue
        
        # Kruskal-Wallis H test (omnibus)
        try:
            h_stat, p_omnibus = kruskal(*groups_data)
        except:
            continue
        
        # Store group statistics
        result = {
            'Feature': feature,
            'H_statistic': h_stat,
            'p_omnibus': p_omnibus,
        }
        
        for i, g in enumerate(group_order):
            result[f'{g}_Mean'] = np.mean(groups_data[i])
            result[f'{g}_SD'] = np.std(groups_data[i])
            result[f'{g}_N'] = len(groups_data[i])
        
        # Dunn's post-hoc test (only if omnibus is significant)
        if p_omnibus < ALPHA:
            feature_data = df[df['Group'].isin(group_order)][['Group', feature]].dropna()
            try:
                dunn_result = sp.posthoc_dunn(feature_data, val_col=feature, group_col='Group', p_adjust='bonferroni')
                
                # Extract pairwise comparisons
                comparisons = [
                    ('Healthy_NoCollagen', 'TAA_NoCollagen', 'Disease_NoColl'),       # Disease effect (no coll)
                    ('Healthy_Collagen', 'TAA_Collagen', 'Disease_Coll'),             # Disease effect (+coll)
                    ('Healthy_NoCollagen', 'Healthy_Collagen', 'Collagen_Healthy'),   # Collagen effect (healthy)
                    ('TAA_NoCollagen', 'TAA_Collagen', 'Collagen_TAA'),               # Collagen effect (TAA)
                    ('Healthy_NoCollagen', 'TAA_Collagen', 'DiagColl_vs_HealthyNoColl'),
                    ('Healthy_Collagen', 'TAA_NoCollagen', 'HealthyColl_vs_TAA_NoColl'),
                ]
                
                for g1, g2, name in comparisons:
                    if g1 in dunn_result.index and g2 in dunn_result.columns:
                        result[f'p_{name}'] = dunn_result.loc[g1, g2]
                        
                        posthoc_results.append({
                            'Feature': feature,
                            'Comparison': name,
                            'Group1': g1,
                            'Group2': g2,
                            'p_value': dunn_result.loc[g1, g2],
                            'Significant': dunn_result.loc[g1, g2] < ALPHA
                        })
            except Exception as e:
                print(f"  Warning: Dunn's test failed for {feature}: {e}")
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # 6. FDR correction on omnibus p-values
    _, p_adjusted, _, _ = multipletests(results_df['p_omnibus'], alpha=ALPHA, method='fdr_bh')
    results_df['p_omnibus_FDR'] = p_adjusted
    results_df['Significant_FDR'] = p_adjusted < ALPHA
    
    # Sort by omnibus p-value
    results_df = results_df.sort_values('p_omnibus')
    
    # 7. Save results
    results_df.to_csv(f'{OUTPUT_DIR}/kruskal_wallis_results.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR}/kruskal_wallis_results.csv")
    
    if posthoc_results:
        posthoc_df = pd.DataFrame(posthoc_results)
        posthoc_df.to_csv(f'{OUTPUT_DIR}/dunns_posthoc_results.csv', index=False)
        print(f"Saved: {OUTPUT_DIR}/dunns_posthoc_results.csv")
    
    # 8. Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total features tested: {len(results_df)}")
    print(f"Significant (omnibus p < 0.05): {sum(results_df['p_omnibus'] < 0.05)}")
    print(f"Significant (FDR < 0.05): {sum(results_df['Significant_FDR'])}")
    
    print(f"\nMost Significant Features (Kruskal-Wallis):")
    print(results_df[['Feature', 'H_statistic', 'p_omnibus', 'p_omnibus_FDR']].head(10).to_string(index=False))
    
    # 9. Violin plots for top 6 features
    top_features = results_df.head(6)['Feature'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    palette = {
        'Healthy_NoCollagen': '#3498DB',
        'Healthy_Collagen': '#85C1E9',
        'TAA_NoCollagen': '#E74C3C',
        'TAA_Collagen': '#F1948A'
    }
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        
        plot_data = df[df['Group'].isin(group_order)][['Group', feature]].dropna()
        plot_data['Group'] = pd.Categorical(plot_data['Group'], categories=group_order, ordered=True)
        
        sns.boxplot(data=plot_data, x='Group', y=feature, ax=ax, 
                    order=group_order, palette=palette)
        
        # Get stats
        feature_stats = results_df[results_df['Feature'] == feature].iloc[0]
        p_val = feature_stats['p_omnibus_FDR']
        
        if p_val < 0.001:
            p_str = f"p = {p_val:.2e} ***"
        elif p_val < 0.01:
            p_str = f"p = {p_val:.4f} **"
        elif p_val < 0.05:
            p_str = f"p = {p_val:.4f} *"
        else:
            p_str = f"p = {p_val:.4f}"
        
        ax.set_title(f"{feature}\n{p_str}", fontsize=14, fontweight='bold')
        ax.set_xticklabels(group_labels, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel(feature, fontsize=13)
        ax.tick_params(axis='y', labelsize=11)
    
    plt.suptitle('Most Significant Features (4-Group Comparison)\nKruskal-Wallis with FDR correction', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top_features_boxplots_4groups.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/top_features_boxplots_4groups.png")
    
    # 10. Post-hoc comparison heatmap for significant features
    if posthoc_results:
        posthoc_df = pd.DataFrame(posthoc_results)
        
        # Create summary heatmap
        comparison_names = ['Disease_NoColl', 'Disease_Coll', 'Collagen_Healthy', 'Collagen_TAA']
        comparison_labels = ['Disease\n(No Coll)', 'Disease\n(+Coll)', 'Collagen\n(Healthy)', 'Collagen\n(TAA)']
        
        # Get top 10 significant features
        top_sig = results_df[results_df['Significant_FDR']].head(10)['Feature'].tolist()
        
        if top_sig:
            heatmap_data = []
            for feat in top_sig:
                row = {'Feature': feat}
                for comp in comparison_names:
                    subset = posthoc_df[(posthoc_df['Feature'] == feat) & (posthoc_df['Comparison'] == comp)]
                    if len(subset) > 0:
                        p = subset['p_value'].values[0]
                        row[comp] = -np.log10(p) if p > 0 else 10  # Transform for visualization
                    else:
                        row[comp] = 0
                heatmap_data.append(row)
            
            heatmap_df = pd.DataFrame(heatmap_data).set_index('Feature')
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                       xticklabels=comparison_labels,
                       cbar_kws={'label': '-log10(p-value)'})
            ax.set_title("Dunn's Post-hoc Test Results\nHigher values = more significant", 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Feature', fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/posthoc_heatmap.png', dpi=300)
            print(f"Saved: {OUTPUT_DIR}/posthoc_heatmap.png")
    
    print(f"\n{'='*80}")
    print("4-Group Statistical Analysis Complete")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
