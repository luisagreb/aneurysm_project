"""
Pairwise Comparisons - All Relevant Group Comparisons

Performs Mann-Whitney U tests for each pairwise comparison:
1. Disease Effect (No Collagen): Healthy_NoColl vs TAA_NoColl
2. Disease Effect (+Collagen): Healthy_Coll vs TAA_Coll
3. Collagen Effect (Healthy): Healthy_NoColl vs Healthy_Coll
4. Collagen Effect (TAA): TAA_NoColl vs TAA_Coll
5. Interaction: Healthy_NoColl vs TAA_Coll
6. Interaction: Healthy_Coll vs TAA_NoColl

Outputs:
- Summary tables for top 5 features per comparison
- Combined results CSV
- Visualization of significant features

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import os

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = 'src/analysis/src/analysis/outputs/pairwise_comparisons'
ALPHA = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_numeric_id(text):
    """Robustly extracts Subject ID from filename."""
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

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (group2.mean() - group1.mean()) / pooled_std

def run_pairwise_comparison(df, group1_name, group2_name, feature_cols):
    """Run Mann-Whitney U test between two groups for all features."""
    results = []
    
    group1_data = df[df['Group'] == group1_name]
    group2_data = df[df['Group'] == group2_name]
    
    for feature in feature_cols:
        g1 = group1_data[feature].dropna()
        g2 = group2_data[feature].dropna()
        
        if len(g1) < 3 or len(g2) < 3:
            continue
        
        try:
            stat, p_value = mannwhitneyu(g1, g2, alternative='two-sided')
        except:
            continue
        
        effect = cohens_d(g1, g2)
        
        results.append({
            'Feature': feature,
            f'{group1_name}_Mean': g1.mean(),
            f'{group1_name}_SD': g1.std(),
            f'{group1_name}_N': len(g1),
            f'{group2_name}_Mean': g2.mean(),
            f'{group2_name}_SD': g2.std(),
            f'{group2_name}_N': len(g2),
            'Mean_Difference': g2.mean() - g1.mean(),
            'Cohens_d': effect,
            'U_statistic': stat,
            'p_value': p_value
        })
    
    results_df = pd.DataFrame(results)
    
    # FDR correction
    if len(results_df) > 0:
        _, p_adjusted, _, _ = multipletests(results_df['p_value'], alpha=ALPHA, method='fdr_bh')
        results_df['p_adjusted_FDR'] = p_adjusted
        results_df['Significant'] = p_adjusted < ALPHA
        results_df = results_df.sort_values('p_value')
    
    return results_df

def create_summary_table(results_df, comparison_name, top_n=5):
    """Create a formatted summary table for top N features."""
    top = results_df.head(top_n).copy()
    
    # Format for display
    summary = pd.DataFrame({
        'Feature': top['Feature'],
        'Cohen\'s d': top['Cohens_d'].apply(lambda x: f"{x:.2f}"),
        'p-value': top['p_value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}"),
        'FDR p': top['p_adjusted_FDR'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}"),
        'Sig': top['Significant'].apply(lambda x: '***' if x else '')
    })
    
    return summary

def main():
    print("=" * 80)
    print("PAIRWISE COMPARISONS - All Group Combinations")
    print("=" * 80)
    
    # 1. Load and prepare data
    healthy_ids, taa_ids = load_metadata(METADATA_FILE)
    df = pd.read_csv(FEATURES_FILE)
    
    df['Numeric_ID'] = df['CellName'].apply(extract_numeric_id)
    df['Collagen_Status'] = df['CellName'].apply(extract_collagen_status)
    
    def get_disease_label(nid):
        if nid in healthy_ids:
            return 'Healthy'
        if nid in taa_ids:
            return 'TAA'
        return None
    
    df['Disease'] = df['Numeric_ID'].apply(get_disease_label)
    
    def get_group(row):
        if pd.isna(row['Disease']) or pd.isna(row['Collagen_Status']):
            return None
        return f"{row['Disease']}_{row['Collagen_Status']}"
    
    df['Group'] = df.apply(get_group, axis=1)
    df = df.dropna(subset=['Group'])
    
    print(f"\nDataset: {len(df)} cells")
    for g in df['Group'].unique():
        print(f"  {g}: {sum(df['Group'] == g)}")
    
    # 2. Define feature columns
    feature_cols = [col for col in df.columns if 
                    col.endswith('_µm') or col.endswith('_ratio') or col.endswith('_µm³') or 
                    col.endswith('_µm²') or col.endswith('_n')]
    
    # 3. Define all pairwise comparisons
    comparisons = [
        ('Healthy_NoCollagen', 'TAA_NoCollagen', 'Disease Effect (No Collagen)'),
        ('Healthy_Collagen', 'TAA_Collagen', 'Disease Effect (+Collagen)'),
        ('Healthy_NoCollagen', 'Healthy_Collagen', 'Collagen Effect (Healthy)'),
        ('TAA_NoCollagen', 'TAA_Collagen', 'Collagen Effect (TAA)'),
        ('Healthy_NoCollagen', 'TAA_Collagen', 'Healthy_NoColl vs TAA_Coll'),
        ('Healthy_Collagen', 'TAA_NoCollagen', 'Healthy_Coll vs TAA_NoColl'),
    ]
    
    # 4. Run all comparisons
    all_results = []
    summary_tables = {}
    
    print(f"\n{'='*80}")
    print("RUNNING PAIRWISE COMPARISONS (Mann-Whitney U + FDR)")
    print(f"{'='*80}")
    
    for group1, group2, name in comparisons:
        print(f"\n--- {name} ---")
        print(f"    {group1} vs {group2}")
        
        results = run_pairwise_comparison(df, group1, group2, feature_cols)
        
        if len(results) == 0:
            print("    No valid comparisons")
            continue
        
        # Add comparison info
        results['Comparison'] = name
        results['Group1'] = group1
        results['Group2'] = group2
        
        n_sig = sum(results['Significant'])
        print(f"    Significant features (FDR < 0.05): {n_sig}/{len(results)}")
        
        # Save individual comparison
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')
        results.to_csv(f'{OUTPUT_DIR}/{safe_name}.csv', index=False)
        
        # Create summary table
        summary = create_summary_table(results, name)
        summary_tables[name] = summary
        
        # Print top 5
        print(f"\n    Top 5 Features:")
        print(summary.to_string(index=False))
        
        all_results.append(results)
    
    # 5. Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(f'{OUTPUT_DIR}/all_pairwise_comparisons.csv', index=False)
    print(f"\n\nSaved: {OUTPUT_DIR}/all_pairwise_comparisons.csv")
    
    # 6. Create comparison summary figure
    # Sort features to ensure consistent order across all subplots
    # Order: Actin -> Mito -> Nucleus (displayed Top to Bottom, so plot y-order is reversed)
    def feature_sort_key(name):
        priorities = {'Actin': 0, 'Mito': 1, 'Nucleus': 2}
        prefix = name.split('_')[0]
        prio = priorities.get(prefix, 3)
        return (prio, name)

    # We want Actin at the top of the Y-axis. 
    # In barh, y=0 is the bottom. The first item in the list is plotted at y=0.
    # So if we want [Actin, Mito, Nucleus] from Top to Bottom:
    # Top (y=Max) = Actin
    # Bottom (y=0) = Nucleus
    # So the list should be [Nucleus, Mito, Actin]
    # Sorting with Actin=0, Mito=1, Nucleus=2 gives [Actin, Mito, Nucleus]
    # Reversing gives [Nucleus, Mito, Actin]. Perfect.
    unique_features = sorted(feature_cols, key=feature_sort_key, reverse=True)
    n_features = len(unique_features)
    
    # Adjustable height based on number of features (approx 0.4 inch per feature per subplot row)
    fig_height = max(15, n_features * 0.6)
    fig, axes = plt.subplots(2, 3, figsize=(24, fig_height))
    axes = axes.flatten()
    
    for i, (group1, group2, name) in enumerate(comparisons):
        ax = axes[i]
        
        # Get results for this comparison
        comp_results = combined_df[combined_df['Comparison'] == name].set_index('Feature')
        
        # Reindex to enforce consistent order and include all features
        plot_data = comp_results.reindex(unique_features)
        
        # Prepare data for plotting
        effect_sizes = plot_data['Cohens_d'].fillna(0)
        is_sig = plot_data['Significant'].fillna(False)
        
        # Colors
        colors = ['#E74C3C' if d > 0 else '#3498DB' for d in effect_sizes]
        
        y_pos = range(len(unique_features))
        ax.barh(y_pos, effect_sizes, color=colors, edgecolor='black', alpha=0.8)
        ax.set_yticks(y_pos)
        
        # Labels with significance
        labels = []
        for feat, sig in zip(unique_features, is_sig):
            if sig:
                labels.append(f"{feat} ***")
            else:
                labels.append(feat)
        
        ax.set_yticklabels(labels, fontsize=11)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=-0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel("Cohen's d", fontsize=12, fontweight='bold')
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Set symmetric x-axis limits for better comparison
        max_val = max(effect_sizes.abs().max(), 1.0) * 1.1
        ax.set_xlim(-max_val, max_val)
    
    plt.suptitle("Effect Sizes (Cohen's d) for All Pairwise Comparisons\nAll Features (Sorted by Structure)", 
                 fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93) # Make room for suptitle
    plt.savefig(f'{OUTPUT_DIR}/pairwise_effect_sizes.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR}/pairwise_effect_sizes.png")
    
    # 7. Create summary count table
    print(f"\n{'='*80}")
    print("SUMMARY: Significant Features per Comparison")
    print(f"{'='*80}")
    
    summary_counts = []
    for name in [c[2] for c in comparisons]:
        comp_data = combined_df[combined_df['Comparison'] == name]
        n_sig = sum(comp_data['Significant'])
        n_total = len(comp_data)
        summary_counts.append({
            'Comparison': name,
            'Significant': n_sig,
            'Total': n_total,
            'Percent': f"{100*n_sig/n_total:.1f}%" if n_total > 0 else "N/A"
        })
    
    summary_counts_df = pd.DataFrame(summary_counts)
    print(summary_counts_df.to_string(index=False))
    summary_counts_df.to_csv(f'{OUTPUT_DIR}/summary_counts.csv', index=False)
    
    # 8. Rescue Analysis: Compare rescued vs not-rescued features
    print(f"\n{'='*80}")
    print("RESCUE ANALYSIS: Features rescued by collagen")
    print(f"{'='*80}")
    
    # Load individual comparison results
    nocoll_results = combined_df[combined_df['Comparison'] == 'Disease Effect (No Collagen)']
    coll_results = combined_df[combined_df['Comparison'] == 'Disease Effect (+Collagen)']
    
    sig_nocoll = set(nocoll_results[nocoll_results['Significant']]['Feature'])
    sig_coll = set(coll_results[coll_results['Significant']]['Feature'])
    
    rescued_features = list(sig_nocoll - sig_coll)
    not_rescued_features = list(sig_nocoll & sig_coll)
    
    print(f"Features rescued by collagen: {len(rescued_features)}")
    for f in rescued_features:
        row = nocoll_results[nocoll_results['Feature'] == f].iloc[0]
        print(f"  - {f}: d={row['Cohens_d']:.2f}")
    
    print(f"\nFeatures still different with collagen: {len(not_rescued_features)}")
    for f in not_rescued_features:
        print(f"  - {f}")
    
    # 9. Create rescue boxplot (cleaner layout: 2 rows x 3 cols)
    # Prepare group labels for plotting
    df['Group_Label'] = df['Group'].map({
        'Healthy_NoCollagen': 'Healthy\nNo Coll',
        'Healthy_Collagen': 'Healthy\n+Coll',
        'TAA_NoCollagen': 'TAA\nNo Coll',
        'TAA_Collagen': 'TAA\n+Coll'
    })
    
    palette = {'Healthy\nNo Coll': '#3498DB', 'Healthy\n+Coll': '#85C1E9', 
               'TAA\nNo Coll': '#E74C3C', 'TAA\n+Coll': '#F1948A'}
    order = ['Healthy\nNo Coll', 'Healthy\n+Coll', 'TAA\nNo Coll', 'TAA\n+Coll']
    
    # Select features for plot: 3 rescued + 3 not rescued = 6 plots (2x3 grid)
    rescued_plot = rescued_features[:3] if len(rescued_features) >= 3 else rescued_features
    not_rescued_plot = not_rescued_features[:3] if len(not_rescued_features) >= 3 else not_rescued_features
    
    n_rescued = len(rescued_plot)
    n_not_rescued = len(not_rescued_plot)
    
    if n_rescued > 0 or n_not_rescued > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Top row: Rescued features
        for i in range(3):
            ax = axes[0, i]
            if i < n_rescued:
                feat = rescued_plot[i]
                sns.boxplot(data=df, x='Group_Label', y=feat, ax=ax, order=order, palette=palette)
                ax.set_title(f'{feat}\n(RESCUED)', fontsize=14, fontweight='bold', color='black')
                ax.set_xlabel('')
                ax.set_ylabel(feat.split('_')[0], fontsize=13)
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=11)
            else:
                ax.axis('off')
        
        # Bottom row: Not rescued features
        for i in range(3):
            ax = axes[1, i]
            if i < n_not_rescued:
                feat = not_rescued_plot[i]
                sns.boxplot(data=df, x='Group_Label', y=feat, ax=ax, order=order, palette=palette)
                ax.set_title(f'{feat}\n(NOT RESCUED)', fontsize=14, fontweight='bold', color='black')
                ax.set_xlabel('')
                ax.set_ylabel(feat.split('_')[0], fontsize=13)
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=11)
            else:
                ax.axis('off')
        
        plt.suptitle('Collagen Rescue Effect: TAA Phenotype Recovery', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/rescue_boxplots.png', dpi=300)
        print(f"\nSaved: {OUTPUT_DIR}/rescue_boxplots.png")

    # 10. Generate detailed interaction plots for ALL features
    print(f"\n{'='*80}")
    print("GENERATING INTERACTION PLOTS FOR ALL FEATURES")
    print(f"{'='*80}")
    
    # Create directory for plots
    plots_dir = os.path.join(OUTPUT_DIR, 'interaction_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot ALL features (not just significant ones)
    print(f"Generating plots for all {len(feature_cols)} features...")
    
    # Prepare data for interaction plotting
    plot_df = df.copy()
    
    # Ensure Collagen_Status is cleaner for plotting
    plot_df['Collagen_Status'] = plot_df['Collagen_Status'].replace({
        'Collagen': '+Collagen',
        'NoCollagen': 'No Collagen'
    })
    
    # Define order and palette
    collagen_order = ['No Collagen', '+Collagen']  
    disease_order = ['Healthy', 'TAA']
    palette = {'Healthy': '#2ECC71', 'TAA': '#E74C3C'}  # Green vs Red
    
    def format_pvalue(p):
        """Format p-value in scientific paper style."""
        if p < 0.001:
            return "***p<0.001"
        elif p < 0.01:
            return "**p<0.01"
        elif p < 0.05:
            return "*p<0.05"
        else:
            return f"ns (p={p:.3f})"
    
    def get_significance_stars(p):
        """Return significance stars."""
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""
    
    for feature in feature_cols:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Point plot with error bars (mean +/- ci)
            sns.pointplot(
                data=plot_df, 
                x='Collagen_Status', 
                y=feature, 
                hue='Disease', 
                order=collagen_order,
                hue_order=disease_order,
                palette=palette,
                markers=['o', 's'],
                capsize=0.1,
                linestyles=['-', '-'],
                errorbar=('ci', 95), # 95% confidence interval
                dodge=True,
                ax=ax
            )
            
            # Collect p-values for all relevant comparisons for this feature
            p_annotations = []
            
            for res in all_results:
                if feature in res['Feature'].values:
                    row = res[res['Feature'] == feature].iloc[0]
                    comp_name = res['Comparison'].iloc[0]
                    p_fdr = row['p_adjusted_FDR']
                    is_sig = row['Significant']
                    
                    # Format based on comparison type
                    if 'Disease Effect' in comp_name:
                        if 'No Collagen' in comp_name:
                            label = "Disease (No Coll)"
                        else:
                            label = "Disease (+Coll)"
                    elif 'Collagen Effect' in comp_name:
                        if 'Healthy' in comp_name:
                            label = "Coll (Healthy)"
                        else:
                            label = "Coll (TAA)"
                    else:
                        label = comp_name
                    
                    p_str = format_pvalue(p_fdr)
                    p_annotations.append(f"{label}: {p_str}")
            
            # Add title with feature name
            title_text = f"Interaction Effect: {feature}"
            ax.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
            
            # Add p-value annotations in a text box
            if p_annotations:
                # Join first 4 most important comparisons
                # Priority: Disease (No Coll), Disease (+Coll), Coll (TAA), Coll (Healthy)
                ordered_annotations = []
                priority_order = ["Disease (No Coll)", "Disease (+Coll)", "Coll (TAA)", "Coll (Healthy)"]
                
                for priority in priority_order:
                    for annot in p_annotations:
                        if annot.startswith(priority):
                            ordered_annotations.append(annot)
                
                # Add any remaining
                for annot in p_annotations:
                    if annot not in ordered_annotations:
                        ordered_annotations.append(annot)
                
                # Show top 4
                textstr = '\n'.join(ordered_annotations[:4])
                
                # Add text box
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props, family='monospace')
            
            ax.set_ylabel(feature, fontsize=12)
            ax.set_xlabel('')
            ax.legend(title='Group', fontsize=11, title_fontsize=12, loc='upper right')
            ax.grid(axis='y', alpha=0.3)
            
            # Save plot
            fname = f"{feature}_interaction.png"
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, fname), dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Could not plot {feature}: {e}")
            plt.close()

    print(f"Saved {len(feature_cols)} interaction plots to {plots_dir}")

if __name__ == '__main__':
    main()

