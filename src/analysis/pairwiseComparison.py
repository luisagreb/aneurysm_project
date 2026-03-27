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
    
    # Define helper function for significance stars (needed for boxplot summary)
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
    
    # 6. Create comparison summary figure with Effect Size BAR CHARTS
    # Show ALL features with Cohen's d as horizontal bars
    # Order features by "Disease Effect (No Collagen)" significance
    
    # Get feature order from "Disease Effect (No Collagen)" - most significant first
    disease_no_coll_results = combined_df[combined_df['Comparison'] == 'Disease Effect (No Collagen)'].sort_values('p_adjusted_FDR')
    feature_order = disease_no_coll_results['Feature'].tolist()
    
    # Reverse the order so most significant is at TOP
    feature_order_reversed = feature_order[::-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(48, 32))  # LANDSCAPE: Wider than tall
    axes = axes.flatten()
    
    for i, (group1, group2, name) in enumerate(comparisons):
        ax = axes[i]
        
        # Get results for this comparison
        comp_results = combined_df[combined_df['Comparison'] == name]
        
        # Reorder by the pre-defined feature order (reversed for top-to-bottom)
        comp_results_ordered = comp_results.set_index('Feature').loc[feature_order_reversed].reset_index()
        
        if len(comp_results_ordered) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=18)
            ax.set_title(name, fontsize=18, fontweight='bold')
            continue
        
        # Bar plot of Cohen's d with colors based on direction
        colors = ['#E74C3C' if d > 0 else '#3498DB' for d in comp_results_ordered['Cohens_d']]
        
        y_pos = range(len(comp_results_ordered))
        bars = ax.barh(y_pos, comp_results_ordered['Cohens_d'], color=colors, 
                       edgecolor='black', alpha=0.7, linewidth=0.8)
        
        ax.set_yticks(y_pos)
        
        # Create labels WITHOUT stars but with color-coding
        labels = []
        label_colors = []
        for _, row in comp_results_ordered.iterrows():
            feat = row['Feature']
            # Remove stars from labels - just keep feature name
            labels.append(feat)
            # Still color by significance
            if row['Significant']:
                label_colors.append('#D32F2F')  # Red for significant
            else:
                label_colors.append('#424242')  # Dark gray for non-significant
        
        ax.set_yticklabels(labels, fontsize=15)
        
        # Color each y-tick label individually
        for ticklabel, color in zip(ax.get_yticklabels(), label_colors):
            ticklabel.set_color(color)
            # Make significant features bold
            if color == '#D32F2F':
                ticklabel.set_weight('bold')
        
        # Add significance stars at the end of bars
        for idx, (_, row) in enumerate(comp_results_ordered.iterrows()):
            stars = get_significance_stars(row['p_adjusted_FDR'])
            if stars:
                d_val = row['Cohens_d']
                # Position star at end of bar with small offset
                if d_val > 0:
                    star_x = d_val + 0.08
                else:
                    star_x = d_val - 0.08
                
                ax.text(star_x, idx, stars, 
                       ha='left' if d_val > 0 else 'right',
                       va='center', fontsize=16, fontweight='bold',
                       color='#D32F2F', zorder=4)
        
        # Add reference lines
        ax.axvline(x=0, color='black', linewidth=2)
        ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Large effect')
        ax.axvline(x=-0.8, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.2, alpha=0.5, label='Medium effect')
        ax.axvline(x=-0.5, color='gray', linestyle=':', linewidth=1.2, alpha=0.5)
        
        ax.set_xlabel("Cohen's d (Effect Size)", fontsize=16, fontweight='bold')
        ax.set_title(name, fontsize=18, fontweight='bold')
        ax.tick_params(axis='x', labelsize=13)
        ax.grid(axis='x', alpha=0.3, linewidth=0.8)
        
        # Add legend for first subplot
        if i == 0:
            ax.legend(loc='lower right', fontsize=13)
        
        # Adjust x-axis limits to accommodate all values
        max_d = max(abs(comp_results_ordered['Cohens_d'].min()), 
                    abs(comp_results_ordered['Cohens_d'].max()))
        ax.set_xlim(-max_d * 1.15, max_d * 1.15)
    
    plt.suptitle("Effect Sizes (Cohen's d) for All Pairwise Comparisons", 
                 fontsize=24, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(f'{OUTPUT_DIR}/pairwise_effect_sizes.png', dpi=300, bbox_inches='tight')
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
            
            # Calculate mean +/- SD for each group
            grouped_stats = plot_df.groupby(['Collagen_Status', 'Disease'])[feature].agg(['mean', 'std']).reset_index()
            
            # Plot lines manually with SD error bars in black
            x_positions = {'No Collagen': 0, '+Collagen': 1}
            
            for disease_group in disease_order:
                group_data = grouped_stats[grouped_stats['Disease'] == disease_group]
                
                x_vals = [x_positions[coll] for coll in group_data['Collagen_Status']]
                y_vals = group_data['mean'].values
                y_err = group_data['std'].values
                
                # Plot line and points
                color = palette[disease_group]
                marker = 'o' if disease_group == 'Healthy' else 's'
                ax.plot(x_vals, y_vals, marker=marker, color=color, label=disease_group, 
                       linewidth=2, markersize=8, linestyle='-')
                
                # Add black SD error bars
                ax.errorbar(x_vals, y_vals, yerr=y_err, fmt='none', ecolor='black', 
                           capsize=5, capthick=1.5, elinewidth=1.5)
            
            # Now add significance stars for relevant comparisons
            # We need to position stars next to the comparison group (Group2)
            
            # Extract comparison results for this feature
            for res in all_results:
                if feature in res['Feature'].values:
                    row = res[res['Feature'] == feature].iloc[0]
                    comp_name = res['Comparison'].iloc[0]
                    p_fdr = row['p_adjusted_FDR']
                    group1 = row['Group1']
                    group2 = row['Group2']
                    
                    # Get significance stars
                    stars = get_significance_stars(p_fdr)
                    
                    if stars:  # Only annotate if significant (p < 0.05)
                        # Determine position for the star (next to Group2 data point)
                        # Parse group names to get x position and disease
                        if 'NoCollagen' in group2:
                            x_pos = 0  # No Collagen position
                        elif 'Collagen' in group2:
                            x_pos = 1  # +Collagen position
                        else:
                            continue
                        
                        if 'Healthy' in group2:
                            disease = 'Healthy'
                        elif 'TAA' in group2:
                            disease = 'TAA'
                        else:
                            continue
                        
                        # Get the y-value for this group (raw values)
                        collagen_status = '+Collagen' if x_pos == 1 else 'No Collagen'
                        y_val = grouped_stats[(grouped_stats['Disease'] == disease) & 
                                             (grouped_stats['Collagen_Status'] == collagen_status)]['mean'].values
                        
                        if len(y_val) > 0:
                            y_val = y_val[0]
                            
                            # Add star annotation slightly offset from the point
                            ax.text(x_pos, y_val, f' {stars}', fontsize=14, 
                                   verticalalignment='center', horizontalalignment='left',
                                   color='black', fontweight='bold')
            
            # Add horizontal brackets for Collagen Effect comparisons
            # These show significance between No Collagen and +Collagen within same disease group
            y_max_for_bracket = max(grouped_stats['mean'] + grouped_stats['std']) * 1.02
            bracket_offset = 0
            
            for res in all_results:
                if feature in res['Feature'].values:
                    row = res[res['Feature'] == feature].iloc[0]
                    comp_name = res['Comparison'].iloc[0]
                    p_fdr = row['p_adjusted_FDR']
                    
                    # Get significance stars
                    stars = get_significance_stars(p_fdr)
                    
                    if stars and 'Collagen Effect' in comp_name:
                        # Determine which disease group
                        if 'Healthy' in comp_name:
                            disease = 'Healthy'
                            color_bracket = palette['Healthy']
                        elif 'TAA' in comp_name:
                            disease = 'TAA'
                            color_bracket = palette['TAA']
                        else:
                            continue
                        
                        # Get max y-value for this disease group to position bracket above it
                        group_y_vals = []
                        for coll_status in ['No Collagen', '+Collagen']:
                            vals = grouped_stats[(grouped_stats['Disease'] == disease) & 
                                               (grouped_stats['Collagen_Status'] == coll_status)]
                            if len(vals) > 0:
                                group_y_vals.append(vals['mean'].values[0] + vals['std'].values[0])
                        
                        if len(group_y_vals) > 0:
                            bracket_y = max(group_y_vals) + (max(group_y_vals) * 0.05) + bracket_offset
                            
                            # Draw horizontal bracket
                            x1, x2 = 0, 1  # No Collagen to +Collagen
                            
                            # Draw the bracket line
                            ax.plot([x1, x1, x2, x2], 
                                   [bracket_y, bracket_y + (bracket_y * 0.02), bracket_y + (bracket_y * 0.02), bracket_y],
                                   color=color_bracket, linewidth=1.5)
                            
                            # Add stars in the middle
                            ax.text((x1 + x2) / 2, bracket_y + (bracket_y * 0.03), stars, 
                                   fontsize=12, ha='center', va='bottom', 
                                   color='black', fontweight='bold')
                            
                            bracket_offset += max(group_y_vals) * 0.08  # Offset for multiple brackets
            
            # Set x-axis
            ax.set_xticks([0, 1])
            ax.set_xticklabels(collagen_order)
            
            # Set appropriate y-axis limits based on feature type and actual data range
            all_means = grouped_stats['mean'].values
            all_stds = grouped_stats['std'].values
            data_min = min(all_means - all_stds)
            data_max = max(all_means + all_stds)
            
            # Only use 0-1 scale for ratios that are actually normalized to [0, 1]
            if '_ratio' in feature and data_min >= 0 and data_max <= 1:
                # True normalized ratios: use 0 to 1 scale
                ax.set_ylim(0, 1)
            else:
                # All other features (measurements and non-normalized ratios): use rounded values
                # Round to nice values
                # Start at 0 or a rounded value slightly below minimum
                if data_min >= 0:
                    y_min = 0
                else:
                    # Round down to nearest 10, 100, etc.
                    magnitude = 10 ** np.floor(np.log10(abs(data_min)))
                    y_min = np.floor(data_min / magnitude) * magnitude
                
                # Round up maximum to next round number
                if data_max > 0:
                    magnitude = 10 ** np.floor(np.log10(data_max))
                    y_max = np.ceil(data_max / magnitude) * magnitude
                else:
                    y_max = 0
                
                # Add 10% padding to top
                y_max = y_max * 1.1
                
                ax.set_ylim(y_min, y_max)
            
            # Add title with feature name
            title_text = f"Interaction Effect: {feature}"
            ax.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
            
            ax.set_ylabel(feature, fontsize=12)
            ax.set_xlabel('')
            ax.legend(title='Group', fontsize=11, title_fontsize=12, loc='best')
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

    # 11. Generate boxplot versions for better visual understanding
    print(f"\n{'='*80}")
    print("GENERATING BOXPLOT VERSIONS FOR ALL FEATURES")
    print(f"{'='*80}")
    
    # Create directory for boxplots
    boxplots_dir = os.path.join(OUTPUT_DIR, 'boxplots')
    os.makedirs(boxplots_dir, exist_ok=True)
    
    print(f"Generating boxplots for all {len(feature_cols)} features...")
    
    # Define group order and colors for boxplots
    group_order_box = ['Healthy_NoCollagen', 'Healthy_Collagen', 'TAA_NoCollagen', 'TAA_Collagen']
    group_labels_box = ['Healthy\nNo Coll', 'Healthy\n+Coll', 'TAA\nNo Coll', 'TAA\n+Coll']
    box_palette = {
        'Healthy_NoCollagen': '#2ECC71',
        'Healthy_Collagen': '#85E1B5',
        'TAA_NoCollagen': '#E74C3C',
        'TAA_Collagen': '#F1948A'
    }
    
    for feature in feature_cols:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create boxplot
            positions = [0, 1, 2.5, 3.5]  # Space between disease groups
            bp = ax.boxplot(
                [df[df['Group'] == g][feature].dropna() for g in group_order_box],
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                boxprops=dict(linewidth=1.5),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5)
            )
            
            # Color the boxes
            for patch, group in zip(bp['boxes'], group_order_box):
                patch.set_facecolor(box_palette[group])
                patch.set_alpha(0.7)
            
            # Add mean and SD markers
            for i, group in enumerate(group_order_box):
                data = df[df['Group'] == group][feature].dropna()
                if len(data) > 0:
                    mean_val = data.mean()
                    std_val = data.std()
                    
                    # Plot mean as diamond
                    ax.plot(positions[i], mean_val, marker='D', color='black', 
                           markersize=8, zorder=3)
                    
                    # Plot SD as error bar
                    ax.errorbar(positions[i], mean_val, yerr=std_val, 
                               fmt='none', ecolor='black', capsize=8, 
                               capthick=2, elinewidth=2, zorder=2)
            
            # Add significance annotations
            # We'll add stars for the main comparisons
            
            # Get max y value for positioning stars
            all_data = [df[df['Group'] == g][feature].dropna() for g in group_order_box]
            y_max = max([d.max() for d in all_data if len(d) > 0])
            y_min = min([d.min() for d in all_data if len(d) > 0])
            y_range = y_max - y_min
            
            # Function to add significance bracket
            def add_significance_bracket(x1, x2, y, p_val, ax):
                stars = get_significance_stars(p_val)
                if stars:
                    # Draw bracket
                    bracket_h = y_range * 0.02
                    ax.plot([x1, x1, x2, x2], [y, y + bracket_h, y + bracket_h, y], 
                           'k-', linewidth=1.5)
                    # Add stars
                    ax.text((x1 + x2) / 2, y + bracket_h * 1.5, stars, 
                           ha='center', va='bottom', fontsize=14, fontweight='bold')
                    return True
                return False
            
            # Add significance annotations based on comparisons
            bracket_y = y_max + y_range * 0.05
            bracket_increment = y_range * 0.08
            
            for res in all_results:
                if feature in res['Feature'].values:
                    row = res[res['Feature'] == feature].iloc[0]
                    comp_name = res['Comparison'].iloc[0]
                    p_fdr = row['p_adjusted_FDR']
                    group1 = row['Group1']
                    group2 = row['Group2']
                    
                    # Map groups to positions
                    group_to_pos = {
                        'Healthy_NoCollagen': 0,
                        'Healthy_Collagen': 1,
                        'TAA_NoCollagen': 2.5,
                        'TAA_Collagen': 3.5
                    }
                    
                    # Only show main disease effect comparisons
                    if 'Disease Effect' in comp_name:
                        if group1 in group_to_pos and group2 in group_to_pos:
                            x1 = group_to_pos[group1]
                            x2 = group_to_pos[group2]
                            if add_significance_bracket(x1, x2, bracket_y, p_fdr, ax):
                                bracket_y += bracket_increment
            
            # Set x-axis
            ax.set_xticks(positions)
            ax.set_xticklabels(group_labels_box, fontsize=11)
            
            # Set appropriate y-axis limits based on feature type and actual data range
            # Only use 0-1 scale for ratios that are actually normalized to [0, 1]
            if '_ratio' in feature and y_min >= 0 and y_max <= 1:
                ax.set_ylim(0, 1)
            else:
                # Use same logic as before for measurements
                data_min = y_min
                data_max = bracket_y + bracket_increment  # Include space for brackets
                
                if data_min >= 0:
                    y_min_plot = 0
                else:
                    magnitude = 10 ** np.floor(np.log10(abs(data_min)))
                    y_min_plot = np.floor(data_min / magnitude) * magnitude
                
                if data_max > 0:
                    magnitude = 10 ** np.floor(np.log10(data_max))
                    y_max_plot = np.ceil(data_max / magnitude) * magnitude
                else:
                    y_max_plot = 0
                
                y_max_plot = y_max_plot * 1.05
                ax.set_ylim(y_min_plot, y_max_plot)
            
            # Labels and title
            ax.set_ylabel(feature, fontsize=12, fontweight='bold')
            ax.set_title(f'{feature}', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add legend for mean marker
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='D', color='w', 
                                     markerfacecolor='black', markersize=8, 
                                     label='Mean ± SD')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # Save plot
            fname = f"{feature}_boxplot.png"
            plt.tight_layout()
            plt.savefig(os.path.join(boxplots_dir, fname), dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Could not create boxplot for {feature}: {e}")
            plt.close()
    
    print(f"Saved {len(feature_cols)} boxplots to {boxplots_dir}")


if __name__ == '__main__':
    main()

