import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
import os

OUTPUT_DIR = '/home/luisa/aneurysm_project/classification_results/presentation_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data and metrics
df = pd.read_csv('/home/luisa/aneurysm_project/outputs/Advanced_Features_Raw_Final.csv')
metrics = pd.read_csv('/home/luisa/aneurysm_project/classification_results/comparison_3groups/all_tasks_metrics.csv')

def get_color(feature):
    if feature.startswith('Actin'): return '#E99695' # Light Red
    if feature.startswith('Mito'): return '#8FBC8F'  # Light Green
    if feature.startswith('Nucleus'): return '#8CB4E2' # Light Blue
    return '#808080' # Gray

def normalize_filename(f):
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()

df['Cell_ID'] = df['CellName'].apply(normalize_filename)
df = df.dropna(subset=['Disease', 'Collagen_Status'])

feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['Numeric_ID', 'Age', 'FourClass']]
agg_dict = {col: 'max' for col in feature_cols}
agg_dict.update({'Disease': 'first', 'Collagen_Status': 'first'})
df_agg = df.groupby('Cell_ID').agg(agg_dict)

def create_task_plot(task_name, g1_disease, g2_disease):
    print(f"Creating plots for {task_name}")
    
    task_metrics = metrics[metrics['Task'] == task_name].copy()
    if task_metrics.empty:
        return
        
    task_metrics = task_metrics.sort_values('Accuracy', ascending=False)
    
    # 1. Plot the Table (Left Panel)
    fig_table, ax_table = plt.subplots(figsize=(8, 4))
    ax_table.axis('off')
    
    table_data = []
    for _, row in task_metrics.iterrows():
        table_data.append([
            row['Classifier'],
            f"{row['Accuracy']:.1%}",
            f"{row['F1_Score']:.1%}",
            f"{row['AUC_ROC']:.1%}"
        ])
    
    col_labels = ["Classifier", "Accuracy", "F1 Score", "AUC-ROC"]
    table = ax_table.table(
        cellText=table_data, 
        colLabels=col_labels, 
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.0)
    
    # Format table exactly like the picture
    for j, _ in enumerate(col_labels):
        cell = table[0, j]
        cell.set_facecolor('#F6B2B2') # Custom coral pink
        cell.set_text_props(weight='bold')
        
    for i in range(1, len(table_data)+1):
        cell = table[i, 0] # First column
        cell.set_facecolor('#F6B2B2')
        cell.set_text_props(weight='bold')
        
    safe_name = task_name.replace(' ', '_').replace(':', '').replace('/', '-')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{safe_name}_table.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Get Features (ANOVA F-Value Score)
    mask1 = (df_agg['Disease'] == g1_disease) & (df_agg['Collagen_Status'] == 'NoCollagen')
    mask2 = (df_agg['Disease'] == g2_disease) & (df_agg['Collagen_Status'] == 'NoCollagen')
    mask = mask1 | mask2
    
    X = df_agg.loc[mask, feature_cols].fillna(0)
    y = np.where(df_agg.loc[mask, 'Disease'] == g2_disease, 1, 0)
    
    f_val, p_val = f_classif(X, y)
    
    f_df = pd.DataFrame({
        'Feature': feature_cols,
        'F_Value': f_val
    }).sort_values('F_Value', ascending=True).tail(15)
    
    # 3. Plot Feature Importance Bar Chart
    fig_feat, ax_feat = plt.subplots(figsize=(10, 8))
    
    colors = [get_color(f) for f in f_df['Feature']]
    
    bars = ax_feat.barh(f_df['Feature'], f_df['F_Value'], color=colors, edgecolor='gray', linewidth=0.5)
    
    ax_feat.set_xlabel('ANOVA F-Value Score', fontsize=12, fontweight='bold')
    ax_feat.set_title(f"Top 15 Features: {g1_disease} vs {g2_disease}", fontsize=14, fontweight='bold')
    
    # Style to match the reference image
    ax_feat.spines['top'].set_visible(False)
    ax_feat.spines['right'].set_visible(False)
    ax_feat.tick_params(axis='y', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{safe_name}_features.png', bbox_inches='tight', dpi=300)
    plt.close()

create_task_plot('Diseased Task: TAA vs Healthy (Baseline)', 'Healthy', 'TAA')
create_task_plot('Diseased Task: BAV vs Healthy (Baseline)', 'Healthy', 'BAV')
create_task_plot('Diseased Task: TAA vs BAV (Baseline)', 'BAV', 'TAA')


def get_significance_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''

def create_interaction_plots_for_all_features():
    plots_dir = os.path.join(OUTPUT_DIR, 'interaction_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Generating interaction plots for all {len(feature_cols)} features...")
    
    # Load significance stats from pairwise comparison
    stats_file = '/home/luisa/aneurysm_project/classification_results/Healthy_TAV_BAV/pairwise_comparisons/all_pairwise_comparisons.csv'
    all_results = pd.read_csv(stats_file) if os.path.exists(stats_file) else pd.DataFrame()
    
    plot_df = df.copy()
    plot_df['Disease'] = plot_df['Disease'].replace({'TAA': 'TAV_TAA', 'BAV': 'BAV_TAA'})
    plot_df['Collagen_Status'] = plot_df['Collagen_Status'].replace({'Collagen': '+Collagen', 'NoCollagen': 'No Collagen'})
    
    collagen_order = ['No Collagen', '+Collagen']  
    disease_order = ['Healthy', 'TAV_TAA', 'BAV_TAA']
    palette = {'Healthy': '#2ECC71', 'TAV_TAA': '#E74C3C', 'BAV_TAA': '#9B59B6'}
    
    for feature in feature_cols:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            grouped_stats = plot_df.groupby(['Collagen_Status', 'Disease'])[feature].agg(['mean', 'std']).reset_index()
            x_positions = {'No Collagen': 0, '+Collagen': 1}
            
            for disease_group in disease_order:
                group_data = grouped_stats[grouped_stats['Disease'] == disease_group]
                if group_data.empty: continue
                
                x_vals = [x_positions[coll] for coll in group_data['Collagen_Status']]
                y_vals = group_data['mean'].values
                y_err = group_data['std'].values
                
                color = palette.get(disease_group, '#000000')
                marker = 'o' if disease_group == 'Healthy' else ('s' if disease_group == 'TAV_TAA' else '^')
                
                ax.plot(x_vals, y_vals, marker=marker, color=color, label=disease_group.replace('_', '-'), 
                       linewidth=2, markersize=8, linestyle='-')
                ax.errorbar(x_vals, y_vals, yerr=y_err, fmt='none', ecolor='black', capsize=5, capthick=1.5, elinewidth=1.5)
            
            if not all_results.empty and feature in all_results['Feature'].values:
                f_results = all_results[all_results['Feature'] == feature]
                y_max_for_bracket = max(grouped_stats['mean'] + grouped_stats['std']) * 1.02
                bracket_offset = 0
                
                for _, row in f_results.iterrows():
                    p_fdr = row['p_adjusted_FDR']
                    comp = row['Comparison']
                    group2 = row['Group2']
                    stars = get_significance_stars(p_fdr)
                    
                    if not stars: continue
                    
                    # Annotate disease effect (point annotations)
                    if 'Disease Effect' in comp or 'Interaction' in comp:
                        x_pos = 0 if 'NoCollagen' in group2 or 'NoColl' in group2 else 1
                        dis = 'Healthy' if 'Healthy' in group2 else ('TAV_TAA' if 'TAV_TAA' in group2 else ('BAV_TAA' if 'BAV_TAA' in group2 else None))
                        if dis:
                            coll_stat = '+Collagen' if x_pos == 1 else 'No Collagen'
                            y_v = grouped_stats[(grouped_stats['Disease'] == dis) & (grouped_stats['Collagen_Status'] == coll_stat)]['mean'].values
                            if len(y_v) > 0:
                                ax.text(x_pos, y_v[0], f' {stars}', fontsize=14, verticalalignment='center', color='black', fontweight='bold')
                                
                    # Annotate collagen effect (brackets)
                    if 'Collagen Effect' in comp:
                        dis = 'Healthy' if 'Healthy' in comp else ('TAV_TAA' if 'TAV_TAA' in comp else ('BAV_TAA' if 'BAV_TAA' in comp else None))
                        if dis:
                            color_bracket = palette.get(dis, 'black')
                            g_vals = []
                            for c_stat in ['No Collagen', '+Collagen']:
                                v = grouped_stats[(grouped_stats['Disease'] == dis) & (grouped_stats['Collagen_Status'] == c_stat)]
                                if not v.empty: g_vals.append((v['mean'].values[0] + v['std'].values[0]))
                            
                            if len(g_vals) == 2:
                                bracket_y = max(g_vals) + (max(g_vals) * 0.05) + bracket_offset
                                ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + (bracket_y * 0.02), bracket_y + (bracket_y * 0.02), bracket_y], color=color_bracket, linewidth=1.5)
                                ax.text(0.5, bracket_y + (bracket_y * 0.03), stars, fontsize=12, ha='center', va='bottom', color='black', fontweight='bold')
                                bracket_offset += max(g_vals) * 0.08
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(collagen_order)
            
            all_means = grouped_stats['mean'].values
            all_stds = grouped_stats['std'].values
            data_min = min(all_means - all_stds) if len(all_means)>0 else 0
            data_max = max(all_means + all_stds) if len(all_means)>0 else 1
            if '_ratio' in feature and data_min >= 0 and data_max <= 1: ax.set_ylim(0, 1)
            else:
                y_min = 0 if data_min >= 0 else np.floor(data_min / (10 ** np.floor(np.log10(abs(data_min))))) * (10 ** np.floor(np.log10(abs(data_min))))
                y_max = np.ceil(data_max / (10 ** np.floor(np.log10(data_max)))) * (10 ** np.floor(np.log10(data_max))) if data_max > 0 else 0
                ax.set_ylim(y_min, (y_max * 1.1) if y_max != 0 else 1.1)
            
            ax.set_title(f"Interaction Effect: {feature.replace('_', ' ')}", fontsize=14, fontweight='bold', pad=15)
            ax.set_ylabel(feature, fontsize=12)
            ax.set_xlabel('')
            ax.legend(title='Group', fontsize=11, title_fontsize=12, loc='best')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{feature}_interaction.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Could not plot {feature}: {e}")
            plt.close()

create_interaction_plots_for_all_features()
print(f"Generated presentation plots (and interaction plots) in {OUTPUT_DIR}")
