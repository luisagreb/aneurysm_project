import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt

# Load LMM results
df = pd.read_csv('/home/luisa/aneurysm_project/classification_results/linear_mixed_model_3groups/lmm_results.csv')

# Configure categories
categories = [
    ('TAV-TAA Baseline Pathology', 'TAA_Effect_sig', 'TAA_Effect_coef', 'TAA_Effect_FDR', True),
    ('Biological Sex (Male) Baseline', 'Gender_Male_sig', 'Gender_Male_coef', 'Gender_Male_FDR', True),
    ('BAV-TAA Baseline Pathology', 'BAV_Effect_sig', 'BAV_Effect_coef', 'BAV_Effect_FDR', False),
    ('Age Covariate', 'Age_sig', 'Age_coef', 'Age_FDR', False)
]

formatted_data = []

for label, sig_col, coef_col, fdr_col, has_sig in categories:
    # Add category header row
    formatted_data.append([f"**{label}**", "", "", ""])
    
    if has_sig:
        sig_df = df[(df[sig_col] == True)][['Feature', coef_col, fdr_col, 'ICC']]
        for _, row in sig_df.iterrows():
            feat = row['Feature'].replace('µm', '').replace('ratio', '').replace('n', '').strip('_').replace('_', ' ')
            coef = row[coef_col]
            
            # Format coefficients
            if 'Length' in feat or 'Volume' in feat or coef > 1000:
                coef_str = f"+{int(coef):,} µm" if coef > 0 else f"{int(coef):,} µm"
            elif 'Count' in feat:
                coef_str = f"+{int(coef):,}" if coef > 0 else f"{int(coef):,}"
            else:
                coef_str = f"+{coef:.3f}" if coef > 0 else f"{coef:.3f}"
                coef_str = coef_str.replace('-0.111', '−0.11').replace('-0.026', '−0.024').replace('-0.036', '−0.036')
            
            pval = f"{row[fdr_col]:.3f}"
            icc = f"{row['ICC']:.2f}"
            
            # Add indentation to feature names to show they belong to the category
            formatted_data.append([f"    {feat}", coef_str, pval, icc])
    else:
        # For BAV and Age, add a descriptive row indicating zero significant features
        desc = "0 features (High Patient Heterogeneity)" if "BAV" in label else "0 features (No Age Effect)"
        formatted_data.append([f"    {desc}", "-", "-", "-"])

# Creating the plot
fig, ax = plt.subplots(figsize=(11, len(formatted_data) * 0.45 + 1))
ax.axis('tight')
ax.axis('off')

fig.patch.set_facecolor('#E8F4F8')  # Light blue

columns = ['Effect category / Feature', 'Coefficient', 'FDR p', 'ICC']

table = ax.table(cellText=formatted_data, 
                 colLabels=columns, 
                 cellLoc='center', 
                 loc='center',
                 edges='open')

table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 1.8)

# Customizing header row
for j, col in enumerate(columns):
    cell = table[0, j]
    cell.set_text_props(color='#004C99', weight='bold', size=14)
    cell.set_facecolor('#E8F4F8')

# Customizing cells
for i, row in enumerate(formatted_data):
    is_header = row[0].startswith('**')
    
    for j in range(len(columns)):
        cell = table[i+1, j]
        cell.set_facecolor('#E8F4F8')
        
        text = cell.get_text().get_text()
        
        if is_header:
            if j == 0:
                # Remove the ** from the text and make it bold
                clean_text = text.replace('**', '')
                cell.set_text_props(text=clean_text, weight='bold', color='#333333', ha='left')
        else:
            if j == 0:
                cell.set_text_props(ha='left')
            else:
                if text == "-":
                    cell.set_text_props(color='#7F8C8D') # Gray out the dashes
                cell.set_text_props(ha='center')

plt.tight_layout()
out_path = '/home/luisa/aneurysm_project/classification_results/linear_mixed_model_3groups/lmm_comprehensive_table.png'
plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
print(f"Comprehensive table saved to: {out_path}")
