import pandas as pd
import matplotlib.pyplot as plt

# Load LMM results
df = pd.read_csv('/home/luisa/aneurysm_project/classification_results/linear_mixed_model_3groups/lmm_results.csv')

# List of effects and their corresponding significance columns
effects = [
    ('Disease Effect TAV', 'TAA_Effect_sig'),
    ('Disease Effect BAV', 'BAV_Effect_sig'),
    ('Collagen Effect', 'Collagen_sig'),
    ('Interaction TAV × Collagen', 'Interaction_TAA_Coll_sig'),
    ('Interaction BAV × Collagen', 'Interaction_BAV_Coll_sig'),
    ('Age Covariate', 'Age_sig'),
    ('Gender Covariate', 'Gender_Male_sig')
]

data = []
total_features = len(df)

for label, col in effects:
    if col in df.columns:
        count = df[col].sum()
        pct = (count / total_features) * 100
        data.append({
            'Effect_Label': label,
            'Count': count,
            'Percentage': pct
        })

plot_df = pd.DataFrame(data)

# Reverse so the first row plots at the top
plot_df = plot_df.iloc[::-1]

# Set up matplotlib figure matching the clean style
fig, ax = plt.subplots(figsize=(10, 6))

colors = []
for comp in plot_df['Effect_Label']:
    if 'BAV' in comp:
        colors.append('#C39BD3') # Purple
    elif 'TAV' in comp:
        colors.append('#F1948A') # Coral Red
    elif 'Covariate' in comp:
        colors.append('#D5DBDB') # Gray for covariates
    else:
        colors.append('#82E0AA') # Mint Green for general collagen

bars = ax.barh(plot_df['Effect_Label'], plot_df['Percentage'], color=colors, edgecolor='dimgrey', height=0.6, alpha=0.9)

# Formatting
ax.set_xlim(0, 100)
ax.set_xlabel('Significant features (%)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Linear Mixed Model: Summary of Significant Effects (FDR < 0.05)', fontsize=15, fontweight='bold', pad=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=11)

# Annotate values directly on chart
for i, bar in enumerate(bars):
    width = bar.get_width()
    count = plot_df.iloc[i]['Count']
    label_text = f'{int(count)} / {total_features} ({width:.0f}%)'
    
    # Offset text slightly to the right of the bar edge
    ax.text(width + 2, bar.get_y() + bar.get_height() / 2, label_text,
            ha='left', va='center', fontsize=11, fontweight='bold', color='black')

plt.tight_layout()

out_path = '/home/luisa/aneurysm_project/classification_results/linear_mixed_model_3groups/lmm_summary_plot.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"LMM summary plot saved to: {out_path}")

md = '| LMM Effect / Predictor | Significant features / 33 | % |\n|---|---|---|\n'
for row in data:
    md += f"| {row['Effect_Label']} | {int(row['Count'])} | {row['Percentage']:.0f}% |\n"

with open('/home/luisa/.gemini/antigravity/brain/c3e37427-82d7-4bc6-b2fc-542443399ba3/artifacts/lmm_significance_summary.md', 'w') as f:
    f.write(md)
