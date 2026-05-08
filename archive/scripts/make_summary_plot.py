import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

df = pd.read_csv('/home/luisa/aneurysm_project/classification_results/Healthy_TAV_BAV/pairwise_comparisons/all_pairwise_comparisons.csv')

# Only significant features
sig = df[df['Significant'] == True]
total_features = 33

# Custom mappings to keep labels clean and formatted exactly as requested
label_map = {
    'Disease Effect (No Collagen) TAV-TAA': 'Disease effect\n(No Collagen) TAV',
    'Disease Effect (+Collagen) TAV-TAA': 'Disease effect\n(+Collagen) TAV',
    'Disease Effect (No Collagen) BAV-TAA': 'Disease effect\n(No Collagen) BAV',
    'Disease Effect (+Collagen) BAV-TAA': 'Disease effect\n(+Collagen) BAV',
    'Collagen Effect (Healthy)': 'Collagen effect\n(Healthy)',
    'Collagen Effect (TAV_TAA)': 'Collagen effect\n(TAV)',
    'Collagen Effect (BAV_TAA)': 'Collagen effect\n(BAV)',
    'Interaction Healthy_NoColl vs BAV_TAA_Coll': 'Healthy NoColl vs\nBAV +Coll',
    'Interaction Healthy_Coll vs TAV_TAA_NoColl': 'Healthy +Coll vs\nTAV NoColl',
    'Interaction Healthy_Coll vs BAV_TAA_NoColl': 'Healthy +Coll vs\nBAV NoColl'
}

# The specific sequence to iterate
comparisons_to_plot = [
    'Disease Effect (No Collagen) TAV-TAA',
    'Disease Effect (+Collagen) TAV-TAA',
    'Disease Effect (No Collagen) BAV-TAA',
    'Disease Effect (+Collagen) BAV-TAA',
    'Collagen Effect (Healthy)',
    'Collagen Effect (TAV_TAA)',
    'Collagen Effect (BAV_TAA)',
    'Interaction Healthy_NoColl vs BAV_TAA_Coll',
    'Interaction Healthy_Coll vs TAV_TAA_NoColl',
    'Interaction Healthy_Coll vs BAV_TAA_NoColl'
]

# Extract data
data = []
for comp in comparisons_to_plot:
    count = len(sig[sig['Comparison'] == comp])
    pct = (count / total_features) * 100
    display_name = label_map.get(comp, comp)
    data.append({
        'Comparison_Label': display_name,
        'Count': count,
        'Percentage': pct
    })

plot_df = pd.DataFrame(data)

# Reverse so that the first item plots at the top of the horizontal bar chart
plot_df = plot_df.iloc[::-1]

# Set up matplotlib figure matching the clean style
fig, ax = plt.subplots(figsize=(10, 8))

# Define color palette separating TAV vs BAV vs Healthy comparisons
colors = []
for comp in plot_df['Comparison_Label']:
    if 'BAV' in comp:
        colors.append('#C39BD3') # Purple
    elif 'TAV' in comp:
        colors.append('#F1948A') # Coral Red
    else:
        colors.append('#82E0AA') # Mint Green

bars = ax.barh(plot_df['Comparison_Label'], plot_df['Percentage'], color=colors, edgecolor='dimgrey', height=0.6, alpha=0.9)

# Formatting
ax.set_xlim(0, 100)
ax.set_xlabel('Significant features / 33 (%)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Summary of Significance Across Groups', fontsize=18, fontweight='bold', pad=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=11)

# Annotate values directly on chart
for i, bar in enumerate(bars):
    width = bar.get_width()
    count = plot_df.iloc[i]['Count']
    label_text = f'{count}  ({width:.0f}%)'
    
    # Offset text slightly to the right of the bar edge
    ax.text(width + 2, bar.get_y() + bar.get_height() / 2, label_text,
            ha='left', va='center', fontsize=12, fontweight='bold', color='black')

plt.tight_layout()

out_path = '/home/luisa/aneurysm_project/classification_results/Healthy_TAV_BAV/pairwise_comparisons/significance_summary_plot.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Summary plot saved to: {out_path}")

# Also output the markdown table for quick reading
md = '| Comparison | Significant features / 33 | % |\n|---|---|---|\n'
for row in data:
    clean_name = row['Comparison_Label'].replace('\n', ' ')
    md += f"| {clean_name} | {row['Count']} | {row['Percentage']:.0f}% |\n"

with open('/home/luisa/.gemini/antigravity/brain/c3e37427-82d7-4bc6-b2fc-542443399ba3/artifacts/significance_summary.md', 'w') as f:
    f.write(md)
