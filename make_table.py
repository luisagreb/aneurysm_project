import pandas as pd
import os

df = pd.read_csv('/home/luisa/aneurysm_project/classification_results/Healthy_TAV_BAV/pairwise_comparisons/all_pairwise_comparisons.csv')

cols = ['Comparison', 'Feature', 'Cohens_d', 'p_value', 'p_adjusted_FDR', 'Significant']
df_sub = df[cols].copy()
df_sub = df_sub.sort_values(by=['Comparison', 'p_value'])

md = '# All Pairwise Comparisons\n\n'
md += '| Comparison | Feature | Cohen\\'s d | p-value | FDR p-value | Significant |\n'
md += '|---|---|---|---|---|---|\n'

for _, row in df_sub.iterrows():
    md += f'| {row["Comparison"]} | {row["Feature"]} | {row["Cohens_d"]:.3f} | {row["p_value"]:.2e} | {row["p_adjusted_FDR"]:.2e} | {row["Significant"]} |\n'

with open('/home/luisa/.gemini/antigravity/brain/c3e37427-82d7-4bc6-b2fc-542443399ba3/artifacts/pairwise_comparisons_table.md', 'w') as f:
    f.write(md)
    
print("Successfully generated markdown table!")
