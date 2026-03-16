import matplotlib.pyplot as plt

# Configuration
formula_text = "Feature ~ Disease * Collagen + Age + Gender + (1|Patient)"
out_path = "/home/luisa/aneurysm_project/classification_results/linear_mixed_model_3groups/lmm_equation.png"

fig = plt.figure(figsize=(10, 1.5))
fig.patch.set_alpha(0.0) # Transparent background

# Add text, centered
plt.text(0.5, 0.5, formula_text, 
         fontsize=28, 
         fontweight='bold', 
         fontfamily='sans-serif',
         ha='center', 
         va='center',
         color='black')

plt.axis('off')
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"Saved to {out_path}")
