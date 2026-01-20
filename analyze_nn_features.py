import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'

def extract_numeric_id(text):
    """Extract Subject ID from filename."""
    import re
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
    print(f"Loading metadata from {filepath}...")
    try:
        df = pd.read_excel(filepath, header=None, skiprows=18)
        healthy_col = df[0]
        taa_col = df[1]
        
        healthy_ids = set()
        taa_ids = set()
        
        for x in healthy_col.dropna():
            nid = extract_numeric_id(x)
            if nid is not None:
                healthy_ids.add(nid)
        for x in taa_col.dropna():
            nid = extract_numeric_id(x)
            if nid is not None:
                taa_ids.add(nid)
        
        return healthy_ids, taa_ids
    except Exception as e:
        print(f"Error parse metadata: {e}")
        return set(), set()

print("=" * 70)
print("NEURAL NETWORK FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# Load Metadata
healthy_ids, taa_ids = load_metadata(METADATA_FILE)
print(healthy_ids)
print(taa_ids)

# Load Features
df_features = pd.read_csv(FEATURES_FILE)
print(f"\nFeatures loaded: {df_features.shape}")

# Extract ID and Label
df_features['Numeric_ID'] = df_features['CellName'].apply(extract_numeric_id)
df_features = df_features.dropna(subset=['Numeric_ID'])

def get_label(nid):
    if nid in healthy_ids:
        return 0
    if nid in taa_ids:
        return 1
    return None

df_features['Label'] = df_features['Numeric_ID'].apply(get_label)
df_features = df_features.dropna(subset=['Label'])

# Aggregate by Cell
def normalize_filename(f):
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()

df_features['Cell_ID'] = df_features['CellName'].apply(normalize_filename)

feature_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
if 'Label' in feature_cols:
    feature_cols.remove('Label')
if 'Numeric_ID' in feature_cols:
    feature_cols.remove('Numeric_ID')

agg_dict = {col: 'max' for col in feature_cols}
agg_dict['Label'] = 'first'
agg_dict['Numeric_ID'] = 'first'

df_agg = df_features.groupby('Cell_ID').agg(agg_dict)

# Prepare ML Data
X = df_agg.drop(columns=['Label', 'Numeric_ID'])
y = df_agg['Label']

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"Final X: {X.shape}, y: {y.shape}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Neural Network
print("\nTraining Neural Network...")
nn = MLPClassifier(
    hidden_layer_sizes=(50, 25),
    max_iter=1000,
    random_state=42,
    solver='adam',
    learning_rate_init=0.001,
    alpha=0.1,
    early_stopping=False
)
nn.fit(X_train_scaled, y_train)

from sklearn.metrics import accuracy_score
y_pred = nn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Neural Network Test Accuracy: {accuracy:.1%}")

# Compute Permutation Importance
print("\nComputing permutation importance...")
perm_importance = permutation_importance(
    nn, X_test_scaled, y_test,
    n_repeats=30,
    random_state=42,
    scoring='accuracy'
)

# Create importance dataframe
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance_Mean': perm_importance.importances_mean,
    'Importance_Std': perm_importance.importances_std
})

importance_df = importance_df.sort_values('Importance_Mean', ascending=False)

print("\n" + "="*70)
print("NEURAL NETWORK FEATURE IMPORTANCE (Permutation-based)")
print("="*70)
print("\nTop 15 Features:\n")

for idx, row in importance_df.head(15).iterrows():
    print(f"{row['Feature']:45s} {row['Importance_Mean']:>8.4f} ± {row['Importance_Std']:.4f}")

# Visualize
fig, ax = plt.subplots(figsize=(12, 8))

top_n = 15
top_features = importance_df.head(top_n)

# Color code by structure
colors = []
for feat in top_features['Feature']:
    if 'Actin' in feat:
        colors.append('#E74C3C')
    elif 'Mito' in feat:
        colors.append('#27AE60')
    elif 'Nucleus' in feat:
        colors.append('#3498DB')
    else:
        colors.append('#95A5A6')

bars = ax.barh(range(top_n), top_features['Importance_Mean'].values, 
               xerr=top_features['Importance_Std'].values,
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['Feature'].values, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Permutation Importance (Mean Decrease in Accuracy)', fontsize=12, fontweight='bold')
ax.set_title(f'Neural Network Feature Importance\nTAA vs Healthy Classification (Accuracy: {accuracy:.1%})', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E74C3C', edgecolor='black', label='Actin'),
    Patch(facecolor='#27AE60', edgecolor='black', label='Mitochondria'),
    Patch(facecolor='#3498DB', edgecolor='black', label='Nucleus')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
output_path = 'classification_results/nn_feature_importance.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# Save to CSV
importance_df.to_csv('classification_results/nn_feature_importance.csv', index=False)
print(f"Feature importance data saved to: classification_results/nn_feature_importance.csv")

# ============================================================================
# TORTUOSITY HEATMAP: Visualize branch-level tortuosity (Top Feature)
# ============================================================================
print("\n" + "="*70)
print("VISUALIZING TOP FEATURE: Mito_Mean_Tortuosity_ratio")
print("="*70)

import nibabel as nib
from skimage import morphology
from scipy.ndimage import binary_closing
from pathlib import Path

# Example cells to visualize
cell1_path = Path('/home/luisa/aneurysm_project/experiments/V2/restored_names/Mito/Training_Labels/01ASC-0180 -coll 60x DMSO48h-Zstack cell10.nii.gz')
cell2_path = Path('/home/luisa/aneurysm_project/experiments/V2/restored_names/Mito/Inference_Raw/03Rt-0045 -coll 60x veh 48h-Zstack cell2_segmentation.nii.gz')

def create_tortuosity_heatmap(mask, voxel_size):
    """Create a tortuosity heatmap where each branch is colored by its tortuosity."""
    
    # Skeletonize
    skeleton_image = morphology.skeletonize(mask)
    
    if np.sum(skeleton_image) < 2:
        return None
    
    # Create skeleton object with proper spacing
    from skan import Skeleton, summarize
    skel_obj = Skeleton(skeleton_image, spacing=voxel_size)
    branch_data = summarize(skel_obj)
    
    # Create empty tortuosity map
    tortuosity_map = np.zeros(mask.shape, dtype=float)
    
    # Paint every branch with its specific tortuosity score
    for i in range(branch_data.shape[0]):
        try:
            # Get coordinates of this branch
            coords = skel_obj.path_coordinates(i).astype(int)
            
            # Calculate tortuosity (Path Length / Straight Distance)
            curve = branch_data.loc[i, 'branch-distance']
            straight = branch_data.loc[i, 'euclidean-distance']
            
            # Avoid division by zero
            if straight > 0:
                val = curve / straight
            else:
                val = 1.0
            
            # Paint pixels (clip at 2.0 to prevent extreme loops from washing out image)
            tortuosity_map[coords[:, 0], coords[:, 1], coords[:, 2]] = min(val, 2.0)
            
        except Exception:
            continue
    
    return tortuosity_map

print("\nCreating tortuosity heatmaps for example cells...")

# Load Cell 1 (Healthy)
print("  Loading Cell 1 (Healthy)...")
nii1 = nib.load(cell1_path)
mask1 = (nii1.get_fdata() > 0).astype(np.uint8)
mask1 = binary_closing(mask1, structure=np.ones((3, 3, 3))).astype(np.uint8)
voxel1 = list(nii1.header.get_zooms()[:3])
tort_map1 = create_tortuosity_heatmap(mask1, voxel1)

# Load Cell 2 (TAA)
print("  Loading Cell 2 (TAA)...")
nii2 = nib.load(cell2_path)
mask2 = (nii2.get_fdata() > 0).astype(np.uint8)
mask2 = binary_closing(mask2, structure=np.ones((3, 3, 3))).astype(np.uint8)
voxel2 = list(nii2.header.get_zooms()[:3])
tort_map2 = create_tortuosity_heatmap(mask2, voxel2)

# Create visualization - YZ projection with black background
fig3 = plt.figure(figsize=(16, 7), facecolor='black')
gs = fig3.add_gridspec(1, 3, width_ratios=[1, 1, 0.4], wspace=0.3)

ax1 = fig3.add_subplot(gs[0, 0])
ax2 = fig3.add_subplot(gs[0, 1])
ax3 = fig3.add_subplot(gs[0, 2])

# Cell 1 - YZ projection
proj1_yz = np.max(tort_map1, axis=2)
im1 = ax1.imshow(proj1_yz, cmap='inferno', vmin=1.0, vmax=1.5, interpolation='bilinear')
ax1.set_title('Healthy Cell \nTortuosity Heatmap', fontsize=14, fontweight='bold', color='white')
ax1.axis('off')
ax1.set_facecolor('black')

# Cell 2 - YZ projection
proj2_yz = np.max(tort_map2, axis=2)
im2 = ax2.imshow(proj2_yz, cmap='inferno', vmin=1.0, vmax=1.5, interpolation='bilinear')
ax2.set_title('TAA Cell\nTortuosity Heatmap', fontsize=14, fontweight='bold', color='white')
ax2.axis('off')
ax2.set_facecolor('black')

# Add colorbar
cbar = plt.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.05)
cbar.set_label('Tortuosity Ratio (1.0 = Straight, 1.5+ = Very Curved)', 
               fontsize=12, fontweight='bold', color='white')
cbar.ax.tick_params(labelcolor='white')

# Add interpretation on the right side
ax3.axis('off')
ax3.set_facecolor('black')
interpretation_text = (
    "Color Legend:\n\n"
    "• Purple/Black:\n"
    "  Straight branches\n"
    "  (≈1.0)\n\n"
    "• Orange/Red:\n"
    "  Moderately curved\n"
    "  (1.2-1.5)\n\n"
    "• Yellow/White:\n"
    "  Highly curved\n"
    "  (>1.5)\n\n\n"
    "Interpretation:\n\n"
    "Each pixel shows the\n"
    "tortuosity of the\n"
    "mitochondrial branch\n"
    "passing through it.\n\n"
    "TAA cells show\n"
    "straighter branches\n"
    "despite having more\n"
    "total branches."
)

ax3.text(0.1, 0.5, interpretation_text, 
         fontsize=11, 
         verticalalignment='center',
         color='white',
         bbox=dict(boxstyle='round,pad=1', facecolor='#2C2C2C', edgecolor='white', linewidth=2),
         family='monospace')

plt.suptitle('Mitochondrial Branch Tortuosity Visualization\nHealthy vs TAA (Top Neural Network Feature)', 
             fontsize=15, fontweight='bold', color='white')
plt.tight_layout()

heatmap_path = 'classification_results/tortuosity_heatmap.png'
plt.savefig(heatmap_path, dpi=200, bbox_inches='tight', facecolor='black')
print(f"\nTortuosity heatmap saved to: {heatmap_path}")

print("\nDone!")
