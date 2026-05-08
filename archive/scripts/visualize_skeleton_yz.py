#!/usr/bin/env python3
"""
Simple skeleton YZ view comparison
"""
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import morphology
from pathlib import Path
from scipy.ndimage import binary_closing

# File paths
cell1_path = Path('/home/luisa/aneurysm_project/experiments/V2/restored_names/Mito/Training_Labels/01ASC-0180 -coll 60x DMSO48h-Zstack cell10.nii.gz')
cell2_path = Path('/home/luisa/aneurysm_project/experiments/V2/restored_names/Mito/Inference_Raw/03Rt-0045 -coll 60x veh 48h-Zstack cell2_segmentation.nii.gz')

print(f"Loading Cell 1: {cell1_path.name}")
nii1 = nib.load(cell1_path)
mask1 = (nii1.get_fdata() > 0).astype(np.uint8)

print(f"Loading Cell 2: {cell2_path.name}")
nii2 = nib.load(cell2_path)
mask2 = (nii2.get_fdata() > 0).astype(np.uint8)

# Apply binary closing
mask1 = binary_closing(mask1, structure=np.ones((3, 3, 3))).astype(np.uint8)
mask2 = binary_closing(mask2, structure=np.ones((3, 3, 3))).astype(np.uint8)

# Skeletonize
print("Skeletonizing Cell 1...")
skeleton1 = morphology.skeletonize(mask1)

print("Skeletonizing Cell 2...")
skeleton2 = morphology.skeletonize(mask2)

# Create YZ projection (side view)
skel1_yz = np.max(skeleton1, axis=2)
skel2_yz = np.max(skeleton2, axis=2)

# Create simple comparison - just skeleton YZ views
fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')

# Cell 1 - YZ skeleton
axes[0].imshow(skel1_yz, cmap='hot', interpolation='nearest')
axes[0].set_title('Cell 1: 01ASC-0180 -coll cell10\nSkeleton (YZ side view)', 
                  fontsize=14, fontweight='bold', color='white')
axes[0].set_facecolor('black')
axes[0].axis('off')

# Cell 2 - YZ skeleton  
axes[1].imshow(skel2_yz, cmap='hot', interpolation='nearest')
axes[1].set_title('Cell 2: 03Rt-0045 -coll cell2\nSkeleton (YZ side view)', 
                  fontsize=14, fontweight='bold', color='white')
axes[1].set_facecolor('black')
axes[1].axis('off')

plt.tight_layout()

output_path = '/home/luisa/aneurysm_project/outputs/skeleton_yz_comparison.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
print(f"\nYZ skeleton comparison saved to: {output_path}")
plt.close()

print("Done!")
