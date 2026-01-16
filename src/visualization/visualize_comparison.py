"""
Visualize Skeletonization Comparison (Healthy vs TAA)
=====================================================
Generates a side-by-side comparison of 4 cells:
1. Healthy 1
2. Healthy 2
3. TAA 1
4. TAA 2

Shows overlays to visually value "fragmentation" or "network density".
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from pathlib import Path

# Config
# 2 Healthy (083, 096) vs 2 TAA (024, 043)
FILES = [
    ('Healthy 1', 'experiments/V2/restored_names/Actin/Inference_Raw/01C-0083 -coll 60x vehicleZstack cell3_segmentation.nii.gz'),
    ('Healthy 2', 'experiments/V2/restored_names/Actin/Inference_Raw/01C-0096 +coll 60x vehicleZstack cell1_segmentation.nii.gz'),
    ('TAA 1', 'experiments/V2/restored_names/Actin/Inference_Raw/03Asc-024 +coll 60x veh 48h-Zstack cell1_segmentation.nii.gz'),
    ('TAA 2', 'experiments/V2/restored_names/Actin/Inference_Raw/03Asc-0043 +coll 60x veh 48h-Zstack cell1_segmentation.nii.gz')
]
OUTPUT_IMAGE = 'skeleton_comparison_grid.png'

def process_image(filepath):
    nii = nib.load(filepath)
    mask = nii.get_fdata()
    mask = (mask > 0).astype(int)
    skeleton = skeletonize(mask)
    skeleton = (skeleton > 0).astype(int)
    
    # Max projection along Z (axis 2)
    proj_mask = np.max(mask, axis=2)
    proj_skel = np.max(skeleton, axis=2)
    
    vol = np.sum(mask)
    skel = np.sum(skeleton)
    ratio = skel / vol if vol > 0 else 0
    
    return proj_mask, proj_skel, vol, skel, ratio

def main():
    plt.figure(figsize=(20, 10))
    
    for i, (label, filepath) in enumerate(FILES):
        print(f"Processing {label}...")
        mask, skel, v, s, r = process_image(filepath)
        
        # Plot Overlay
        plt.subplot(2, 4, i+1) # Update layout logic to be 1 row, 4 cols or 2x2. Let's do 1 row of overlays first.
        # Actually let's do: Top Row = Mask, Bottom Row = Overlay with Stats
        
        # Row 1: Mask
        plt.subplot(2, 4, i+1)
        plt.imshow(mask, cmap='gray')
        plt.title(f"{label}\nVol: {v//1000}k", fontsize=12)
        plt.axis('off')
        
        # Row 2: Overlay
        plt.subplot(2, 4, i+5)
        plt.imshow(mask, cmap='gray', alpha=0.5)
        plt.imshow(skel, cmap='hot', alpha=0.8)
        plt.title(f"Skel: {s//1000}k\nRatio: {r:.3f}", fontsize=12, fontweight='bold')
        plt.axis('off')

    plt.suptitle("Healthy vs TAA Actin Network Comparison\nDoes TAA look more fragmented/dense?", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"Saved comparison to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
