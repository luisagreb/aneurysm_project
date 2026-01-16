"""
Visualize Skeletonization
=========================
Generates a max-projection comparison of:
1. Original Actin Mask
2. Skeletonized Network
3. Overlay

Saves the result as a PNG image for visual verification.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from pathlib import Path

# Config
# Picking a representative TAA file (Subject 24)
INPUT_FILE = 'experiments/V2/restored_names/Actin/Inference_Raw/03Asc-024 +coll 60x veh 48h-Zstack cell1_segmentation.nii.gz'
OUTPUT_IMAGE = 'skeleton_verification_TAA.png'

def main():
    print(f"Loading {INPUT_FILE}...")
    nii = nib.load(INPUT_FILE)
    mask = nii.get_fdata()
    mask = (mask > 0).astype(int)
    
    print("Skeletonizing...")
    skeleton = skeletonize(mask)
    skeleton = (skeleton > 0).astype(int)
    
    print(f"Mask Volume (voxels): {np.sum(mask)}")
    print(f"Skeleton Length (voxels): {np.sum(skeleton)}")
    
    # Create Maximum Intensity Projections (collapse Z-axis, usually axis 2)
    # Shape is (X, Y, Z) = (929, 836, 10)
    # We want to see X-Y plane, so we collapse axis 2
    proj_mask = np.max(mask, axis=2)
    proj_skel = np.max(skeleton, axis=2)
    
    # Create visualization
    plt.figure(figsize=(15, 6))
    
    # 1. Original Mask
    plt.subplot(1, 3, 1)
    plt.imshow(proj_mask, cmap='gray')
    plt.title("Original Actin Mask (Max Proj)")
    plt.axis('off')
    
    # 2. Skeleton
    plt.subplot(1, 3, 2)
    plt.imshow(proj_skel, cmap='magma')
    plt.title("Skeletonized Centerline")
    plt.axis('off')
    
    # 3. Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(proj_mask, cmap='gray', alpha=0.5)
    plt.imshow(proj_skel, cmap='hot', alpha=0.7) # Red/Yellow skeleton on top
    plt.title("Overlay (Skeleton in Red/Hot)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"Visualization saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
