"""
Visualize Skeletonization Comparison (No Collagen vs Plus Collagen)
===================================================================
Generates TWO comparision images:
1. comparison_nocoll.png: 2 Healthy vs 2 TAA (all -Collagen)
2. comparison_pluscoll.png: 2 Healthy vs 2 TAA (all +Collagen)
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import os

# Define Datasets
NO_COLL_FILES = [
    ('Healthy -Coll 1', 'experiments/V2/restored_names/Actin/Inference_Raw/01C-0097 -coll 60x VehicleZstack cell1_segmentation.nii.gz'),
    ('Healthy -Coll 2', 'experiments/V2/restored_names/Actin/Inference_Raw/01C-0097 -coll 60x VehicleZstack cell2_segmentation.nii.gz'),
    ('TAA -Coll 1', 'experiments/V2/restored_names/Actin/Inference_Raw/03Asc-0043 -coll 60x veh 48h-Zstack cell1_segmentation.nii.gz'),
    ('TAA -Coll 2', 'experiments/V2/restored_names/Actin/Inference_Raw/03Asc-0043 -coll 60x veh 48h-Zstack cell2_segmentation.nii.gz')
]

PLUS_COLL_FILES = [
    ('Healthy +Coll 1', 'experiments/V2/restored_names/Actin/Inference_Raw/01C-0096 +coll 60x vehicleZstack cell2_segmentation.nii.gz'),
    ('Healthy +Coll 2', 'experiments/V2/restored_names/Actin/Inference_Raw/01C-0096 +coll 60x vehicleZstack cell3_segmentation.nii.gz'),
    ('TAA +Coll 1', 'experiments/V2/restored_names/Actin/Inference_Raw/03Asc-0043 +coll 60x veh 48h-Zstack cell1_segmentation.nii.gz'),
    ('TAA +Coll 2', 'experiments/V2/restored_names/Actin/Inference_Raw/03Asc-0043 +coll 60x veh 48h-Zstack cell2_segmentation.nii.gz')
]

def process_and_plot(file_list, output_filename, title_main):
    plt.figure(figsize=(20, 10))
    
    for i, (label, filepath) in enumerate(file_list):
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            continue
            
        print(f"Processing {label}...")
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
        
        # Row 1: Mask
        plt.subplot(2, 4, i+1)
        plt.imshow(proj_mask, cmap='gray')
        plt.title(f"{label}\nVol: {v//1000}k" if 'v' in locals() else f"{label}\nVol: {vol//1000}k", fontsize=12)
        plt.axis('off')
        
        # Row 2: Overlay
        plt.subplot(2, 4, i+5)
        plt.imshow(proj_mask, cmap='gray', alpha=0.5)
        plt.imshow(proj_skel, cmap='hot', alpha=0.8)
        plt.title(f"Skel: {skel//1000}k\nRatio: {ratio:.3f}", fontsize=12, fontweight='bold')
        plt.axis('off')

    plt.suptitle(title_main, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    print(f"Saved comparison to {output_filename}")

def main():
    print("--- Generating No-Collagen Comparison ---")
    process_and_plot(NO_COLL_FILES, 'comparison_nocoll.png', "Healthy vs TAA (No Collagen)")
    
    print("\n--- Generating Plus-Collagen Comparison ---")
    process_and_plot(PLUS_COLL_FILES, 'comparison_pluscoll.png', "Healthy vs TAA (Plus Collagen)")

if __name__ == "__main__":
    main()
