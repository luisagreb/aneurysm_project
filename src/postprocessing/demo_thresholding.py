#!/usr/bin/env python3
"""
Demonstrate thresholding limitations for nucleus segmentation.
Compares Otsu/Sauvola thresholding with ground truth (manual/nnUNet) labels.
Generates presentation-ready comparison figures.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu, threshold_yen, threshold_sauvola, gaussian
from skimage.morphology import remove_small_objects, remove_small_holes, ball, binary_opening, binary_closing
from skimage.exposure import rescale_intensity

# Configuration
DATA_DIR = Path('/home/luisa/aneurysm_project/data/nnUNet/nnUNet_raw/Dataset003_Nucleus')
IMAGES_DIR = DATA_DIR / 'imagesTr'
LABELS_DIR = DATA_DIR / 'labelsTr'
OUTPUT_DIR = Path('/home/luisa/aneurysm_project/thresholding_comparison')

# Select 3 diverse samples
SAMPLE_IDS = ['0001', '0010', '0050']


def load_nifti(path):
    """Load NIfTI file and return numpy array."""
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img


def threshold_segmentation(arr):
    """Apply combined thresholding (Otsu + Sauvola) like the original script."""
    # Robust intensity normalization
    lo, hi = np.percentile(arr, (1, 99.9))
    if hi <= lo:
        hi = arr.max() if arr.max() > 0 else 1.0
        lo = arr.min()
    arr_n = rescale_intensity(arr, in_range=(lo, hi), out_range=(0.0, 1.0))
    
    # Light denoise
    arr_s = gaussian(arr_n, sigma=1.0, preserve_range=True)
    
    # Global thresholds
    try:
        t_otsu = threshold_otsu(arr_s)
    except Exception:
        t_otsu = 0.5
    
    t_yen = threshold_yen(arr_s)
    
    # Per-slice Sauvola (adaptive)
    Z = arr_s.shape[0]
    mask_sauvola = np.zeros_like(arr_s, dtype=bool)
    for z in range(Z):
        sl = arr_s[z]
        if sl.max() > 0:
            thr = threshold_sauvola(sl, window_size=31, k=0.2)
            mask_sauvola[z] = sl > thr
    
    # Combine with global threshold
    t_global = min(t for t in [t_otsu, t_yen] if np.isfinite(t))
    mask_global = arr_s > t_global * 0.9
    
    mask = mask_sauvola | mask_global
    
    # 3D morphological cleanup
    try:
        mask = binary_opening(mask, ball(1))
        mask = binary_closing(mask, ball(1))
        mask = remove_small_holes(mask, area_threshold=500)
        mask = remove_small_objects(mask, min_size=800)
    except Exception:
        pass
    
    return mask, arr_n, {
        'otsu': t_otsu,
        'yen': t_yen,
        'method': 'Otsu + Sauvola Adaptive'
    }


def create_comparison_figure(image, gt_mask, thresh_mask, sample_id, thresholds, output_path):
    """Create a comparison figure for a single sample."""
    # Get middle slice
    mid_z = image.shape[0] // 2
    
    img_slice = image[mid_z]
    gt_slice = gt_mask[mid_z] if gt_mask is not None else np.zeros_like(img_slice)
    thresh_slice = thresh_mask[mid_z]
    
    # Calculate metrics
    if gt_mask is not None:
        # Dice score
        intersection = np.sum(gt_slice.astype(bool) & thresh_slice.astype(bool))
        dice = 2 * intersection / (np.sum(gt_slice > 0) + np.sum(thresh_slice > 0) + 1e-8)
        
        # False positives and negatives
        fp = np.sum(thresh_slice.astype(bool) & ~gt_slice.astype(bool))
        fn = np.sum(~thresh_slice.astype(bool) & gt_slice.astype(bool))
    else:
        dice, fp, fn = 0, 0, 0
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Raw image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('Raw Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Thresholding result
    axes[1].imshow(img_slice, cmap='gray', alpha=0.7)
    axes[1].imshow(thresh_slice, cmap='Reds', alpha=0.5)
    axes[1].set_title(f'Thresholding\n(Otsu={thresholds["otsu"]:.3f})', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Ground truth
    if gt_mask is not None:
        axes[2].imshow(img_slice, cmap='gray', alpha=0.7)
        axes[2].imshow(gt_slice, cmap='Greens', alpha=0.5)
        axes[2].set_title('Ground Truth', fontsize=12, fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, 'No GT Available', ha='center', va='center', fontsize=14)
    axes[2].axis('off')
    
    # Error visualization (FP=Red, FN=Blue)
    error_img = np.zeros((*img_slice.shape, 3))
    error_img[..., 0] = thresh_slice.astype(bool) & ~gt_slice.astype(bool)  # FP = Red
    error_img[..., 2] = ~thresh_slice.astype(bool) & gt_slice.astype(bool)  # FN = Blue
    error_img[..., 1] = thresh_slice.astype(bool) & gt_slice.astype(bool)   # TP = Green
    
    axes[3].imshow(error_img)
    axes[3].set_title(f'Errors\nDice: {dice:.3f} | FP: {fp} | FN: {fn}', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.suptitle(f'Nucleus {sample_id} - Thresholding Limitations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return dice


def main():
    print("=== Thresholding vs nnU-Net Comparison ===\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    results = []
    
    for sample_id in SAMPLE_IDS:
        img_path = IMAGES_DIR / f'Nucleus_{sample_id}_0000.nii.gz'
        label_path = LABELS_DIR / f'Nucleus_{sample_id}.nii.gz'
        
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue
        
        print(f"Processing Nucleus_{sample_id}...")
        
        # Load image
        image, _ = load_nifti(img_path)
        print(f"  Image shape: {image.shape}")
        
        # Load ground truth if available
        if label_path.exists():
            gt_mask, _ = load_nifti(label_path)
            print(f"  Ground truth loaded")
        else:
            gt_mask = None
            print(f"  No ground truth available")
        
        # Apply thresholding
        thresh_mask, normalized, thresholds = threshold_segmentation(image)
        print(f"  Thresholds: Otsu={thresholds['otsu']:.4f}, Yen={thresholds['yen']:.4f}")
        print(f"  Threshold mask voxels: {thresh_mask.sum()}")
        
        # Create comparison figure
        output_path = OUTPUT_DIR / f'comparison_{sample_id}.png'
        dice = create_comparison_figure(image, gt_mask, thresh_mask, sample_id, thresholds, output_path)
        print(f"  Dice Score: {dice:.4f}")
        print(f"  Saved: {output_path}\n")
        
        results.append({
            'sample': sample_id,
            'dice': dice,
            'thresh_voxels': int(thresh_mask.sum()),
            'gt_voxels': int(gt_mask.sum()) if gt_mask is not None else 0
        })
    
    # Create summary figure
    if results:
        print("Creating summary comparison...")
        
        # Load all comparison images and combine
        fig, axes = plt.subplots(len(results), 1, figsize=(16, 4 * len(results)))
        if len(results) == 1:
            axes = [axes]
        
        for idx, result in enumerate(results):
            img = plt.imread(OUTPUT_DIR / f'comparison_{result["sample"]}.png')
            axes[idx].imshow(img)
            axes[idx].axis('off')
        
        plt.suptitle('Thresholding Limitations Summary\n(Red=False Positive, Blue=False Negative, Green=True Positive)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'summary.png', dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved summary: {OUTPUT_DIR / 'summary.png'}")
        
        # Print summary table
        print("\n=== Results Summary ===")
        print(f"{'Sample':<10} {'Dice':<10} {'Thresh Vox':<15} {'GT Vox':<15}")
        print("-" * 50)
        for r in results:
            print(f"{r['sample']:<10} {r['dice']:<10.4f} {r['thresh_voxels']:<15} {r['gt_voxels']:<15}")
        
        avg_dice = np.mean([r['dice'] for r in results])
        print("-" * 50)
        print(f"{'Average':<10} {avg_dice:<10.4f}")


if __name__ == '__main__':
    main()
