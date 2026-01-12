#!/usr/bin/env python3
"""
Generate 3D thresholded masks for Nucleus, Actin, and Mitochondria.
Outputs NIfTI files for viewing in 3D Slicer.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.filters import threshold_otsu, threshold_yen, threshold_sauvola, gaussian
from skimage.morphology import remove_small_objects, remove_small_holes, ball, binary_opening, binary_closing
from skimage.exposure import rescale_intensity

# Configuration
NNUNET_RAW = Path('/home/luisa/aneurysm_project/data/nnUNet/nnUNet_raw')
OUTPUT_DIR = Path('/home/luisa/aneurysm_project/thresholding_comparison')

SAMPLES = {
    'Nucleus': {
        'dataset': 'Dataset003_Nucleus',
        'image': 'Nucleus_0001_0000.nii.gz',
        'label': 'Nucleus_0001.nii.gz'
    },
    'Actin': {
        'dataset': 'Dataset001_Actin',
        'image': 'Actin_0001_0000.nii.gz',
        'label': 'Actin_0001.nii.gz'
    },
    'Mitochondria': {
        'dataset': 'Dataset002_Mito',
        'image': 'Mito_0001_0000.nii.gz',
        'label': 'Mito_0001.nii.gz'
    }
}


def load_nifti(path):
    """Load NIfTI file and return numpy array and image object."""
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img


def threshold_segmentation(arr, structure_type='Nucleus'):
    """Apply combined thresholding (Otsu + Sauvola)."""
    print(f"    Input shape: {arr.shape}, range: [{arr.min():.2f}, {arr.max():.2f}]")
    
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
    
    try:
        t_yen = threshold_yen(arr_s)
    except Exception:
        t_yen = 0.5
    
    print(f"    Thresholds: Otsu={t_otsu:.4f}, Yen={t_yen:.4f}")
    
    # Determine Z axis (usually the smallest dimension for microscopy)
    z_axis = np.argmin(arr_s.shape)
    Z = arr_s.shape[z_axis]
    
    # Per-slice Sauvola (adaptive)
    mask_sauvola = np.zeros_like(arr_s, dtype=bool)
    for z in range(Z):
        if z_axis == 0:
            sl = arr_s[z]
        elif z_axis == 1:
            sl = arr_s[:, z, :]
        else:
            sl = arr_s[:, :, z]
            
        if sl.max() > 0:
            try:
                thr = threshold_sauvola(sl, window_size=31, k=0.2)
                sl_mask = sl > thr
            except Exception:
                sl_mask = sl > t_otsu
            
            if z_axis == 0:
                mask_sauvola[z] = sl_mask
            elif z_axis == 1:
                mask_sauvola[:, z, :] = sl_mask
            else:
                mask_sauvola[:, :, z] = sl_mask
    
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
    except Exception as e:
        print(f"    Warning: Morphology cleanup failed: {e}")
    
    return mask


def main():
    print("=== Generating 3D Thresholded Masks ===\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    for structure, info in SAMPLES.items():
        print(f"Processing {structure}...")
        
        img_path = NNUNET_RAW / info['dataset'] / 'imagesTr' / info['image']
        label_path = NNUNET_RAW / info['dataset'] / 'labelsTr' / info['label']
        
        if not img_path.exists():
            print(f"  Image not found: {img_path}")
            continue
        
        # Load image
        image, img_obj = load_nifti(img_path)
        
        # Apply thresholding
        thresh_mask = threshold_segmentation(image, structure)
        
        # Save thresholded mask
        mask_path = OUTPUT_DIR / f'{structure.lower()}_threshold_mask.nii.gz'
        mask_uint8 = thresh_mask.astype(np.uint8)
        nib.save(nib.Nifti1Image(mask_uint8, img_obj.affine, img_obj.header), str(mask_path))
        print(f"  Saved threshold mask: {mask_path}")
        print(f"  Mask voxels: {thresh_mask.sum():,}")
        
        # Copy original image for reference
        orig_path = OUTPUT_DIR / f'{structure.lower()}_original.nii.gz'
        nib.save(nib.Nifti1Image(image, img_obj.affine, img_obj.header), str(orig_path))
        print(f"  Saved original image: {orig_path}")
        
        # Copy ground truth label if available
        if label_path.exists():
            gt_mask, gt_obj = load_nifti(label_path)
            gt_path = OUTPUT_DIR / f'{structure.lower()}_ground_truth.nii.gz'
            nib.save(nib.Nifti1Image(gt_mask.astype(np.uint8), gt_obj.affine, gt_obj.header), str(gt_path))
            print(f"  Saved ground truth: {gt_path}")
            print(f"  GT voxels: {(gt_mask > 0).sum():,}")
        
        print()
    
    print("=== Done ===")
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print("\nTo compare in 3D Slicer:")
    print("  1. Open the *_original.nii.gz as the base image")
    print("  2. Load *_threshold_mask.nii.gz as a segmentation overlay")
    print("  3. Load *_ground_truth.nii.gz as another overlay")
    print("  4. Compare the massive over-segmentation from thresholding!")


if __name__ == '__main__':
    main()
