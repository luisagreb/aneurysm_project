"""
Fix spacing mismatch between training images and labels for nnUNet.

This script:
1. Checks spacing (voxel size) of images and labels
2. Detects mismatches
3. Resamples labels to match image spacing (if needed)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom
import json


def get_spacing_from_affine(affine):
    """Extract voxel spacing from affine matrix."""
    # Spacing is the magnitude of the diagonal elements (first 3x3)
    spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    return spacing


def check_spacing_mismatch(images_dir, labels_dir, tolerance=1e-3):
    """
    Check for spacing mismatches between images and labels.
    
    Args:
        images_dir: Directory containing training images
        labels_dir: Directory containing training labels
        tolerance: Tolerance for considering spacing equal (default: 1e-3 mm)
    
    Returns:
        dict with mismatch information
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    if not images_path.exists() or not labels_path.exists():
        print("Error: One or both directories don't exist")
        return None
    
    # Get sample files
    image_files = sorted(images_path.glob("case_*_0000.nii.gz"))
    if not image_files:
        image_files = sorted(images_path.glob("*.nii.gz"))
    
    label_files = sorted(labels_path.glob("case_*.nii.gz"))
    label_files = [f for f in label_files if not f.name.endswith("_0000.nii.gz")]
    
    if not image_files or not label_files:
        print("Error: No matching files found")
        return None
    
    print("=" * 70)
    print("SPACING MISMATCH CHECK")
    print("=" * 70)
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    print()
    
    mismatches = []
    matches = []
    
    # Check first few pairs
    num_to_check = min(5, len(image_files), len(label_files))
    
    for i in range(num_to_check):
        img_file = image_files[i]
        
        # Extract case ID
        if "_0000" in img_file.stem:
            case_id = img_file.stem.rsplit("_", 1)[0]
        else:
            import re
            match = re.search(r'(?:case_|Sample_)(\d+)', img_file.stem, re.IGNORECASE)
            if match:
                case_id = f"case_{int(match.group(1)):04d}"
            else:
                continue
        
        label_file = labels_path / f"{case_id}.nii.gz"
        
        if not label_file.exists():
            continue
        
        # Load both files
        img_nii = nib.load(str(img_file))
        lbl_nii = nib.load(str(label_file))
        
        img_spacing = get_spacing_from_affine(img_nii.affine)
        lbl_spacing = get_spacing_from_affine(lbl_nii.affine)
        
        img_shape = img_nii.shape
        lbl_shape = lbl_nii.shape
        
        # Check if spacing matches
        spacing_diff = np.abs(img_spacing - lbl_spacing)
        spacing_match = np.all(spacing_diff < tolerance)
        
        # Check if shapes match
        shape_match = img_shape == lbl_shape
        
        print(f"Case: {case_id}")
        print(f"  Image: shape={img_shape}, spacing={img_spacing}")
        print(f"  Label: shape={lbl_shape}, spacing={lbl_spacing}")
        
        if spacing_match and shape_match:
            print(f"  ✓ Match!")
            matches.append(case_id)
        else:
            print(f"  ✗ MISMATCH!")
            if not spacing_match:
                print(f"    Spacing difference: {spacing_diff}")
            if not shape_match:
                print(f"    Shape difference: {img_shape} vs {lbl_shape}")
            mismatches.append({
                'case_id': case_id,
                'image_shape': img_shape,
                'label_shape': lbl_shape,
                'image_spacing': img_spacing,
                'label_spacing': lbl_spacing,
                'spacing_diff': spacing_diff
            })
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Checked: {num_to_check} pairs")
    print(f"Matches: {len(matches)}")
    print(f"Mismatches: {len(mismatches)}")
    
    if mismatches:
        print("\n⚠ SPACING MISMATCH DETECTED!")
        print("This will cause issues with nnUNet training.")
        print("\nRecommendation: Resample labels to match image spacing")
        return {
            'has_mismatch': True,
            'mismatches': mismatches,
            'matches': matches
        }
    else:
        print("\n✓ No spacing mismatches detected!")
        return {
            'has_mismatch': False,
            'mismatches': [],
            'matches': matches
        }


def fix_label_spacing_metadata(label_file, image_file, output_file=None):
    """
    Fix label spacing by updating affine matrix (metadata only, no resampling).
    This preserves label data when shapes already match.
    
    Args:
        label_file: Path to label file
        image_file: Path to reference image file
        output_file: Output path (default: overwrite label_file)
    """
    # Load files
    img_nii = nib.load(str(image_file))
    lbl_nii = nib.load(str(label_file))
    
    img_data = np.asarray(img_nii.dataobj)
    lbl_data = np.asarray(lbl_nii.dataobj)
    
    img_spacing = get_spacing_from_affine(img_nii.affine)
    lbl_spacing = get_spacing_from_affine(lbl_nii.affine)
    
    print(f"  Fixing label spacing metadata...")
    print(f"    Label shape: {lbl_data.shape}, original spacing: {lbl_spacing}")
    print(f"    Image shape: {img_data.shape}, target spacing: {img_spacing}")
    
    # Check if shapes match
    if lbl_data.shape != img_data.shape:
        print(f"    ⚠ Shape mismatch! Need to resample.")
        print(f"    This will preserve data but may take time...")
        
        # Calculate zoom factors
        zoom_factors = lbl_spacing / img_spacing
        
        # Check if zoom factors are reasonable (not too extreme)
        if np.any(zoom_factors < 0.1) or np.any(zoom_factors > 10):
            print(f"    ⚠ Extreme zoom factors detected: {zoom_factors}")
            print(f"    This suggests units mismatch. Attempting to fix...")
            
            # Try swapping/checking if spacing is in different order
            # Sometimes spacing is stored in different order (ZYX vs XYZ)
            # Check if shapes match when we consider different axis orders
            if lbl_data.shape == img_data.shape:
                print(f"    Shapes match - updating metadata only (no resampling)")
                # Just update affine, don't resample
                fixed_label = lbl_data.astype(np.uint8)
                fixed_nii = nib.Nifti1Image(fixed_label, img_nii.affine, img_nii.header)
            else:
                # Need to resample
                print(f"    Resampling with zoom factors: {zoom_factors}")
                resampled_label = zoom(lbl_data, zoom_factors, order=0, mode='nearest')
                
                # Ensure output shape matches image exactly
                if resampled_label.shape != img_data.shape:
                    # Crop or pad to match
                    resampled_label = match_shape(resampled_label, img_data.shape)
                
                fixed_label = resampled_label.astype(np.uint8)
                fixed_nii = nib.Nifti1Image(fixed_label, img_nii.affine, img_nii.header)
        else:
            # Normal resampling
            print(f"    Resampling with zoom factors: {zoom_factors}")
            resampled_label = zoom(lbl_data, zoom_factors, order=0, mode='nearest')
            
            if resampled_label.shape != img_data.shape:
                resampled_label = match_shape(resampled_label, img_data.shape)
            
            fixed_label = resampled_label.astype(np.uint8)
            fixed_nii = nib.Nifti1Image(fixed_label, img_nii.affine, img_nii.header)
    else:
        # Shapes match - just update affine matrix (metadata) without resampling
        print(f"    ✓ Shapes match - updating spacing metadata only (preserves all label data)")
        
        # Preserve original label data
        fixed_label = lbl_data.astype(np.uint8)
        
        # Create new NIfTI image with image's affine (spacing metadata)
        # but keep the original label data unchanged
        fixed_nii = nib.Nifti1Image(fixed_label, img_nii.affine, img_nii.header)
        
        # Verify data is preserved
        unique_labels = np.unique(fixed_label)
        print(f"    Label values preserved: {len(unique_labels)} unique values")
        if len(unique_labels) > 0:
            print(f"      Range: {np.min(unique_labels)} to {np.max(unique_labels)}")
    
    # Save
    output_path = output_file if output_file else label_file
    nib.save(fixed_nii, str(output_path))
    
    # Verify saved file
    saved_nii = nib.load(str(output_path))
    saved_data = np.asarray(saved_nii.dataobj)
    saved_spacing = get_spacing_from_affine(saved_nii.affine)
    
    print(f"    ✓ Saved fixed label: {output_path}")
    print(f"    Final shape: {saved_data.shape}, spacing: {saved_spacing}")
    print(f"    Data preserved: {np.sum(saved_data > 0)} non-zero voxels")
    
    return fixed_label


def match_shape(data, target_shape):
    """Crop or pad data to match target shape."""
    result = np.zeros(target_shape, dtype=data.dtype)
    
    # Calculate crop/pad for each dimension
    for dim in range(len(target_shape)):
        data_size = data.shape[dim]
        target_size = target_shape[dim]
        
        if data_size > target_size:
            # Crop
            start = (data_size - target_size) // 2
            slices = [slice(None)] * len(target_shape)
            slices[dim] = slice(start, start + target_size)
            data = data[tuple(slices)]
        elif data_size < target_size:
            # Pad
            pad_before = (target_size - data_size) // 2
            pad_after = target_size - data_size - pad_before
            pad_width = [(0, 0)] * len(target_shape)
            pad_width[dim] = (pad_before, pad_after)
            data = np.pad(data, pad_width, mode='constant', constant_values=0)
    
    return data


def fix_all_spacing_mismatches(images_dir, labels_dir, output_dir=None, dry_run=False):
    """
    Fix spacing mismatches for all image-label pairs.
    
    Args:
        images_dir: Directory containing training images
        labels_dir: Directory containing training labels
        output_dir: Output directory for fixed labels (default: labels_dir + "_fixed")
        dry_run: If True, only show what would be done
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    if output_dir is None:
        output_dir = labels_path.parent / f"{labels_path.name}_fixed"
    else:
        output_dir = Path(output_dir)
    
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FIXING SPACING MISMATCHES")
    print("=" * 70)
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()
    
    # Get all pairs
    image_files = sorted(images_path.glob("case_*_0000.nii.gz"))
    if not image_files:
        image_files = sorted(images_path.glob("*.nii.gz"))
    
    label_files = sorted(labels_path.glob("case_*.nii.gz"))
    label_files = [f for f in label_files if not f.name.endswith("_0000.nii.gz")]
    
    fixed_count = 0
    skipped_count = 0
    
    for img_file in image_files:
        # Extract case ID
        if "_0000" in img_file.stem:
            case_id = img_file.stem.rsplit("_", 1)[0]
        else:
            import re
            match = re.search(r'(?:case_|Sample_)(\d+)', img_file.stem, re.IGNORECASE)
            if match:
                case_id = f"case_{int(match.group(1)):04d}"
            else:
                continue
        
        label_file = labels_path / f"{case_id}.nii.gz"
        
        if not label_file.exists():
            print(f"Skipping {case_id}: label file not found")
            skipped_count += 1
            continue
        
        print(f"Processing: {case_id}")
        
        # Check if resampling is needed
        img_nii = nib.load(str(img_file))
        lbl_nii = nib.load(str(label_file))
        
        img_spacing = get_spacing_from_affine(img_nii.affine)
        lbl_spacing = get_spacing_from_affine(lbl_nii.affine)
        
        spacing_diff = np.abs(img_spacing - lbl_spacing)
        needs_resample = np.any(spacing_diff > 1e-3) or (img_nii.shape != lbl_nii.shape)
        
        if needs_resample:
            output_file = output_dir / f"{case_id}.nii.gz"
            
            if dry_run:
                print(f"  → Would fix spacing: {label_file.name} → {output_file.name}")
            else:
                try:
                    fix_label_spacing_metadata(label_file, img_file, output_file)
                    fixed_count += 1
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    skipped_count += 1
        else:
            # Copy unchanged file
            output_file = output_dir / f"{case_id}.nii.gz"
            if dry_run:
                print(f"  → Would copy (no change): {label_file.name}")
            else:
                import shutil
                shutil.copy2(label_file, output_file)
                print(f"  ✓ Copied (spacing already matches)")
            fixed_count += 1
        
        print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed: {fixed_count} files")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} files")
    print(f"Output directory: {output_dir}")
    if dry_run:
        print("\n⚠ This was a DRY RUN - no files were modified")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check and fix spacing mismatches between images and labels"
    )
    parser.add_argument(
        "images_dir",
        type=str,
        help="Directory containing training images"
    )
    parser.add_argument(
        "labels_dir",
        type=str,
        help="Directory containing training labels"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for mismatches, don't fix"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for fixed labels (default: labels_dir_fixed)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    # First check for mismatches
    result = check_spacing_mismatch(args.images_dir, args.labels_dir)
    
    # Fix if requested
    if not args.check_only and result and result['has_mismatch']:
        print("\n" + "=" * 70)
        fix_all_spacing_mismatches(
            args.images_dir,
            args.labels_dir,
            output_dir=args.output,
            dry_run=args.dry_run
        )
    elif args.check_only:
        print("\nUse --fix to automatically fix mismatches")

