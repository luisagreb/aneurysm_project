"""
Fix orientation metadata (direction matrix) for actin labels to match their images.

This ensures labels and images have the same coordinate system orientation.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd


def fix_label_orientation_to_match_image(label_file, image_file, output_file=None):
    """
    Fix label orientation by copying the direction matrix from the image.
    This preserves label data but updates orientation metadata.
    
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
    
    print(f"  Fixing orientation...")
    print(f"    Label shape: {lbl_data.shape}")
    print(f"    Image shape: {img_data.shape}")
    
    # Check shapes match
    if lbl_data.shape != img_data.shape:
        print(f"    ⚠ Warning: Shapes don't match! Label: {lbl_data.shape}, Image: {img_data.shape}")
        print(f"    This might cause issues. Consider fixing spacing first.")
    
    # Preserve label data
    fixed_label = lbl_data.astype(np.uint8)
    
    # Copy affine matrix from image (this includes direction/orientation)
    # This ensures labels and images have the same coordinate system
    fixed_affine = img_nii.affine.copy()
    
    # Create new NIfTI image with image's affine (orientation + spacing)
    # but keep the original label data unchanged
    fixed_nii = nib.Nifti1Image(fixed_label, fixed_affine, img_nii.header)
    
    # Update header to match image's orientation info
    fixed_nii.header.set_qform(img_nii.header.get_qform(), code=img_nii.header.get_qform_code())
    fixed_nii.header.set_sform(img_nii.header.get_sform(), code=img_nii.header.get_sform_code())
    
    # Save
    output_path = output_file if output_file else label_file
    nib.save(fixed_nii, str(output_path))
    
    # Verify
    saved_nii = nib.load(str(output_path))
    saved_data = np.asarray(saved_nii.dataobj)
    
    # Check direction matrices match
    img_direction = img_nii.affine[:3, :3]
    lbl_direction = saved_nii.affine[:3, :3]
    direction_match = np.allclose(img_direction, lbl_direction, atol=1e-6)
    
    print(f"    ✓ Saved fixed label: {output_path.name}")
    print(f"    Data preserved: {np.sum(saved_data > 0)} non-zero voxels")
    print(f"    Direction matrix match: {direction_match}")
    
    if not direction_match:
        print(f"    ⚠ Warning: Direction matrices still don't match exactly")
        print(f"      Image direction:\n{img_direction}")
        print(f"      Label direction:\n{lbl_direction}")
    
    return fixed_label


def fix_all_actin_orientations(images_dir, labels_dir, mapping_csv=None, output_dir=None, dry_run=False):
    """
    Fix orientation for all actin label files to match their images.
    
    Args:
        images_dir: Directory containing training images
        labels_dir: Directory containing training labels
        mapping_csv: CSV file with image-label mappings (optional)
        output_dir: Output directory for fixed labels (default: labels_dir + "_orientation_fixed")
        dry_run: If True, only show what would be done
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    if output_dir is None:
        output_dir = labels_path.parent / f"{labels_path.name}_orientation_fixed"
    else:
        output_dir = Path(output_dir)
    
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FIX ACTIN LABEL ORIENTATION")
    print("=" * 70)
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()
    
    # Get image-label pairs
    if mapping_csv and Path(mapping_csv).exists():
        # Use mapping CSV if provided
        df = pd.read_csv(mapping_csv)
        print(f"Using mapping CSV: {mapping_csv}")
        print(f"Found {len(df)} pairs")
        print()
        
        pairs = []
        for _, row in df.iterrows():
            case_num = row['case_number']
            img_file = images_path / f"case_{case_num:04d}_0000.nii.gz"
            lbl_file = labels_path / f"case_{case_num:04d}.nii.gz"
            
            if img_file.exists() and lbl_file.exists():
                pairs.append((case_num, img_file, lbl_file))
            else:
                print(f"⚠ Case {case_num}: Missing image or label file")
    else:
        # Auto-detect pairs
        image_files = sorted(images_path.glob("case_*_0000.nii.gz"))
        pairs = []
        
        for img_file in image_files:
            case_id = img_file.stem.rsplit("_", 1)[0]  # case_0000_0000 -> case_0000
            lbl_file = labels_path / f"{case_id}.nii.gz"
            
            if lbl_file.exists():
                case_num = int(case_id.split("_")[1])
                pairs.append((case_num, img_file, lbl_file))
    
    if not pairs:
        print("No matching image-label pairs found!")
        return
    
    print(f"Found {len(pairs)} image-label pairs\n")
    
    fixed_count = 0
    error_count = 0
    
    for case_num, img_file, lbl_file in sorted(pairs):
        print(f"Processing case_{case_num:04d}")
        
        output_file = output_dir / f"case_{case_num:04d}.nii.gz"
        
        if dry_run:
            print(f"  → Would fix orientation: {lbl_file.name} → {output_file.name}")
        else:
            try:
                fix_label_orientation_to_match_image(lbl_file, img_file, output_file)
                fixed_count += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed: {len(pairs)} pairs")
    print(f"Successfully fixed: {fixed_count}")
    if error_count > 0:
        print(f"Errors: {error_count}")
    print(f"Output directory: {output_dir}")
    if dry_run:
        print("\n⚠ This was a DRY RUN - no files were modified")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix orientation metadata for actin labels to match images"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/Volumes/LuisaHD/NewData/nnU-Net/actin_segmentation/imagesTr",
        help="Directory containing training images"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="/Volumes/LuisaHD/NewData/nnU-Net/actin_segmentation/labelsTr",
        help="Directory containing training labels"
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        default="/Volumes/LuisaHD/NewData/nnU-Net/actin_segmentation/actin_label_mapping.csv",
        help="CSV file with image-label mappings (optional)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for fixed labels"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    fix_all_actin_orientations(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        mapping_csv=args.mapping_csv,
        output_dir=args.output,
        dry_run=args.dry_run
    )

