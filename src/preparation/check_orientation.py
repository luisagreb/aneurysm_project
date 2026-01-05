"""
Check orientation metadata (direction matrix) between images and labels.

This script helps identify orientation mismatches that cause flipped/mirrored views.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd


def get_orientation_info(nii_file):
    """
    Extract orientation information from NIfTI file.
    
    Returns:
        dict with orientation details
    """
    img = nib.load(str(nii_file))
    affine = img.affine
    
    # Direction matrix (3x3 rotation/scaling part of affine)
    direction = affine[:3, :3]
    
    # Spacing (magnitude of direction vectors)
    spacing = np.sqrt(np.sum(direction**2, axis=0))
    
    # Normalized direction (orientation without scaling)
    normalized_direction = direction / spacing[np.newaxis, :]
    
    # Q-form and S-form codes
    qform_code = img.header.get_qform_code()
    sform_code = img.header.get_sform_code()
    
    # Q-form and S-form matrices
    qform = img.header.get_qform()
    sform = img.header.get_sform()
    
    return {
        'affine': affine,
        'direction': direction,
        'normalized_direction': normalized_direction,
        'spacing': spacing,
        'qform': qform,
        'sform': sform,
        'qform_code': qform_code,
        'sform_code': sform_code,
        'shape': img.shape
    }


def compare_orientations(img_file, lbl_file):
    """
    Compare orientation between image and label files.
    
    Returns:
        dict with comparison results
    """
    img_info = get_orientation_info(img_file)
    lbl_info = get_orientation_info(lbl_file)
    
    # Compare direction matrices
    direction_diff = np.abs(img_info['direction'] - lbl_info['direction'])
    direction_match = np.allclose(img_info['direction'], lbl_info['direction'], atol=1e-6)
    
    # Compare normalized directions (orientation without scaling)
    norm_dir_diff = np.abs(img_info['normalized_direction'] - lbl_info['normalized_direction'])
    norm_dir_match = np.allclose(img_info['normalized_direction'], 
                                 lbl_info['normalized_direction'], atol=1e-6)
    
    # Check if one is flipped (determinant sign)
    img_det = np.linalg.det(img_info['normalized_direction'])
    lbl_det = np.linalg.det(lbl_info['normalized_direction'])
    same_handedness = np.sign(img_det) == np.sign(lbl_det)
    
    # Check spacing
    spacing_diff = np.abs(img_info['spacing'] - lbl_info['spacing'])
    spacing_match = np.allclose(img_info['spacing'], lbl_info['spacing'], atol=1e-3)
    
    return {
        'direction_match': direction_match,
        'normalized_direction_match': norm_dir_match,
        'same_handedness': same_handedness,
        'spacing_match': spacing_match,
        'direction_diff': direction_diff,
        'normalized_direction_diff': norm_dir_diff,
        'spacing_diff': spacing_diff,
        'img_info': img_info,
        'lbl_info': lbl_info,
        'img_determinant': img_det,
        'lbl_determinant': lbl_det
    }


def check_all_orientations(images_dir, labels_dir, mapping_csv=None, sample_size=5):
    """
    Check orientation for all image-label pairs.
    
    Args:
        images_dir: Directory containing training images
        labels_dir: Directory containing training labels
        mapping_csv: Optional CSV with mappings
        sample_size: Number of files to check in detail (0 = all)
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    print("=" * 70)
    print("ORIENTATION CHECK")
    print("=" * 70)
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print()
    
    # Get pairs
    if mapping_csv and Path(mapping_csv).exists():
        df = pd.read_csv(mapping_csv)
        pairs = []
        for _, row in df.iterrows():
            case_num = row['case_number']
            img_file = images_path / f"case_{case_num:04d}_0000.nii.gz"
            lbl_file = labels_path / f"case_{case_num:04d}.nii.gz"
            if img_file.exists() and lbl_file.exists():
                pairs.append((case_num, img_file, lbl_file))
    else:
        image_files = sorted(images_path.glob("case_*_0000.nii.gz"))
        pairs = []
        for img_file in image_files:
            case_id = img_file.stem.rsplit("_", 1)[0]
            lbl_file = labels_path / f"{case_id}.nii.gz"
            if lbl_file.exists():
                case_num = int(case_id.split("_")[1])
                pairs.append((case_num, img_file, lbl_file))
    
    if not pairs:
        print("No matching pairs found!")
        return
    
    print(f"Found {len(pairs)} image-label pairs\n")
    
    # Check all pairs (quick check)
    mismatches = []
    matches = []
    
    print("Quick check (comparing direction matrices)...")
    for case_num, img_file, lbl_file in sorted(pairs):
        try:
            result = compare_orientations(img_file, lbl_file)
            
            if not result['direction_match']:
                mismatches.append((case_num, img_file, lbl_file, result))
            else:
                matches.append(case_num)
        except Exception as e:
            print(f"  ✗ Error checking case_{case_num:04d}: {e}")
    
    print(f"  Matches: {len(matches)}")
    print(f"  Mismatches: {len(mismatches)}")
    print()
    
    # Detailed check on sample or all mismatches
    if sample_size > 0 and len(mismatches) > sample_size:
        print(f"Showing detailed analysis of first {sample_size} mismatches:\n")
        check_list = mismatches[:sample_size]
    elif mismatches:
        print("Detailed analysis of all mismatches:\n")
        check_list = mismatches
    else:
        print("✓ All orientations match!")
        return
    
    for case_num, img_file, lbl_file, result in check_list:
        print(f"Case: case_{case_num:04d}")
        print(f"  Image: {img_file.name}")
        print(f"  Label: {lbl_file.name}")
        print()
        
        img_info = result['img_info']
        lbl_info = result['lbl_info']
        
        print("  Image orientation:")
        print(f"    Direction matrix:\n{img_info['direction']}")
        print(f"    Normalized direction:\n{img_info['normalized_direction']}")
        print(f"    Spacing: {img_info['spacing']}")
        print(f"    Determinant: {result['img_determinant']:.6f}")
        print(f"    Q-form code: {img_info['qform_code']}, S-form code: {img_info['sform_code']}")
        print()
        
        print("  Label orientation:")
        print(f"    Direction matrix:\n{lbl_info['direction']}")
        print(f"    Normalized direction:\n{lbl_info['normalized_direction']}")
        print(f"    Spacing: {lbl_info['spacing']}")
        print(f"    Determinant: {result['lbl_determinant']:.6f}")
        print(f"    Q-form code: {lbl_info['qform_code']}, S-form code: {lbl_info['sform_code']}")
        print()
        
        print("  Differences:")
        print(f"    Direction matrix difference:\n{result['direction_diff']}")
        print(f"    Normalized direction difference:\n{result['normalized_direction_diff']}")
        print(f"    Spacing difference: {result['spacing_diff']}")
        print()
        
        if not result['same_handedness']:
            print("  ⚠ HANDEDNESS MISMATCH!")
            print("    One file is flipped/mirrored relative to the other")
            print(f"    Image determinant: {result['img_determinant']:.6f}")
            print(f"    Label determinant: {result['lbl_determinant']:.6f}")
        elif not result['normalized_direction_match']:
            print("  ⚠ ORIENTATION MISMATCH!")
            print("    Direction matrices differ (may cause flipped/mirrored views)")
        elif not result['spacing_match']:
            print("  ⚠ SPACING MISMATCH!")
            print("    Spacing differs but orientation is the same")
        else:
            print("  ✓ Orientation matches")
        
        print("-" * 70)
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total pairs: {len(pairs)}")
    print(f"Orientation matches: {len(matches)}")
    print(f"Orientation mismatches: {len(mismatches)}")
    
    if mismatches:
        print("\n⚠ Orientation mismatches detected!")
        print("This can cause labels to appear flipped/mirrored in viewers.")
        print("\nTo fix, run:")
        print("  python helper_code/fix_actin_orientation.py")
    else:
        print("\n✓ All orientations match correctly!")
    
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check orientation metadata between images and labels"
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
        "--sample-size",
        type=int,
        default=5,
        help="Number of mismatches to show in detail (0 = all)"
    )
    
    args = parser.parse_args()
    
    check_all_orientations(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        mapping_csv=args.mapping_csv,
        sample_size=args.sample_size
    )

