"""
Validate that NII.GZ files are ready for nnUNet training.

nnUNet requirements:
1. File naming: case_0000.nii.gz, case_0001.nii.gz, etc.
2. Labels must be consecutive integers starting from 0 (0=background, 1=class1, ...)
3. Labels must be integer type (uint8 recommended)
4. Images and labels must have matching shapes
5. Files should be in imagesTr/ and labelsTr/ folders
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import re


def validate_nnunet_files(images_dir, labels_dir=None):
    """
    Validate files for nnUNet compatibility.
    
    Args:
        images_dir: Directory containing image files (case_XXXX_0000.nii.gz)
        labels_dir: Directory containing label files (case_XXXX.nii.gz), optional
    """
    images_path = Path(images_dir)
    
    if not images_path.exists():
        print(f"Error: Images directory {images_dir} does not exist")
        return False
    
    # Find all image files
    image_files = sorted(images_path.glob("case_*_0000.nii.gz"))
    
    if not image_files:
        print(f"No image files found matching pattern 'case_*_0000.nii.gz' in {images_dir}")
        return False
    
    print(f"Found {len(image_files)} image file(s)")
    print("=" * 70)
    
    all_valid = True
    issues = []
    
    # Validate image files
    for img_file in image_files:
        case_id = img_file.stem.rsplit("_", 1)[0]  # case_0000_0000 -> case_0000
        print(f"\nValidating: {img_file.name}")
        
        try:
            img = nib.load(str(img_file))
            data = img.get_fdata()
            
            # Check shape
            if data.ndim != 3:
                issues.append(f"{img_file.name}: Not 3D (shape: {data.shape})")
                all_valid = False
                print(f"  ✗ Shape issue: {data.shape} (expected 3D)")
            else:
                print(f"  ✓ Shape: {data.shape}")
            
            # Check dtype
            if data.dtype not in [np.uint8, np.uint16, np.int16, np.float32, np.float64]:
                issues.append(f"{img_file.name}: Unusual dtype {data.dtype}")
                print(f"  ⚠ Dtype: {data.dtype} (may need conversion)")
            else:
                print(f"  ✓ Dtype: {data.dtype}")
            
            # Check if corresponding label exists
            if labels_dir:
                labels_path = Path(labels_dir)
                label_file = labels_path / f"{case_id}.nii.gz"
                
                if label_file.exists():
                    lbl = nib.load(str(label_file))
                    lbl_data = lbl.get_fdata()
                    
                    # Check shape match
                    if img.shape != lbl.shape:
                        issues.append(f"{case_id}: Shape mismatch - img {img.shape} vs lbl {lbl.shape}")
                        all_valid = False
                        print(f"  ✗ Label shape mismatch: {lbl.shape} (image: {img.shape})")
                    else:
                        print(f"  ✓ Label shape matches: {lbl.shape}")
                    
                    # Check label format
                    if lbl_data.dtype not in [np.uint8, np.uint16, np.int16]:
                        issues.append(f"{case_id}: Label dtype {lbl_data.dtype} should be integer")
                        print(f"  ✗ Label dtype: {lbl_data.dtype} (should be integer)")
                        all_valid = False
                    else:
                        print(f"  ✓ Label dtype: {lbl_data.dtype}")
                    
                    # Check label values
                    unique_labels = np.unique(lbl_data.astype(np.int32))
                    min_label = int(np.min(unique_labels))
                    max_label = int(np.max(unique_labels))
                    
                    if min_label != 0:
                        issues.append(f"{case_id}: Labels don't start from 0 (min: {min_label})")
                        print(f"  ✗ Labels start from {min_label}, should start from 0")
                        all_valid = False
                    else:
                        print(f"  ✓ Labels start from 0")
                    
                    # Check if consecutive
                    expected_labels = set(range(min_label, max_label + 1))
                    actual_labels = set(unique_labels)
                    if expected_labels != actual_labels:
                        missing = expected_labels - actual_labels
                        issues.append(f"{case_id}: Labels not consecutive (missing: {missing})")
                        print(f"  ⚠ Labels not consecutive (missing: {missing})")
                    else:
                        print(f"  ✓ Labels are consecutive: {unique_labels}")
                    
                    print(f"  ✓ Unique labels: {len(unique_labels)} ({unique_labels[:10]}...)" if len(unique_labels) > 10 else f"  ✓ Unique labels: {unique_labels}")
                else:
                    issues.append(f"{case_id}: Missing label file {label_file.name}")
                    print(f"  ⚠ No corresponding label file found")
        
        except Exception as e:
            issues.append(f"{img_file.name}: Error - {str(e)}")
            all_valid = False
            print(f"  ✗ Error: {str(e)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    if all_valid and not issues:
        print("✓ All files are ready for nnUNet!")
    else:
        print(f"✗ Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
    
    return all_valid


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate NII.GZ files for nnUNet compatibility"
    )
    parser.add_argument(
        "images_dir",
        type=str,
        help="Directory containing image files (case_XXXX_0000.nii.gz)"
    )
    parser.add_argument(
        "-l", "--labels",
        type=str,
        default=None,
        help="Directory containing label files (case_XXXX.nii.gz)"
    )
    
    args = parser.parse_args()
    
    validate_nnunet_files(args.images_dir, args.labels)

