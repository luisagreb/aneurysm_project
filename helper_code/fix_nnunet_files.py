"""
Fix existing NII.GZ files to be nnUNet compatible.

This script:
1. Validates existing nii.gz files
2. Renames files to nnUNet format if needed
3. Fixes label format (consecutive integers starting from 0)
4. Validates shapes and data types
"""

import os
import re
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path


def fix_nnunet_files(input_folder, output_folder=None, is_labels=False, dry_run=False):
    """
    Fix NII.GZ files to be nnUNet compatible.
    
    Args:
        input_folder: Directory containing NII.GZ files
        output_folder: Output directory (default: input_folder + "_nnunet")
        is_labels: True if these are label files, False for images
        dry_run: If True, only show what would be done without making changes
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Directory {input_folder} does not exist")
        return
    
    # Set up output directory
    if output_folder is None:
        output_folder = input_path.parent / f"{input_path.name}_nnunet"
    else:
        output_folder = Path(output_folder)
    
    if not dry_run:
        output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Type: {'Labels' if is_labels else 'Images'}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will modify files)'}")
    print("=" * 70)
    
    # Find all NII.GZ files
    nii_files = sorted(input_path.glob("*.nii.gz"))
    
    if not nii_files:
        print(f"No .nii.gz files found in {input_folder}")
        return
    
    print(f"Found {len(nii_files)} NII.GZ file(s)\n")
    
    fixed_count = 0
    error_count = 0
    renamed_files = []
    
    for nii_file in nii_files:
        print(f"Processing: {nii_file.name}")
        
        try:
            # Load the file
            img = nib.load(str(nii_file))
            
            # Preserve original data type - use get_data() instead of get_fdata()
            # get_fdata() converts to float64, which changes file size
            data = np.asarray(img.dataobj)
            affine = img.affine
            header = img.header.copy()
            
            # Check shape
            if data.ndim != 3:
                print(f"  ✗ Skipping: Not 3D (shape: {data.shape})")
                error_count += 1
                continue
            
            original_dtype = data.dtype
            print(f"  Shape: {data.shape}, dtype: {original_dtype}")
            
            # Determine if this should be treated as a label file
            file_is_label = is_labels or "label" in str(nii_file).lower() or "mask" in str(nii_file).lower()
            
            # Track if modifications are needed
            needs_fix = False
            
            # Fix label format if needed
            if file_is_label:
                print(f"  Detected as LABEL file")
                
                # Convert to integer type
                data_int = data.astype(np.int32)
                unique_labels = np.unique(data_int)
                
                if len(unique_labels) > 0:
                    min_label = int(np.min(unique_labels))
                    max_label = int(np.max(unique_labels))
                    
                    print(f"  Original labels: min={min_label}, max={max_label}, unique={len(unique_labels)}")
                    
                    # Check if labels need fixing
                    needs_fix = False
                    if min_label != 0:
                        needs_fix = True
                        print(f"  ⚠ Labels don't start from 0")
                    
                    if len(unique_labels) != (max_label - min_label + 1):
                        needs_fix = True
                        print(f"  ⚠ Labels are not consecutive")
                    
                    if max_label > 255:
                        needs_fix = True
                        print(f"  ⚠ Max label {max_label} exceeds uint8 range")
                    
                    # Fix labels if needed
                    if needs_fix and not dry_run:
                        print(f"  → Fixing labels...")
                        # Remap to consecutive starting from 0
                        label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
                        data_fixed = np.zeros_like(data_int)
                        for old_label, new_label in label_map.items():
                            data_fixed[data_int == old_label] = new_label
                        data_int = data_fixed
                        
                        # Clip to uint8 range
                        data_int = np.clip(data_int, 0, 255)
                        data = np.ascontiguousarray(data_int.astype(np.uint8))
                        
                        unique_labels = np.unique(data)
                        print(f"  ✓ Fixed: {len(unique_labels)} consecutive labels starting from 0")
                    elif needs_fix:
                        print(f"  → Would fix labels (dry run)")
                    else:
                        # Ensure uint8 type
                        if data.dtype != np.uint8:
                            if not dry_run:
                                data = np.ascontiguousarray(data_int.astype(np.uint8))
                                print(f"  → Converted to uint8")
                            else:
                                print(f"  → Would convert to uint8 (dry run)")
                        else:
                            print(f"  ✓ Labels are already correct")
            
            # Generate nnUNet-compatible filename
            base_name = nii_file.stem.replace(".nii", "")  # Remove .nii from .nii.gz
            
            # Try to extract case number
            case_match = re.search(r'(?:Sample_|case_)(\d+)', base_name, re.IGNORECASE)
            if case_match:
                case_num = int(case_match.group(1))
                if file_is_label:
                    # Labels: case_0000.nii.gz (NO _0000 suffix)
                    new_name = f"case_{case_num:04d}.nii.gz"
                else:
                    # Images: case_0000_0000.nii.gz (WITH _0000 suffix)
                    new_name = f"case_{case_num:04d}_0000.nii.gz"
            else:
                # Try to preserve name but fix format
                if file_is_label:
                    new_name = base_name.replace("Sample_", "case_").replace("_0000", "") + ".nii.gz"
                else:
                    new_name = base_name.replace("Sample_", "case_") + ".nii.gz"
            
            output_file = output_folder / new_name
            
            # Check if name changed
            if nii_file.name != new_name:
                renamed_files.append((nii_file.name, new_name))
                print(f"  → Rename: {nii_file.name} → {new_name}")
            
            # Check if file needs modification
            needs_modification = False
            if file_is_label:
                # Labels might need fixing (already checked above)
                if needs_fix or data.dtype != np.uint8:
                    needs_modification = True
            else:
                # For images, check if we need to modify anything
                # If only renaming and data type is preserved, we can just copy
                if nii_file.name == new_name and data.dtype == original_dtype:
                    needs_modification = False  # Just copy, no modification needed
                else:
                    needs_modification = True
            
            # Save fixed file
            if not dry_run:
                if not needs_modification and nii_file.name == new_name:
                    # No changes needed - just copy the file to preserve exact size
                    import shutil
                    shutil.copy2(nii_file, output_file)
                    print(f"  ✓ Copied (no changes): {output_file.name}")
                else:
                    # Preserve original data type for images (don't convert unnecessarily)
                    if not file_is_label and data.dtype != original_dtype:
                        # Only convert if we actually modified the data
                        # For images, try to preserve original dtype
                        if original_dtype in [np.uint8, np.uint16, np.int16, np.float32, np.float64]:
                            data = data.astype(original_dtype)
                    
                    # Create new NIfTI image with fixed data and original header
                    nii_img = nib.Nifti1Image(data, affine, header)
                    
                    # Preserve compression - use same compression as original if possible
                    nib.save(nii_img, str(output_file))
                    
                    # Check if size changed significantly
                    original_size = nii_file.stat().st_size
                    new_size = output_file.stat().st_size
                    size_diff_pct = ((new_size - original_size) / original_size) * 100
                    
                    if abs(size_diff_pct) > 5:  # More than 5% difference
                        print(f"  ⚠ Size changed: {original_size/1024/1024:.2f} MB → {new_size/1024/1024:.2f} MB ({size_diff_pct:+.1f}%)")
                    else:
                        print(f"  ✓ Saved: {output_file.name} ({new_size/1024/1024:.2f} MB)")
            else:
                if not needs_modification and nii_file.name == new_name:
                    print(f"  → Would copy (no changes): {new_name} (dry run)")
                else:
                    print(f"  → Would save: {new_name} (dry run)")
            
            fixed_count += 1
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully processed: {fixed_count} file(s)")
    if error_count > 0:
        print(f"Errors: {error_count} file(s)")
    if renamed_files:
        print(f"\nFiles renamed ({len(renamed_files)}):")
        for old, new in renamed_files[:10]:  # Show first 10
            print(f"  {old} → {new}")
        if len(renamed_files) > 10:
            print(f"  ... and {len(renamed_files) - 10} more")
    print(f"\nOutput directory: {output_folder}")
    if dry_run:
        print("\n⚠ This was a DRY RUN - no files were modified")
        print("Run without --dry-run to apply changes")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    # ============================================================================
    # SET YOUR DEFAULT PATHS HERE (or use command-line arguments)
    # ============================================================================
    DEFAULT_INPUT_FOLDER_LABELS = "/Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/labelsTr"
    DEFAULT_INPUT_FOLDER_IMAGES = "/Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/imagesTr"
    
    # Change this to switch between labels and images:
    DEFAULT_INPUT_FOLDER = DEFAULT_INPUT_FOLDER_IMAGES  # Currently set to imagesTr
    
    parser = argparse.ArgumentParser(
        description="Fix existing NII.GZ files to be nnUNet compatible"
    )
    parser.add_argument(
        "input_folder",
        type=str,
        nargs="?",  # Make it optional
        default=DEFAULT_INPUT_FOLDER,
        help=f"Directory containing NII.GZ files to fix (default: {DEFAULT_INPUT_FOLDER})"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: input_folder_nnunet)"
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Treat files as labels (removes _0000 suffix, fixes label format)"
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="Treat files as images (keeps _0000 suffix)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    # Auto-detect if labels or images based on folder name
    # Default to images if imagesTr is in path, labels if labelsTr is in path
    folder_lower = args.input_folder.lower() if args.input_folder else ""
    
    if args.labels:
        is_labels = True
    elif args.images:
        is_labels = False
    else:
        # Auto-detect: imagesTr -> images, labelsTr -> labels
        if "imagestr" in folder_lower or "images" in folder_lower:
            is_labels = False  # It's images
        elif "labelstr" in folder_lower or "label" in folder_lower or "mask" in folder_lower:
            is_labels = True  # It's labels
        else:
            # Default based on current default path
            is_labels = "label" in DEFAULT_INPUT_FOLDER.lower()
    
    print("=" * 70)
    print("nnUNet File Fixer")
    print("=" * 70)
    print()
    
    fix_nnunet_files(
        input_folder=args.input_folder,
        output_folder=args.output,
        is_labels=is_labels,
        dry_run=args.dry_run
    )

