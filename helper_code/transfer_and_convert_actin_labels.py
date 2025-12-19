"""
Transfer actin label files from archive to nnUNet format.

This script:
1. Copies NRRD files from source directory
2. Converts them to NII.GZ format
3. Renames to nnUNet format (case_XXXX.nii.gz)
4. Validates for nnUNet compatibility
5. Creates CSV mapping old vs new filenames
"""

import os
import re
import csv
import numpy as np
import nrrd
import nibabel as nib
from pathlib import Path
import pandas as pd


def get_spacing_from_affine(affine):
    """Extract voxel spacing from affine matrix."""
    spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    return spacing


def convert_nrrd_to_nii_and_validate(nrrd_file, output_file, case_number):
    """
    Convert NRRD to NII.GZ and validate for nnUNet.
    
    Returns:
        dict with validation results and file info
    """
    result = {
        'success': False,
        'errors': [],
        'warnings': [],
        'shape': None,
        'spacing': None,
        'dtype': None,
        'unique_labels': None,
        'non_zero_voxels': None
    }
    
    try:
        # Read NRRD file
        data, header = nrrd.read(str(nrrd_file))
        
        result['shape'] = data.shape
        result['dtype'] = str(data.dtype)
        
        # Check if 3D
        if data.ndim != 3:
            result['errors'].append(f"Not 3D: {data.ndim} dimensions")
            return result
        
        # Extract spacing from NRRD header
        spacing = None
        if 'space directions' in header:
            space_directions = header['space directions']
            if isinstance(space_directions, np.ndarray):
                spacing = np.sqrt(np.sum(space_directions**2, axis=1))
        elif 'spacings' in header:
            spacing = np.array(header['spacings'])
        
        if spacing is None:
            spacing = np.array([1.0, 1.0, 1.0])
            result['warnings'].append("No spacing found, using default [1.0, 1.0, 1.0]")
        
        result['spacing'] = spacing
        
        # Ensure spacing is 3D
        if len(spacing) != 3:
            if len(spacing) > 3:
                spacing = spacing[:3]
            else:
                spacing = np.pad(spacing, (0, 3 - len(spacing)), constant_values=1.0)
        
        # Check if this is a label file (should be integer type)
        data_int = data.astype(np.int32)
        unique_labels = np.unique(data_int)
        unique_labels = unique_labels[unique_labels >= 0]  # Remove negative
        
        result['unique_labels'] = len(unique_labels)
        result['non_zero_voxels'] = int(np.sum(data_int > 0))
        
        # Validate label format
        if len(unique_labels) == 0:
            result['warnings'].append("No labels found (all zeros)")
        else:
            min_label = int(np.min(unique_labels))
            max_label = int(np.max(unique_labels))
            
            if min_label != 0:
                result['warnings'].append(f"Labels don't start from 0 (min: {min_label})")
            
            if max_label > 255:
                result['warnings'].append(f"Max label {max_label} exceeds uint8 range")
        
        # Ensure uint8 type for labels
        if data.dtype != np.uint8:
            data = np.clip(data_int, 0, 255).astype(np.uint8)
            result['warnings'].append(f"Converted from {data.dtype} to uint8")
        
        # Create affine matrix
        affine = np.eye(4, dtype=np.float64)
        affine[0, 0] = spacing[0] if len(spacing) > 0 else 1.0
        affine[1, 1] = spacing[1] if len(spacing) > 1 else 1.0
        affine[2, 2] = spacing[2] if len(spacing) > 2 else 1.0
        
        # If we have space directions, use them
        if 'space directions' in header and isinstance(header['space directions'], np.ndarray):
            if header['space directions'].shape == (3, 3):
                affine[:3, :3] = header['space directions']
        
        # Create NIfTI image
        nii_img = nib.Nifti1Image(data, affine)
        
        # Save
        nib.save(nii_img, str(output_file))
        
        result['success'] = True
        
    except Exception as e:
        result['errors'].append(str(e))
        import traceback
        result['errors'].append(traceback.format_exc())
    
    return result


def transfer_and_convert_actin_labels(source_dir, dest_dir, mapping_csv_path=None):
    """
    Transfer and convert actin label files.
    
    Args:
        source_dir: Source directory with NRRD files
        dest_dir: Destination directory for NII.GZ files
        mapping_csv_path: Path to save CSV mapping file
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Find all NRRD files
    nrrd_files = sorted(source_path.glob("*.nrrd"))
    
    if not nrrd_files:
        print(f"No NRRD files found in {source_dir}")
        return
    
    print("=" * 70)
    print("TRANSFER AND CONVERT ACTIN LABELS")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Found {len(nrrd_files)} NRRD file(s)")
    print()
    
    # Mapping data
    mapping_data = []
    
    # Process each file
    case_counter = 1
    success_count = 0
    error_count = 0
    
    for nrrd_file in nrrd_files:
        old_name = nrrd_file.name
        old_path = str(nrrd_file)
        
        # Generate nnUNet-compatible name
        new_name = f"case_{case_counter:04d}.nii.gz"
        new_path = dest_path / new_name
        
        print(f"Processing [{case_counter}/{len(nrrd_files)}]: {old_name}")
        print(f"  → {new_name}")
        
        # Convert and validate
        result = convert_nrrd_to_nii_and_validate(nrrd_file, new_path, case_counter)
        
        # Record mapping
        mapping_entry = {
            'old_filename': old_name,
            'old_path': old_path,
            'new_filename': new_name,
            'new_path': str(new_path),
            'case_number': case_counter,
            'success': result['success'],
            'shape': str(result['shape']) if result['shape'] else 'N/A',
            'spacing': str(result['spacing']) if result['spacing'] is not None else 'N/A',
            'dtype': result['dtype'] or 'N/A',
            'unique_labels': result['unique_labels'] or 'N/A',
            'non_zero_voxels': result['non_zero_voxels'] or 'N/A',
            'errors': '; '.join(result['errors']) if result['errors'] else '',
            'warnings': '; '.join(result['warnings']) if result['warnings'] else ''
        }
        mapping_data.append(mapping_entry)
        
        if result['success']:
            print(f"  ✓ Converted successfully")
            print(f"    Shape: {result['shape']}, Spacing: {result['spacing']}")
            print(f"    Labels: {result['unique_labels']} unique, {result['non_zero_voxels']} non-zero voxels")
            if result['warnings']:
                for warning in result['warnings']:
                    print(f"    ⚠ {warning}")
            success_count += 1
        else:
            print(f"  ✗ Conversion failed")
            for error in result['errors']:
                print(f"    Error: {error}")
            error_count += 1
        
        case_counter += 1
        print()
    
    # Save mapping CSV
    if mapping_csv_path is None:
        mapping_csv_path = dest_path.parent / "actin_label_mapping.csv"
    else:
        mapping_csv_path = Path(mapping_csv_path)
    
    df = pd.DataFrame(mapping_data)
    df.to_csv(mapping_csv_path, index=False)
    print(f"✓ Saved mapping CSV: {mapping_csv_path}")
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(nrrd_files)}")
    print(f"Successfully converted: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Mapping CSV: {mapping_csv_path}")
    print("=" * 70)
    
    # Validation summary
    if success_count > 0:
        print("\nValidation Summary:")
        valid_files = [m for m in mapping_data if m['success']]
        
        # Check for common issues
        spacing_issues = [m for m in valid_files if 'spacing' in m['warnings'].lower() or m['spacing'] == 'N/A']
        label_issues = [m for m in valid_files if 'label' in m['warnings'].lower()]
        
        if spacing_issues:
            print(f"  ⚠ {len(spacing_issues)} files with spacing warnings")
        if label_issues:
            print(f"  ⚠ {len(label_issues)} files with label format warnings")
        
        if not spacing_issues and not label_issues:
            print("  ✓ All files validated successfully!")
    
    return mapping_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Transfer and convert actin label files to nnUNet format"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/Users/luisagrebici/Documents/Nezami_Lab/ARCHIVES - TO DELETE/data/labels_actin",
        help="Source directory with NRRD files"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="/Volumes/LuisaHD/NewData/nnU-Net/actin_segmentation/labelsTr",
        help="Destination directory for NII.GZ files"
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        default=None,
        help="Path to save mapping CSV (default: dest_dir/../actin_label_mapping.csv)"
    )
    
    args = parser.parse_args()
    
    transfer_and_convert_actin_labels(
        source_dir=args.source,
        dest_dir=args.dest,
        mapping_csv_path=args.mapping_csv
    )

