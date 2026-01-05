"""
Prepare test data for nnUNet prediction.

This script:
1. Finds channel_00.nrrd files in source directory
2. Converts them to NII.GZ format
3. Validates they are 3D and correct
4. Renames them to nnUNet test format (case_XXXX_0000.nii.gz)
5. Saves to imagesTs folder for nnUNet prediction
"""

import os
import re
import numpy as np
import nrrd
import nibabel as nib
from pathlib import Path
import pandas as pd


def get_spacing_from_affine(affine):
    """Extract voxel spacing from affine matrix."""
    spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    return spacing


def convert_nrrd_to_nii(nrrd_file, output_file):
    """
    Convert NRRD file to NII.GZ format, preserving spacing and orientation.
    
    Returns:
        dict with conversion results and validation info
    """
    result = {
        'success': False,
        'errors': [],
        'warnings': [],
        'shape': None,
        'spacing': None,
        'dtype': None,
        'is_3d': False
    }
    
    try:
        # Read NRRD file
        data, header = nrrd.read(str(nrrd_file))
        
        result['shape'] = data.shape
        result['dtype'] = str(data.dtype)
        
        # Check if 3D
        if data.ndim != 3:
            result['errors'].append(f"Not 3D: {data.ndim} dimensions (expected 3)")
            result['is_3d'] = False
            return result
        
        result['is_3d'] = True
        
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
        
        # Ensure spacing is 3D
        if len(spacing) != 3:
            if len(spacing) > 3:
                spacing = spacing[:3]
            else:
                spacing = np.pad(spacing, (0, 3 - len(spacing)), constant_values=1.0)
        
        result['spacing'] = spacing
        
        # Ensure data is in correct format
        if data.dtype not in [np.uint8, np.uint16, np.int16, np.float32, np.float64]:
            data = data.astype(np.float32)
            result['warnings'].append(f"Converted dtype from {data.dtype} to float32")
        else:
            data = np.ascontiguousarray(data)
        
        # Create affine matrix
        affine = np.eye(4, dtype=np.float64)
        affine[0, 0] = spacing[0] if len(spacing) > 0 else 1.0
        affine[1, 1] = spacing[1] if len(spacing) > 1 else 1.0
        affine[2, 2] = spacing[2] if len(spacing) > 2 else 1.0
        
        # Use space directions if available
        if 'space directions' in header and isinstance(header['space directions'], np.ndarray):
            if header['space directions'].shape == (3, 3):
                affine[:3, :3] = header['space directions']
        
        # Get space origin if available
        if 'space origin' in header:
            space_origin = np.array(header['space origin'])
            if len(space_origin) >= 3:
                affine[:3, 3] = space_origin[:3]
        
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


def prepare_test_data(source_dir, output_dir, mapping_csv_path=None):
    """
    Prepare test data for nnUNet prediction.
    
    Args:
        source_dir: Directory containing channel_00.nrrd files
        output_dir: Output directory for test images (imagesTs)
        mapping_csv_path: Path to save CSV mapping old to new names
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all channel_00.nrrd files
    nrrd_files = sorted(source_path.glob("**/channel_00.nrrd"))
    
    if not nrrd_files:
        print(f"No channel_00.nrrd files found in {source_dir}")
        return
    
    print("=" * 70)
    print("PREPARE TEST DATA FOR nnUNet")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(nrrd_files)} channel_00.nrrd file(s)")
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
        
        # Generate nnUNet-compatible name for test images
        # Test images format: case_XXXX_0000.nii.gz
        new_name = f"case_{case_counter:04d}_0000.nii.gz"
        new_path = output_path / new_name
        
        print(f"Processing [{case_counter}/{len(nrrd_files)}]: {nrrd_file.parent.name}/{old_name}")
        print(f"  → {new_name}")
        
        # Convert and validate
        result = convert_nrrd_to_nii(nrrd_file, new_path)
        
        # Record mapping
        mapping_entry = {
            'case_number': case_counter,
            'old_filename': old_name,
            'old_path': old_path,
            'old_parent_dir': str(nrrd_file.parent.name),
            'new_filename': new_name,
            'new_path': str(new_path),
            'success': result['success'],
            'is_3d': result['is_3d'],
            'shape': str(result['shape']) if result['shape'] else 'N/A',
            'spacing': str(result['spacing']) if result['spacing'] is not None else 'N/A',
            'dtype': result['dtype'] or 'N/A',
            'errors': '; '.join(result['errors']) if result['errors'] else '',
            'warnings': '; '.join(result['warnings']) if result['warnings'] else ''
        }
        mapping_data.append(mapping_entry)
        
        if result['success']:
            print(f"  ✓ Converted successfully")
            print(f"    Shape: {result['shape']}, Spacing: {result['spacing']}")
            if result['warnings']:
                for warning in result['warnings']:
                    print(f"    ⚠ {warning}")
            success_count += 1
        else:
            print(f"  ✗ Conversion failed")
            for error in result['errors'][:2]:  # Show first 2 errors
                print(f"    Error: {error}")
            error_count += 1
        
        case_counter += 1
        print()
    
    # Save mapping CSV
    if mapping_csv_path is None:
        mapping_csv_path = output_path.parent / "test_data_mapping.csv"
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
    print(f"3D files: {sum(1 for m in mapping_data if m['is_3d'])}")
    print(f"Output directory: {output_path}")
    print(f"Mapping CSV: {mapping_csv_path}")
    print()
    
    # Validation summary
    if success_count > 0:
        print("Validation Summary:")
        valid_files = [m for m in mapping_data if m['success']]
        
        # Check for common issues
        not_3d = [m for m in valid_files if not m['is_3d']]
        spacing_issues = [m for m in valid_files if 'spacing' in m['warnings'].lower() or m['spacing'] == 'N/A']
        
        if not_3d:
            print(f"  ✗ {len(not_3d)} files are not 3D (will be skipped)")
        if spacing_issues:
            print(f"  ⚠ {len(spacing_issues)} files with spacing warnings")
        
        if not not_3d and not spacing_issues:
            print("  ✓ All files validated successfully!")
        
        print()
        print("Files are ready for nnUNet prediction!")
        print(f"Use: nnUNetv2_predict -i {output_path} -o /path/to/output -d DatasetXXX -f 0")
    
    print("=" * 70)
    
    return mapping_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare test data for nnUNet prediction"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/Volumes/LuisaHD/NewData/imagesTs_nrrd",
        help="Source directory containing channel_00.nrrd files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/imagesTs",
        help="Output directory for test images (imagesTs)"
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        default=None,
        help="Path to save mapping CSV (default: output_dir/../test_data_mapping.csv)"
    )
    
    args = parser.parse_args()
    
    prepare_test_data(
        source_dir=args.source,
        output_dir=args.output,
        mapping_csv_path=args.mapping_csv
    )

