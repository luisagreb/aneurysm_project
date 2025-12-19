"""
Transfer and match actin image files with their labels.

This script:
1. Reads the label mapping CSV to get old filenames
2. Matches images to labels based on old filename
3. Converts/copies images to nnUNet format (case_XXXX_0000.nii.gz)
4. Validates matching
"""

import os
import re
import csv
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path


def get_spacing_from_affine(affine):
    """Extract voxel spacing from affine matrix."""
    spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    return spacing


def normalize_name_for_matching(name):
    """
    Normalize filename for matching (handle case, typos, zero-padding).
    """
    # Convert to lowercase
    normalized = name.lower()
    
    # Handle common variations
    normalized = normalized.replace('asc', 'asc')  # Keep lowercase
    normalized = normalized.replace('aneursym', 'aneurysm')  # Fix typo
    normalized = normalized.replace('nonaneurysm', 'nonaneurysm')
    
    # Remove zero-padding from numbers (e.g., 0180 -> 180)
    import re
    normalized = re.sub(r'0+(\d+)', r'\1', normalized)
    
    return normalized


def extract_base_name(filename):
    """
    Extract base name from filename for matching.
    E.g., '01Asc-180_nonAneurysm_coll_actin_cell1.nrrd' -> '01Asc-180_nonAneurysm_coll_actin_cell1'
    """
    # Remove extension
    base = Path(filename).stem
    # Handle .nii.gz case
    if base.endswith('.nii'):
        base = base[:-4]
    return base


def find_matching_image(image_dir, label_base_name):
    """
    Find image file that matches the label base name.
    Handles case differences, typos, and zero-padding variations.
    """
    image_path = Path(image_dir)
    
    # Normalize label name for matching
    label_normalized = normalize_name_for_matching(label_base_name)
    
    # Get all image files
    all_images = list(image_path.glob("*.nii.gz")) + list(image_path.glob("*.nrrd")) + \
                 list(image_path.glob("*.nii")) + list(image_path.glob("*.tif")) + \
                 list(image_path.glob("*.tiff"))
    
    # Try exact match first (case-insensitive)
    for img_file in all_images:
        img_base = extract_base_name(img_file.name)
        if normalize_name_for_matching(img_base) == label_normalized:
            return img_file
    
    # Try without 'actin' in name
    label_no_actin = label_base_name.replace('_actin', '').replace('actin', '')
    label_no_actin_norm = normalize_name_for_matching(label_no_actin)
    
    for img_file in all_images:
        img_base = extract_base_name(img_file.name)
        img_normalized = normalize_name_for_matching(img_base)
        if img_normalized == label_no_actin_norm:
            return img_file
    
    # Try pattern matching - extract key parts
    # Format: 01Asc-180_nonAneurysm_coll_actin_cell1
    parts = label_base_name.split('_')
    
    # Try matching by cell number and main identifier
    if len(parts) >= 2:
        # Get cell number (last part)
        cell_part = parts[-1]  # e.g., 'cell1'
        
        # Get main identifier (first part)
        main_id = parts[0]  # e.g., '01Asc-180'
        main_id_norm = normalize_name_for_matching(main_id)
        
        # Try to find file with same cell number and similar main ID
        for img_file in all_images:
            img_base = extract_base_name(img_file.name)
            img_normalized = normalize_name_for_matching(img_base)
            
            # Check if cell number matches and main ID is similar
            if cell_part.lower() in img_normalized and main_id_norm in img_normalized:
                return img_file
    
    # Last resort: try fuzzy matching on key components
    # Extract: sample_id (e.g., 01Asc-180), condition (coll/noColl), cell number
    sample_match = re.search(r'(\d+[a-z]+-\d+)', label_base_name, re.IGNORECASE)
    cell_match = re.search(r'cell(\d+)', label_base_name, re.IGNORECASE)
    
    if sample_match and cell_match:
        sample_id = sample_match.group(1)
        cell_num = cell_match.group(1)
        sample_id_norm = normalize_name_for_matching(sample_id)
        
        for img_file in all_images:
            img_base = extract_base_name(img_file.name)
            img_normalized = normalize_name_for_matching(img_base)
            
            if sample_id_norm in img_normalized and f'cell{cell_num}' in img_normalized:
                return img_file
    
    return None


def convert_to_nii_if_needed(input_file, output_file):
    """
    Convert file to NII.GZ format if needed.
    Handles NRRD, NIfTI, and other formats.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # Check file extension
    ext = input_path.suffix.lower()
    
    if ext == '.gz':
        # Check if it's .nii.gz
        if input_path.name.endswith('.nii.gz'):
            # Already NII.GZ, just copy
            shutil.copy2(input_path, output_path)
            return True
    
    if ext == '.nii':
        # Already NIfTI, just copy and compress
        img = nib.load(str(input_path))
        nib.save(img, str(output_path))
        return True
    
    if ext == '.nrrd':
        # Convert NRRD to NII.GZ
        import nrrd
        data, header = nrrd.read(str(input_path))
        
        # Extract spacing
        spacing = None
        if 'space directions' in header:
            space_directions = header['space directions']
            if isinstance(space_directions, np.ndarray):
                spacing = np.sqrt(np.sum(space_directions**2, axis=1))
        elif 'spacings' in header:
            spacing = np.array(header['spacings'])
        
        if spacing is None:
            spacing = np.array([1.0, 1.0, 1.0])
        
        if len(spacing) != 3:
            if len(spacing) > 3:
                spacing = spacing[:3]
            else:
                spacing = np.pad(spacing, (0, 3 - len(spacing)), constant_values=1.0)
        
        # Create affine
        affine = np.eye(4, dtype=np.float64)
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        
        # Use space directions if available
        if 'space directions' in header and isinstance(header['space directions'], np.ndarray):
            if header['space directions'].shape == (3, 3):
                affine[:3, :3] = header['space directions']
        
        # Ensure data is in correct format
        if data.dtype not in [np.uint8, np.uint16, np.int16, np.float32, np.float64]:
            data = data.astype(np.float32)
        else:
            data = np.ascontiguousarray(data)
        
        # Create and save NIfTI
        nii_img = nib.Nifti1Image(data, affine)
        nib.save(nii_img, str(output_path))
        return True
    
    # For other formats, try loading with nibabel
    try:
        img = nib.load(str(input_path))
        nib.save(img, str(output_path))
        return True
    except:
        pass
    
    return False


def transfer_and_match_images(image_dir, label_mapping_csv, output_dir):
    """
    Transfer and match image files with their labels.
    
    Args:
        image_dir: Directory containing source image files
        label_mapping_csv: Path to CSV file with label mappings
        output_dir: Output directory for training images
    """
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    mapping_path = Path(label_mapping_csv)
    
    if not image_path.exists():
        print(f"Error: Image directory does not exist: {image_dir}")
        return
    
    if not mapping_path.exists():
        print(f"Error: Mapping CSV does not exist: {label_mapping_csv}")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read mapping CSV
    df = pd.read_csv(mapping_path)
    
    print("=" * 70)
    print("TRANSFER AND MATCH ACTIN IMAGES")
    print("=" * 70)
    print(f"Image source: {image_dir}")
    print(f"Label mapping: {label_mapping_csv}")
    print(f"Output: {output_dir}")
    print(f"Found {len(df)} label entries in mapping CSV")
    print()
    
    # Track results
    matched = []
    not_found = []
    errors = []
    
    # Process each label entry
    for idx, row in df.iterrows():
        old_label_name = row['old_filename']
        case_number = row['case_number']
        new_label_name = row['new_filename']
        
        # Extract base name from label filename
        label_base = extract_base_name(old_label_name)
        
        print(f"Processing [{idx+1}/{len(df)}]: {old_label_name}")
        print(f"  Label: {new_label_name} (case_{case_number:04d})")
        
        # Find matching image
        matching_image = find_matching_image(image_path, label_base)
        
        if not matching_image:
            print(f"  ✗ No matching image found")
            not_found.append({
                'case_number': case_number,
                'label_name': old_label_name,
                'label_base': label_base
            })
            continue
        
        print(f"  ✓ Found image: {matching_image.name}")
        
        # Generate output filename (images need _0000 suffix)
        output_filename = f"case_{case_number:04d}_0000.nii.gz"
        output_file = output_path / output_filename
        
        # Convert/copy image
        try:
            success = convert_to_nii_if_needed(matching_image, output_file)
            
            if success:
                # Verify the file was created
                if output_file.exists():
                    # Load and verify
                    img = nib.load(str(output_file))
                    data = np.asarray(img.dataobj)
                    
                    print(f"  ✓ Copied/converted: {output_filename}")
                    print(f"    Shape: {data.shape}")
                    
                    matched.append({
                        'case_number': case_number,
                        'label_name': old_label_name,
                        'image_name': matching_image.name,
                        'new_image_name': output_filename,
                        'shape': str(data.shape)
                    })
                else:
                    raise Exception("Output file was not created")
            else:
                raise Exception("Conversion failed")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            errors.append({
                'case_number': case_number,
                'label_name': old_label_name,
                'image_name': matching_image.name,
                'error': str(e)
            })
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total labels: {len(df)}")
    print(f"Matched images: {len(matched)}")
    print(f"Not found: {len(not_found)}")
    print(f"Errors: {len(errors)}")
    print()
    
    if not_found:
        print("Images not found:")
        for item in not_found[:10]:  # Show first 10
            print(f"  Case {item['case_number']}: {item['label_name']}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")
        print()
    
    if errors:
        print("Errors:")
        for item in errors[:10]:
            print(f"  Case {item['case_number']}: {item['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        print()
    
    # Save matching report
    if matched:
        matched_df = pd.DataFrame(matched)
        report_path = output_path.parent / "actin_image_matching_report.csv"
        matched_df.to_csv(report_path, index=False)
        print(f"✓ Saved matching report: {report_path}")
    
    print("=" * 70)
    
    return matched, not_found, errors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Transfer and match actin image files with labels"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/Users/luisagrebici/Documents/Nezami_Lab/ARCHIVES - TO DELETE/data/raw_ch2_actin_all",
        help="Directory containing source image files"
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        default="/Volumes/LuisaHD/NewData/nnU-Net/actin_segmentation/actin_label_mapping.csv",
        help="Path to label mapping CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Volumes/LuisaHD/NewData/nnU-Net/actin_segmentation/imagesTr",
        help="Output directory for training images"
    )
    
    args = parser.parse_args()
    
    transfer_and_match_images(
        image_dir=args.image_dir,
        label_mapping_csv=args.mapping_csv,
        output_dir=args.output_dir
    )

