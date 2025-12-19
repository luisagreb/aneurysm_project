#!/usr/bin/env python3
"""
Prepare data for nnUNet V2 training
Converts NRRD files to NIfTI format and organizes into nnUNet directory structure

Usage:
    python prepare_nnunet_data.py

Directory structure created:
    output_dir/
    ├── imagesTr/     # Training images (with labels)
    ├── labelsTr/     # Training labels
    └── imagesTs/     # Test images (without labels)
"""

import os
import numpy as np
import nrrd
import nibabel as nib
from pathlib import Path
import shutil


def nrrd_to_nifti(nrrd_path, nifti_path, verbose=True):
    """
    Convert NRRD file to NIfTI format
    
    Args:
        nrrd_path: Path to input NRRD file
        nifti_path: Path to output NIfTI file
        verbose: Print conversion info
    """
    # Read NRRD file
    data, header = nrrd.read(str(nrrd_path))
    
    if verbose:
        print(f"    Reading: {nrrd_path.name}")
        print(f"    Shape: {data.shape}, dtype: {data.dtype}")
    
    # Get spacing information (important for nnUNet)
    if 'space directions' in header:
        # Extract spacing from space directions matrix
        space_directions = header['space directions']
        spacing = np.array([np.linalg.norm(space_directions[i]) for i in range(len(space_directions))])
    elif 'spacings' in header:
        spacing = np.array(header['spacings'])
    else:
        # Default spacing if not found
        spacing = np.ones(len(data.shape))
        print(f"    Warning: No spacing found, using default: {spacing}")
    
    # Create affine matrix for NIfTI
    # nnUNet expects RAS+ orientation
    affine = np.eye(4)
    affine[0, 0] = spacing[2] if len(spacing) > 2 else 1.0  # X spacing
    affine[1, 1] = spacing[1] if len(spacing) > 1 else 1.0  # Y spacing
    affine[2, 2] = spacing[0] if len(spacing) > 0 else 1.0  # Z spacing
    
    # Convert data to expected orientation (Z, Y, X) -> (X, Y, Z) for NIfTI
    # NRRD typically stores as (Z, Y, X), NIfTI expects (X, Y, Z)
    data_nifti = np.transpose(data, (2, 1, 0))
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(data_nifti, affine)
    
    # Save NIfTI file
    nib.save(nifti_img, str(nifti_path))
    
    if verbose:
        print(f"    Saved: {nifti_path.name}")
        print(f"    Spacing: {spacing}")
    
    return True


def get_case_name_from_folder(folder_path):
    """
    Extract case name from folder path
    Example: "aneurysm_001" -> "aneurysm_001"
    """
    return folder_path.name


def prepare_nnunet_dataset(images_dir, labels_dir, output_dir, actin_channel=1):
    """
    Prepare nnUNet V2 dataset from NRRD files
    
    Args:
        images_dir: Directory containing subdirectories with channel NRRD files
        labels_dir: Directory containing subdirectories with label NRRD files
        output_dir: Output directory for nnUNet dataset
        actin_channel: Which channel is the actin channel (default: 1)
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    images_tr_dir = output_path / "imagesTr"
    labels_tr_dir = output_path / "labelsTr"
    images_ts_dir = output_path / "imagesTs"
    
    images_tr_dir.mkdir(parents=True, exist_ok=True)
    labels_tr_dir.mkdir(parents=True, exist_ok=True)
    images_ts_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directories:")
    print(f"  - {images_tr_dir}")
    print(f"  - {labels_tr_dir}")
    print(f"  - {images_ts_dir}")
    print()
    
    # Find all subdirectories in images folder
    image_subdirs = sorted([d for d in images_path.iterdir() if d.is_dir()])
    
    if not image_subdirs:
        print(f"Error: No subdirectories found in {images_dir}")
        return
    
    print(f"Found {len(image_subdirs)} image folders")
    
    # Find all label files (they are .nrrd files directly in the labels directory)
    label_files = list(labels_path.glob("*.nrrd"))
    # Extract case names from label filenames (remove .nrrd extension)
    label_names = {f.stem for f in label_files}
    
    print(f"Found {len(label_files)} label files")
    print()
    
    train_count = 0
    test_count = 0
    
    # Process each image subdirectory
    for idx, image_subdir in enumerate(image_subdirs, start=1):
        case_name = get_case_name_from_folder(image_subdir)
        print(f"[{idx}/{len(image_subdirs)}] Processing: {case_name}")
        
        # Find the actin channel file
        actin_file = image_subdir / f"channel_{actin_channel:02d}.nrrd"
        
        if not actin_file.exists():
            print(f"  Warning: Actin channel file not found: {actin_file}")
            print(f"  Skipping {case_name}")
            print()
            continue
        
        # Check if this case has a label
        has_label = case_name in label_names
        
        if has_label:
            # This is a training case
            print(f"  Type: TRAINING (has label)")
            
            # Convert image to NIfTI
            nnunet_image_name = f"{case_name}_0000.nii.gz"
            image_output_path = images_tr_dir / nnunet_image_name
            
            try:
                nrrd_to_nifti(actin_file, image_output_path, verbose=False)
                print(f"  ✓ Image: {nnunet_image_name}")
            except Exception as e:
                print(f"  ✗ Error converting image: {e}")
                continue
            
            # Find label file directly in labels directory
            label_file = labels_path / f"{case_name}.nrrd"
            
            if not label_file.exists():
                print(f"  ✗ Warning: No label file found: {label_file}")
                print(f"  Skipping this case")
                print()
                continue
            
            # Convert label to NIfTI
            nnunet_label_name = f"{case_name}.nii.gz"
            label_output_path = labels_tr_dir / nnunet_label_name
            
            try:
                nrrd_to_nifti(label_file, label_output_path, verbose=False)
                print(f"  ✓ Label: {nnunet_label_name}")
                train_count += 1
            except Exception as e:
                print(f"  ✗ Error converting label: {e}")
                # Remove the image since we don't have a valid label
                image_output_path.unlink(missing_ok=True)
                continue
        
        else:
            # This is a test case (no label)
            print(f"  Type: TEST (no label)")
            
            # Convert image to NIfTI
            nnunet_image_name = f"{case_name}_0000.nii.gz"
            image_output_path = images_ts_dir / nnunet_image_name
            
            try:
                nrrd_to_nifti(actin_file, image_output_path, verbose=False)
                print(f"  ✓ Image: {nnunet_image_name}")
                test_count += 1
            except Exception as e:
                print(f"  ✗ Error converting image: {e}")
                continue
        
        print()
    
    # Summary
    print("=" * 70)
    print("✓ Dataset preparation complete!")
    print("=" * 70)
    print(f"Training cases: {train_count}")
    print(f"Test cases: {test_count}")
    print(f"Total cases: {train_count + test_count}")
    print()
    print("Output structure:")
    print(f"  {output_path}/")
    print(f"  ├── imagesTr/     ({train_count} files)")
    print(f"  ├── labelsTr/     ({train_count} files)")
    print(f"  └── imagesTs/     ({test_count} files)")
    print()
    print("Next steps:")
    print("1. Create dataset.json in the output directory")
    print("2. Run nnUNet_plan_and_preprocess")
    print("=" * 70)


if __name__ == "__main__":
    # Configuration
    # Paths on VM
    
    images_directory = "/home/luisa/aneurysm_project/data/nrrd_files"
    labels_directory = "/home/luisa/aneurysm_project/data/manually_segmented"
    output_directory = "/home/luisa/aneurysm_project/data/nnunet_data/actin_segmentation"
    
    # Which channel is the actin channel (0, 1, or 2)
    # Based on your previous script, channel 1 is the actin channel
    actin_channel_number = 1
    
    print("=" * 70)
    print("nnUNet V2 DATA PREPARATION")
    print("=" * 70)
    print(f"Images directory: {images_directory}")
    print(f"Labels directory: {labels_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Actin channel: {actin_channel_number}")
    print("=" * 70)
    print()
    
    # Prepare the dataset
    prepare_nnunet_dataset(
        images_dir=images_directory,
        labels_dir=labels_directory,
        output_dir=output_directory,
        actin_channel=actin_channel_number
    )
