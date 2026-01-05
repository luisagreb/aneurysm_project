#!/usr/bin/env python3
"""
Generate test data for BOTH Actin (Dataset 001) and Mitochondria (Dataset 002)
Extracts Channel 1 (Actin) and Channel 2 (Mitochondria) from unused cases in nrrd_files
"""

import nrrd
import nibabel as nib
import numpy as np
from pathlib import Path
import csv

def nrrd_to_nifti(nrrd_path, nifti_path):
    data, header = nrrd.read(str(nrrd_path))
    
    # Handle spacing
    if 'space directions' in header:
        space_directions = header['space directions']
        spacing = np.array([np.linalg.norm(space_directions[i]) for i in range(len(space_directions))])
    elif 'spacings' in header:
        spacing = np.array(header['spacings'])
    else:
        spacing = np.ones(len(data.shape))

    # RAS+ orientation (transpose Z,Y,X -> X,Y,Z)
    affine = np.eye(4)
    affine[0, 0] = spacing[2] if len(spacing) > 2 else 1.0
    affine[1, 1] = spacing[1] if len(spacing) > 1 else 1.0
    affine[2, 2] = spacing[0] if len(spacing) > 0 else 1.0
    
    data_nifti = np.transpose(data, (2, 1, 0))
    nifti_img = nib.Nifti1Image(data_nifti, affine)
    nib.save(nifti_img, str(nifti_path))

def generate_test_set(dataset_type):
    # Paths configuration
    raw_data_dir = Path("/home/luisa/aneurysm_project/data/raw/nrrd_files")
    
    if dataset_type == 'actin':
        data_dir = Path("/home/luisa/aneurysm_project/data/processed/actin_segmentation")
        channel_id = 1
        prefix = "case"
        
        # Determine training cases to skip directly from existing dataset structure if possible
        # For Actin, training cases are in imagesTr
        images_tr_dir = data_dir / "imagesTr"
        # We need to ideally know original names to skip them. 
        # If mapping file exists use it, otherwise rely on case count?
        # Let's try to infer from dataset.json which maps original files if kept, or use mapping csv
        # For safety, let's use the explicit mapping file if it exists, roughly...
        # Actually for Actin, let's look for manually segmented folders to skip
        manually_segmented_dir = Path("/home/luisa/aneurysm_project/data/raw/manually_segmented")
        training_cases = {f.stem for f in manually_segmented_dir.glob("*.nrrd")}
        print(f"Skipping {len(training_cases)} manual training cases for Actin.")
        
    elif dataset_type == 'mito':
        data_dir = Path("/home/luisa/aneurysm_project/data/processed/nnUnet_mitochondria/Dataset002_Mito")
        channel_id = 2
        prefix = "Mito"
        
        # Load training cases to skip from mapping file
        mapping_file = data_dir / "../Mito_mapping.csv"
        training_cases = set()
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    training_cases.add(row['OriginalCellName'])
        print(f"Skipping {len(training_cases)} training cases for Mitochondria.")
        
    else:
        print("Invalid dataset type")
        return

    # Create imagesTs
    images_ts_dir = data_dir / "imagesTs"
    images_ts_dir.mkdir(exist_ok=True, parents=True)
    
    # Check what's already converted to avoid duplicates if possible, or overwrite?
    # Let's overwrite to be safe and ensure consistency
    
    subdirs = sorted([d for d in raw_data_dir.iterdir() if d.is_dir()])
    
    count = 0
    # Start ID: Determine start ID based on existing training set size
    # For Actin: 33 training cases -> start at 034
    # For Mito: 29 training cases -> start at 030
    
    if dataset_type == 'actin':
        start_id = 34
    else:
        start_id = 30
        
    mapping_data = []
    
    print(f"Generating {dataset_type} test set...")
    
    for subdir in subdirs:
        case_name = subdir.name
        
        # Skip if training case
        if case_name in training_cases:
            continue
            
        # File path
        nrrd_file = subdir / f"channel_{channel_id:02d}.nrrd"
        if not nrrd_file.exists():
             nrrd_file = subdir / f"channel_{channel_id}.nrrd"
             
        if nrrd_file.exists():
            # New ID
            new_id = f"{prefix}_{start_id:03d}"
            output_name = f"{new_id}_0000.nii.gz"
            output_path = images_ts_dir / output_name
            
            try:
                nrrd_to_nifti(nrrd_file, output_path)
                # print(f"Converted {case_name} -> {output_name}")
                
                mapping_data.append({
                    'case_id': new_id,
                    'original_name': case_name,
                    'type': 'test'
                })
                
                start_id += 1
                count += 1
            except Exception as e:
                print(f"Failed to convert {case_name}: {e}")

    # Save mapping
    mapping_csv = data_dir / f"{dataset_type}_test_mapping.csv"
    with open(mapping_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'original_name', 'type'])
        writer.writeheader()
        writer.writerows(mapping_data)
        
    print(f"[{dataset_type.upper()}] Generated {count} test cases in {images_ts_dir}")
    print(f"Saved mapping to {mapping_csv}\n")

if __name__ == "__main__":
    # generate_test_set('actin')
    generate_test_set('mito')
