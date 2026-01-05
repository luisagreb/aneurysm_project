#!/usr/bin/env python3
"""
Generate test data for Mitochondria (Dataset 002)
Extracts Channel 2 from unused cases in nrrd_files
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

def generate_mito_test_set():
    # Paths
    raw_data_dir = Path("/home/luisa/aneurysm_project/data/raw/nrrd_files")
    mito_data_dir = Path("/home/luisa/aneurysm_project/data/processed/nnUnet_mitochondria/Dataset002_Mito")
    
    # Create imagesTs if not exists
    images_ts_dir = mito_data_dir / "imagesTs"
    images_ts_dir.mkdir(exist_ok=True, parents=True)
    
    # Read existing mapping to skip training cases
    mapping_file = mito_data_dir.parent / "Mito_mapping.csv"
    training_original_names = set()
    
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                training_original_names.add(row['original_name'])
                
    print(f"Loaded {len(training_original_names)} training cases to skip.")
    
    # Scan raw files
    subdirs = sorted([d for d in raw_data_dir.iterdir() if d.is_dir()])
    
    mito_channel = 2  # Mitochondria is channel 2
    count = 0
    start_case_id = 30 # Dataset002 has 29 training cases
    
    mapping_data = []

    print(f"Scanning {len(subdirs)} folders...")
    
    for subdir in subdirs:
        case_name = subdir.name
        
        # Skip if it was in training set
        if case_name in training_original_names:
            continue
            
        # Look for channel 2
        mito_file = subdir / f"channel_{mito_channel:02d}.nrrd"
        if not mito_file.exists():
            # Try single digit naming just in case
            mito_file = subdir / f"channel_{mito_channel}.nrrd"
            
        if mito_file.exists():
            # New case ID
            new_case_name = f"Mito_{start_case_id:03d}"
            output_filename = f"{new_case_name}_0000.nii.gz"
            output_path = images_ts_dir / output_filename
            
            try:
                nrrd_to_nifti(mito_file, output_path)
                print(f"Generated: {output_filename} from {case_name}")
                
                mapping_data.append({
                    'case_id': new_case_name,
                    'original_name': case_name,
                    'type': 'test'
                })
                
                start_case_id += 1
                count += 1
            except Exception as e:
                print(f"Error converting {case_name}: {e}")
    
    # specific mapping for test set
    if mapping_data:
        test_mapping_file = mito_data_dir.parent / "Mito_test_mapping.csv"
        with open(test_mapping_file, 'w', newline='') as f:
            fieldnames = ['case_id', 'original_name', 'type']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mapping_data)
            print(f"\nSaved test mapping to {test_mapping_file}")

    print(f"\nDone! Generated {count} test cases in {images_ts_dir}")

if __name__ == "__main__":
    generate_mito_test_set()
