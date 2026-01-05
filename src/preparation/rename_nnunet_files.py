#!/usr/bin/env python3
"""
Rename nnUNet dataset files to follow standard case_XXX naming convention
"""

import json
from pathlib import Path
import shutil

def rename_nnunet_files(dataset_dir):
    """
    Rename files in nnUNet dataset to use case_XXX naming convention
    
    Args:
        dataset_dir: Path to nnUNet dataset directory
    """
    dataset_path = Path(dataset_dir)
    
    # Load existing dataset.json
    dataset_json_path = dataset_path / "dataset.json"
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)
    
    # Create mapping from old names to case numbers
    training_cases = dataset_json['training']
    
    # Rename training files
    print("Renaming training files...")
    new_training = []
    
    for idx, case in enumerate(training_cases, start=1):
        case_num = f"{idx:03d}"  # Format as 001, 002, etc.
        
        # Extract old filenames
        old_image_path = dataset_path / case['image'].replace('./', '')
        old_label_path = dataset_path / case['label'].replace('./', '')
        
        # New filenames
        new_image_name = f"case_{case_num}_0000.nii.gz"
        new_label_name = f"case_{case_num}.nii.gz"
        
        new_image_path = dataset_path / "imagesTr" / new_image_name
        new_label_path = dataset_path / "labelsTr" / new_label_name
        
        # Rename files
        if old_image_path.exists():
            old_image_path.rename(new_image_path)
            print(f"  [{idx}/{len(training_cases)}] {old_image_path.name} → {new_image_name}")
        
        if old_label_path.exists():
            old_label_path.rename(new_label_path)
            print(f"  [{idx}/{len(training_cases)}] {old_label_path.name} → {new_label_name}")
        
        # Update reference in dataset.json
        new_training.append({
            "image": f"./imagesTr/{new_image_name}",
            "label": f"./labelsTr/{new_label_name}"
        })
    
    # Rename test files
    print("\nRenaming test files...")
    images_ts_dir = dataset_path / "imagesTs"
    test_files = sorted(images_ts_dir.glob("*.nii.gz"))
    
    # Start case numbers after training cases
    start_num = len(training_cases) + 1
    
    for idx, old_file in enumerate(test_files, start=start_num):
        case_num = f"{idx:03d}"
        new_name = f"case_{case_num}_0000.nii.gz"
        new_path = images_ts_dir / new_name
        
        old_file.rename(new_path)
        print(f"  [{idx-start_num+1}/{len(test_files)}] {old_file.name} → {new_name}")
    
    # Update dataset.json
    dataset_json['training'] = new_training
    
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print("\n" + "=" * 70)
    print("✓ Renaming complete!")
    print("=" * 70)
    print(f"Training cases: {len(training_cases)} (case_001 to case_{len(training_cases):03d})")
    print(f"Test cases: {len(test_files)} (case_{start_num:03d} to case_{start_num + len(test_files) - 1:03d})")
    print(f"\nUpdated: {dataset_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    dataset_directory = "/home/luisa/aneurysm_project/data/nnunet_data/actin_segmentation"
    
    print("=" * 70)
    print("RENAME nnUNet FILES TO STANDARD CONVENTION")
    print("=" * 70)
    print(f"Dataset directory: {dataset_directory}")
    print("=" * 70)
    print()
    
    rename_nnunet_files(dataset_directory)
