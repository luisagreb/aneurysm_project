#!/usr/bin/env python3
"""
Create dataset.json for nnUNet V2

This file describes the dataset for nnUNet training.
Run this script in the same directory as imagesTr, labelsTr, imagesTs folders.

Usage:
    python create_dataset_json.py
"""

import json
from pathlib import Path


def create_dataset_json(output_dir, dataset_name="Dataset001_ActinSegmentation", 
                        description="Actin fiber segmentation from microscopy images"):
    """
    Create dataset.json for nnUNet V2
    
    Args:
        output_dir: Directory containing imagesTr, labelsTr, imagesTs
        dataset_name: Name of the dataset
        description: Description of the dataset
    """
    output_path = Path(output_dir)
    
    # Count training cases
    images_tr_dir = output_path / "imagesTr"
    
    if not images_tr_dir.exists():
        print(f"Error: {images_tr_dir} does not exist")
        return
    
    # Get all training images
    training_images = sorted(images_tr_dir.glob("*_0000.nii.gz"))
    
    # Extract case identifiers
    training_cases = []
    for img_path in training_images:
        # Remove the _0000.nii.gz suffix to get case name
        case_name = img_path.name.replace("_0000.nii.gz", "")
        training_cases.append(case_name)
    
    print(f"Found {len(training_cases)} training cases")
    
    # Create dataset.json structure
    dataset_json = {
        "name": dataset_name,
        "description": description,
        "reference": "Custom dataset for actin segmentation",
        "licence": "private",
        "release": "1.0",
        
        # Channel names - we only have 1 channel (actin)
        "channel_names": {
            "0": "actin"
        },
        
        # Number of channels for training (modalities)
        "numTraining": len(training_cases),
        
        # Label classes
        # 0 is always background
        # 1 is the foreground (actin fibers)
        "labels": {
            "background": 0,
            "actin": 1
        },
        
        # File endings
        "file_ending": ".nii.gz",
        
        # Training cases
        # Format: {"image": "./imagesTr/case_001.nii.gz", "label": "./labelsTr/case_001.nii.gz"}
        # Note: nnUNet V2 expects relative paths starting with ./
        # Note: Image should NOT include the _0000 suffix in this dict
        "training": [
            {
                "image": f"./imagesTr/{case_name}.nii.gz",
                "label": f"./labelsTr/{case_name}.nii.gz"
            }
            for case_name in training_cases
        ]
    }
    
    # Save dataset.json
    json_path = output_path / "dataset.json"
    
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"✓ Created: {json_path}")
    print()
    print("Dataset configuration:")
    print(f"  Name: {dataset_name}")
    print(f"  Training cases: {len(training_cases)}")
    print(f"  Channels: {list(dataset_json['channel_names'].values())}")
    print(f"  Labels: {list(dataset_json['labels'].keys())}")
    print()
    print("Next steps:")
    print("1. Copy this entire directory to your nnUNet raw data directory")
    print("   Example: nnUNet_raw/Dataset001_ActinSegmentation/")
    print("2. Set nnUNet environment variables:")
    print("   export nnUNet_raw='/path/to/nnUNet_raw'")
    print("   export nnUNet_preprocessed='/path/to/nnUNet_preprocessed'")
    print("   export nnUNet_results='/path/to/nnUNet_results'")
    print("3. Run nnUNet planning and preprocessing:")
    print(f"   nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity")
    print("4. Train the model:")
    print("   nnUNetv2_train 001 3d_fullres 0")


if __name__ == "__main__":
    # Configuration
    # This should be the directory containing imagesTr, labelsTr, imagesTs
    output_directory = "/home/luisa/aneurysm_project/data/nnunet_data/actin_segmentation"
    
    print("=" * 70)
    print("CREATE DATASET.JSON FOR nnUNet V2")
    print("=" * 70)
    print(f"Output directory: {output_directory}")
    print("=" * 70)
    print()
    
    create_dataset_json(
        output_dir=output_directory,
        dataset_name="Dataset001_ActinSegmentation",
        description="Actin fiber segmentation from 3D microscopy images"
    )
