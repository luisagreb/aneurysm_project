
import os
import glob
import shutil
import json
import pandas as pd
import nrrd
import nibabel as nib
import numpy as np
from pathlib import Path
import re

# Configuration
DATA_ROOT = Path("/home/luisa/aneurysm_project/data")
RAW_SOURCE_DIR = DATA_ROOT / "raw_nucleus_segmented_manual"
BASE_OUTPUT = DATA_ROOT / "nnUNet/nnUNet_raw"
DATASET_ID = "Dataset003_Nucleus"
OUTPUT_DIR = BASE_OUTPUT / DATASET_ID

def convert_nrrd_to_nifti(nrrd_path, output_path, is_label=False):
    """Reads NRRD and saves as NIfTI, ensuring 3D."""
    try:
        data, header = nrrd.read(str(nrrd_path))
        
        # Handle 4D data (some segmentations might be 4D [1, x, y, z]?)
        if data.ndim == 4:
            data = data.squeeze() # Try to squeeze single dimensions
            
        if data.ndim != 3:
            print(f"Skipping {nrrd_path.name}: Dimensions {data.ndim} is not 3D")
            return False

        # Heuristic: If dim 0 is significantly smaller than others (likely Z-stack), transpose to (X, Y, Z)
        if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
            # Assume (Z, Y, X) -> Transpose to (X, Y, Z) = (2, 1, 0)
            data = np.transpose(data, (2, 1, 0))

        if is_label:
            data = (data > 0).astype(np.uint8)
        else:
            data = data.astype(np.float32)

        # Use identity affine for consistency with other datasets
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, output_path)
        return True
    except Exception as e:
        print(f"Error converting {nrrd_path}: {e}")
        return False

def generate_dataset_json(output_path, num_training_cases):
    json_dict = {
        "channel_names": {"0": "Nucleus"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
        "name": "NucleusSegmentation"
    }
    with open(output_path, 'w') as f:
        json.dump(json_dict, f, indent=4)

def main():
    print(f"Preparing Nucleus Data from: {RAW_SOURCE_DIR}")
    
    # Setup Output Directories
    imagesTr = OUTPUT_DIR / "imagesTr"
    labelsTr = OUTPUT_DIR / "labelsTr"
    
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)
    
    # Scan folders
    case_dirs = sorted([d for d in RAW_SOURCE_DIR.iterdir() if d.is_dir()])
    
    processed_count = 0
    mapping_data = []
    
    for case_dir in case_dirs:
        case_name = case_dir.name
        
        # 1. Look for Raw Image (channel_00.nrrd)
        raw_img = case_dir / "channel_00.nrrd"
        if not raw_img.exists():
            continue
            
        # Strategy: Strictly look for "Segmentation_*.seg.nrrd" (or .nrrd)
        all_files = list(case_dir.glob("*.nrrd"))
        # Filter for Segmentation_ and ensure not hidden
        seg_k_files = [f for f in all_files if "Segmentation_" in f.name and not f.name.startswith("._")]
        
        label_file = None
        label_type = "New/Added"
        
        if seg_k_files:
            # Pick the last one (highest k typically)
            seg_k_files.sort()
            label_file = seg_k_files[-1] 
            
        if label_file:
            # Load and Validate Shapes BEFORE conversion
            try:
                raw_data, _ = nrrd.read(str(raw_img))
                lbl_data, _ = nrrd.read(str(label_file))
                
                # Handle 4D squeeze if needed
                if raw_data.ndim == 4: raw_data = raw_data.squeeze()
                if lbl_data.ndim == 4: lbl_data = lbl_data.squeeze()
                
                # Check shapes (accounting for potential transposition)
                # If shapes are identical, good.
                # If shapes are transposed versions, also good (we will fix them).
                # If shapes are completely different (like 1x1x1 vs 100x100x10), SKIP.
                
                data_match = False
                if raw_data.shape == lbl_data.shape:
                    data_match = True
                elif raw_data.shape == lbl_data.shape[::-1]: # Roughly check reverse? Or permutation?
                    # If raw is (Z, Y, X) and label is (X, Y, Z), shapes might be (9, 1000, 1000) vs (1000, 1000, 9).
                    # Just sort dimensions and compare?
                    if sorted(raw_data.shape) == sorted(lbl_data.shape):
                        data_match = True
                
                if not data_match:
                    print(f"Skipping {case_name}: Shape Mismatch! Raw {raw_data.shape} vs Label {lbl_data.shape}")
                    continue
                    
            except Exception as e:
                print(f"Skipping {case_name}: Error reading files: {e}")
                continue

            case_id = f"Nucleus_{processed_count+1:04d}"
            
            dest_img = imagesTr / f"{case_id}_0000.nii.gz"
            dest_lbl = labelsTr / f"{case_id}.nii.gz"
            
            success_img = convert_nrrd_to_nifti(raw_img, dest_img, is_label=False)
            success_lbl = convert_nrrd_to_nifti(label_file, dest_lbl, is_label=True)
            
            if success_img and success_lbl:
                processed_count += 1
                mapping_data.append({
                    "CaseID": case_id,
                    "OriginalName": case_name,
                    "LabelOriginalName": label_file.name,
                    "Type": label_type
                })

    # Save Mapping
    if mapping_data:
        df = pd.DataFrame(mapping_data)
        df.to_csv(OUTPUT_DIR / "Nucleus_mapping.csv", index=False)
        generate_dataset_json(OUTPUT_DIR / "dataset.json", processed_count)
        print(f"\nDone! Processed {processed_count} Nucleus cases.")
        print(f"Output: {OUTPUT_DIR}")
    else:
        print("\nNo valid cases found!")

if __name__ == "__main__":
    main()
