
import os
import shutil
import json
import pandas as pd
import nrrd
import nibabel as nib
import numpy as np
from pathlib import Path
import unicodedata

# Configuration
DATA_ROOT = Path("/home/luisa/aneurysm_project/data")
RAW_NRRD_DIR = DATA_ROOT / "raw/nrrd_files"
RAW_NUCLEUS_DIR = DATA_ROOT / "raw/raw_nucleus_segmented_manual"
NNUNET_RAW = DATA_ROOT / "nnUNet/nnUNet_raw"

DATASETS = [
    {
        "name": "Actin",
        "id": "Dataset001_Actin",
        "type": "actin_mito",
        "raw_dir": RAW_NRRD_DIR,
        "channel": "channel_01.nrrd",
        "channel_alt": "channel_1.nrrd",
        "mapping_file": NNUNET_RAW / "Actin_mapping.csv",
        "mapping_key": "OriginalCellName",
        "prefix": "Actin_Test"
    },
    {
        "name": "Mito",
        "id": "Dataset002_Mito",
        "type": "actin_mito",
        "raw_dir": RAW_NRRD_DIR,
        "channel": "channel_02.nrrd",
        "channel_alt": "channel_2.nrrd",
        "mapping_file": NNUNET_RAW / "Mito_mapping.csv",
        "mapping_key": "OriginalCellName",
        "prefix": "Mito_Test"
    },
    {
        "name": "Nucleus",
        "id": "Dataset003_Nucleus",
        "type": "nucleus",
        "raw_dir": RAW_NUCLEUS_DIR,
        "channel": "channel_00.nrrd",
        "channel_alt": None,
        "mapping_file": NNUNET_RAW / "Dataset003_Nucleus/Nucleus_mapping.csv",
        "mapping_key": "OriginalName",
        "prefix": "Nucleus_Test"
    }
]

def normalize_name(name):
    return unicodedata.normalize('NFC', name).lower().strip().replace(" ", "")

def convert_nrrd_to_nifti(nrrd_path, output_path):
    try:
        data, header = nrrd.read(str(nrrd_path))
        
        # Handle 4D data if present
        if data.ndim == 4:
            data = data.squeeze()
            
        if data.ndim != 3:
            print(f"  Skipping {nrrd_path.name}: Dimensions {data.ndim} is not 3D")
            return False

        # Heuristic: If dim 0 is significantly smaller (likely Z-stack), transpose to (X, Y, Z)
        # This matches the training data preparation logic exactly.
        if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
            # print(f"  Fixing orientation: {data.shape} -> Transposing to (X, Y, Z)")
            data = np.transpose(data, (2, 1, 0))

        data = data.astype(np.float32)
        
        # Use identity affine for consistency
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, output_path)
        return True
    except Exception as e:
        print(f"  Error converting {nrrd_path}: {e}")
        return False

def generate_test_data():
    print("Generating Test Data (imagesTs) for V2...")
    
    for ds in DATASETS:
        print(f"\n--- Processing {ds['name']} ---")
        
        # 1. Load Exclusions (Training Cases)
        exclusions = set()
        if ds['mapping_file'].exists():
            df = pd.read_csv(ds['mapping_file'])
            key = ds['mapping_key']
            if key in df.columns:
                exclusions = set(df[key].apply(normalize_name))
            else:
                print(f"  Warning: Key '{key}' not found in mapping file.")
        else:
             print(f"  Warning: Mapping file not found: {ds['mapping_file']}")
        
        print(f"  Loaded {len(exclusions)} training cases to exclude.")
        
        # 2. Setup Output Directory
        output_dir = NNUNET_RAW / ds['id'] / "imagesTs"
        if output_dir.exists():
            # Optional: Clean up? Or keep? Let's clean to ensure fresh set
            pass # shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Iterate Candidates
        raw_folders = sorted([d for d in ds['raw_dir'].iterdir() if d.is_dir()])
        
        count = 0
        new_mapping = []
        
        start_id = 1
        
        for folder in raw_folders:
            case_name = folder.name
            norm_name = normalize_name(case_name)
            
            # CHECK EXCLUSION
            if norm_name in exclusions:
                continue
                
            # FIND RAW FILE
            raw_file = folder / ds['channel']
            if not raw_file.exists() and ds['channel_alt']:
                 raw_file = folder / ds['channel_alt']
            
            if raw_file.exists():
                case_id = f"{ds['prefix']}_{start_id:04d}"
                out_path = output_dir / f"{case_id}_0000.nii.gz"
                
                if convert_nrrd_to_nifti(raw_file, out_path):
                    count += 1
                    start_id += 1
                    new_mapping.append({
                        "TestID": case_id,
                        "OriginalName": case_name
                    })
        
        # Save Test Mapping
        if new_mapping:
            mapping_df = pd.DataFrame(new_mapping)
            mapping_df.to_csv(NNUNET_RAW / ds['id'] / "test_mapping.csv", index=False)
            print(f"  Generated {count} test cases.")
            print(f"  Saved mapping to {NNUNET_RAW / ds['id'] / 'test_mapping.csv'}")
        else:
            print("  No new test cases found.")

if __name__ == "__main__":
    generate_test_data()
