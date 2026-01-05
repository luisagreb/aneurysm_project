
import os
import glob
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
RAW_DIR = DATA_ROOT / "raw/nrrd_files"

# Dataset Definitions
DATASETS = [
    {
        "name": "ActinSegmentation",
        "id": "Dataset001_Actin",
        "output_folder": "nnUNet/nnUNet_raw", 
        "raw_channel": "channel_01.nrrd",
        "label_dir_name": "raw/manually_segmented_actin",
        "prefix": "Actin"
    },
    {
        "name": "MitoSegmentation",
        "id": "Dataset002_Mito",
        "output_folder": "nnUNet/nnUNet_raw", 
        "raw_channel": "channel_02.nrrd",
        "label_dir_name": "raw/manually_segmented_mitochondria",
        "prefix": "Mito"
    }
]

def normalize_name(name):
    """Normalize string for robust comparison: lowercase, strip whitespace, unicode normalize."""
    name = unicodedata.normalize('NFC', name)
    return name.lower().strip().replace(" ", "")

def convert_nrrd_to_nifti(nrrd_path, output_path, is_label=False):
    """Reads NRRD and saves as NIfTI, ensuring 3D."""
    data, header = nrrd.read(nrrd_path)
    if data.ndim != 3:
        raise ValueError(f"File {nrrd_path} is not 3D! Dimensions: {data.ndim}")
    
    # Heuristic: If dim 0 is significantly smaller than others (likely Z-stack), transpose to (X, Y, Z)
    if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
        # Assume (Z, Y, X) -> Transpose to (X, Y, Z) = (2, 1, 0)
        print(f"  Fixing orientation: {data.shape} -> Transposing to (X, Y, Z)")
        data = np.transpose(data, (2, 1, 0))

    if is_label:
        # Binarize: Set all non-zero values to 1
        data = (data > 0).astype(np.uint8)
    else:
        # Images: Keep as float32 usually, or original
        data = data.astype(np.float32)

    img = nib.Nifti1Image(data, np.eye(4))
    if is_label:
        img.header.set_data_dtype(np.uint8)
    nib.save(img, output_path)

def generate_dataset_json(output_path, dataset_name, num_training_cases):
    json_dict = {
        "channel_names": {"0": "Microscopy"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
        "name": dataset_name
    }
    with open(output_path, 'w') as f:
        json.dump(json_dict, f, indent=4)

def process_dataset(ds_config):
    print(f"\n--- Processing {ds_config['name']} ---")
    
    label_source_dir = DATA_ROOT / ds_config['label_dir_name']
    base_output = DATA_ROOT / ds_config['output_folder']
    dataset_ptr = base_output / ds_config['id']
    
    imagesTr = dataset_ptr / "imagesTr"
    labelsTr = dataset_ptr / "labelsTr"
    
    if dataset_ptr.exists():
        shutil.rmtree(dataset_ptr)
    
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)
    
    case_mapping = []
    case_counter = 1
    
    # 1. Gather Candidates
    raw_cells = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    label_files = sorted(list(label_source_dir.glob("*.nrrd")))
    
    # Track usage
    used_raw_paths = set()
    used_label_paths = set()
    
    matches = [] # List of (raw_path, label_path, cell_name)

    # Explicit exclusions for invalid files (e.g. shape mismatch)
    EXCLUDED_LABELS = {
        "03Asc46 +coll 60x veh 48h-Zstack cell1.nrrd" # Mismatch: Label (1023x961) vs Raw (964x990)
    }
    label_files = [f for f in label_files if f.name not in EXCLUDED_LABELS]

    # --- PASS 1: Strict/Normalized Match ---
    # Map normalized label names for quick lookup
    label_map = {} 
    for lf in label_files:
        # Handle cases like "name.seg.nrrd" -> "name"
        stem = lf.name.replace(".seg.nrrd", "").replace(".nrrd", "") 
        norm = normalize_name(stem)
        label_map[norm] = lf

    for cell_dir in raw_cells:
        raw_path = cell_dir / ds_config['raw_channel']
        if not raw_path.exists():
            continue
            
        norm_cell = normalize_name(cell_dir.name)
        if norm_cell in label_map:
            label_path = label_map[norm_cell]
            matches.append((raw_path, label_path, cell_dir.name))
            used_raw_paths.add(raw_path)
            used_label_paths.add(label_path)
            # Remove from map to prevent double usage? No, keeping set is enough.
    
    print(f"Pass 1 (strict/norm): Found {len(matches)} matches.")
    
    # --- PASS 2: Fuzzy / Manual Match ---
    
    # User-defined Manual Overrides (Normalized Stem -> Normalized Raw Folder Name)
    # We use partial strings or full normalized names to resolve ambiguity.
    MANUAL_OVERRIDES = {
        # Actin Corrections
        "01asc-230col60xveh-zstackcell2": "01asc-230+col60xveh-zstackcell2",
        "01asc-230col60xveh-zstackcell4": "01asc-230+col60xveh-zstackcell4",
        "01asc-230col60xveh-zstackcell5": "01asc-230+col60xveh-zstackcell5",
        "03asc46coll60xveh48h-zstackcell2": "03asc46+coll60xveh48h-zstackcell2",
        "03asc46coll60xveh48h-zstackcell3": "03asc46+coll60xveh48h-zstackcell3",
        "03asc46coll60xveh48h-zstackcell4": "03asc46+coll60xveh48h-zstackcell4", # Fixed from -coll
        "03asc46coll60xveh48h-zstackcell5": "03asc46+coll60xveh48h-zstackcell5",
        
        # Mito Corrections
        "03asc-0057col60xveh-zstackcell2": "03asc-0057+col60xveh-zstackcell2",
        "03asc-0057col60xveh-zstackcell4": "03asc-0057+col60xveh-zstackcell4", # Duplicate of +Col
        "03asc-0057col60xveh-zstackcell5": "03asc-0057+col60xveh-zstackcell5",
        
        # Nucleus Corrections (Assuming similar patterns)
        # Add any known Nucleus-specific overrides here if needed

    }
    
    # Explicit exclusions for invalid files (e.g. shape mismatch)
    EXCLUDED_LABELS = {
        "03Asc46 +coll 60x veh 48h-Zstack cell1.nrrd" # Mismatch: Label (1023x961) vs Raw (964x990)
    }
    
    unused_raw = [d for d in raw_cells if (d / ds_config['raw_channel']) not in used_raw_paths and (d / ds_config['raw_channel']).exists()]
    unused_labels = [lf for lf in label_files if lf not in used_label_paths and lf.name not in EXCLUDED_LABELS]
    
    print(f"Pass 2: Checking {len(unused_labels)} unused labels against {len(unused_raw)} unused raw folders...")
    
    for lf in unused_labels:
        stem = lf.name.replace(".seg.nrrd", "").replace(".nrrd", "")
        norm_lbl = normalize_name(stem)
        
        best_candidate = None
        
        # 2a. Check Manual Overrides first
        if norm_lbl in MANUAL_OVERRIDES:
            target_norm = MANUAL_OVERRIDES[norm_lbl]
            # Find the raw folder that matches this target
            for cell_dir in unused_raw:
                if normalize_name(cell_dir.name) == target_norm:
                    best_candidate = (cell_dir / ds_config['raw_channel'], cell_dir.name)
                    print(f"  Manual Override: {lf.name} -> {cell_dir.name}")
                    break
            
            # If specified in override, do NOT fall back to fuzzy search.
            # If target was not found (e.g. already used), this label is a duplicate/orphan to be skipped.
            if not best_candidate:
                # print(f"  Skipping {lf.name} (Override target used or not found)")
                pass
        
        # 2b. Fuzzy Search (only if NO override was defined)
        else:
            for cell_dir in unused_raw:
                raw_path = cell_dir / ds_config['raw_channel']
                if raw_path in used_raw_paths: continue 
                
                norm_raw = normalize_name(cell_dir.name)
                
                # Fuzzy Logic: substring match ignoring +/-
                n_lbl_clean = norm_lbl.replace("+", "").replace("-", "")
                n_raw_clean = norm_raw.replace("+", "").replace("-", "")
                
                if n_lbl_clean in n_raw_clean or n_raw_clean in n_lbl_clean:
                    best_candidate = (raw_path, cell_dir.name)
                    break 
        
        if best_candidate:
            raw_path, cell_name = best_candidate
            matches.append((raw_path, lf, cell_name))
            used_raw_paths.add(raw_path)
            used_label_paths.add(lf)
            print(f"  Fuzzy Match: {lf.name} -> {cell_name}")

    print(f"Total Matches to Process: {len(matches)}")

    # --- EXECUTE ---
    for raw_path, label_path, cell_name in matches:
        try:
            case_id = f"{ds_config['prefix']}_{case_counter:04d}"
            out_img = imagesTr / f"{case_id}_0000.nii.gz"
            out_lbl = labelsTr / f"{case_id}.nii.gz"
            
            convert_nrrd_to_nifti(raw_path, out_img, is_label=False)
            convert_nrrd_to_nifti(label_path, out_lbl, is_label=True)
            
            case_mapping.append({
                "CaseID": case_id,
                "OriginalCellName": cell_name,
                "RawPath": str(raw_path),
                "LabelPath": str(label_path)
            })
            case_counter += 1
            print(f"Processed {case_id}: {cell_name}")
            
        except Exception as e:
            print(f"ERROR converting {cell_name}: {e}")

    # Finalize
    mapping_file = base_output / f"{ds_config['prefix']}_mapping.csv"
    pd.DataFrame(case_mapping).to_csv(mapping_file, index=False)
    generate_dataset_json(dataset_ptr / "dataset.json", ds_config['name'], len(case_mapping))
    print(f"Completed {ds_config['name']}: {len(case_mapping)} cases.")

def main():
    print("Starting Data Preparation (2-Pass Matching)...")
    if not RAW_DIR.exists(): return
    for ds in DATASETS:
        process_dataset(ds)
    print("\nAll Done!")

if __name__ == "__main__":
    main()
