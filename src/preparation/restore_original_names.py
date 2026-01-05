
import os
import csv
import shutil
from pathlib import Path

# Paths
V1_MAPPINGS = {
    "actin": "/home/luisa/aneurysm_project/experiments/V1/processed_V1/nnunet_data/actin_segmentation/actin_test_mapping.csv",
    "mito": "/home/luisa/aneurysm_project/experiments/V1/processed_V1/nnUnet_mitochondria/Dataset002_Mito/mito_test_mapping.csv"
}

DIRS = {
    "actin": Path("/home/luisa/aneurysm_project/data/raw/manually_segmented_actin"),
    "mito": Path("/home/luisa/aneurysm_project/data/raw/manually_segmented_mitochondria")
}

def load_mapping(csv_path):
    mapping = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Mapping file not found: {csv_path}")
        return mapping
        
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # map case_id -> original_name
            # CSV columns: case_id,original_name,type
            cid = row['case_id']
            orig = row['original_name']
            
            # Handle potential mismatch in V1 IDs vs filenames?
            # case_id in csv is "case_034", file might be "case_034.nrrd"
            mapping[cid] = orig
    return mapping

def restore_names(dataset_type):
    print(f"\nRestoring names for {dataset_type}...")
    folder = DIRS[dataset_type]
    mapping = load_mapping(V1_MAPPINGS[dataset_type])
    
    if not mapping:
        return

    # Scan folder
    files = list(folder.glob("*.nrrd"))
    count = 0
    
    for f in files:
        # Check if filename corresponds to a case ID
        stem = f.stem # e.g. "case_034"
        
        # For actin, prefix is "case_", for mito "Mito_"
        # Mapping file has full ID "case_034" or "Mito_030"
        
        if stem in mapping:
            original_name = mapping[stem]
            dest_name = f"{original_name}.seg.nrrd" # Adding .seg to distinguish? Or just .nrrd?
            # Existing manual labels in folders often have no suffix or .seg.nrrd
            # Let's check if target already exists to avoid overwriting?
            
            dest_path = folder / dest_name
            
            if dest_path.exists():
                print(f"Skipping {f.name}: Target {dest_name} already exists.")
            else:
                print(f"Renaming {f.name} -> {dest_name}")
                f.rename(dest_path)
                count += 1
        else:
            # print(f"Skipping {f.name}: Not found in mapping (or already named correctly)")
            pass
            
    print(f"Renamed {count} files.")

if __name__ == "__main__":
    restore_names("actin")
    restore_names("mito")
