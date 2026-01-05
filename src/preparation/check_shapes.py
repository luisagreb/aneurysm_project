
import pandas as pd
import nrrd
from pathlib import Path

DATA_ROOT = Path("/home/luisa/aneurysm_project/data")
NNUNET_MAPPING_ACTIN = DATA_ROOT / "nnUnet_actin/Actin_mapping.csv"
NNUNET_MAPPING_MITO = DATA_ROOT / "nnUnet_mitochondria/Mito_mapping.csv"

def check_dataset(csv_path, dataset_name):
    print(f"\nChecking {dataset_name} shapes...")
    if not csv_path.exists():
        print(f"Mapping file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    mismatches = []
    
    for idx, row in df.iterrows():
        raw_path = Path(row['RawPath'])
        label_path = Path(row['LabelPath'])
        
        if not raw_path.exists():
            print(f"Missing Raw: {raw_path}")
            continue
        if not label_path.exists():
            print(f"Missing Label: {label_path}")
            continue
            
        try:
            # Read unique headers only for speed if possible, but pynrrd read reads data by default
            # We can use read_header only if we want speed, but let's just read
            h_raw = nrrd.read_header(str(raw_path))
            h_lbl = nrrd.read_header(str(label_path))
            
            shape_raw = tuple(h_raw['sizes'])
            shape_lbl = tuple(h_lbl['sizes'])
            
            if shape_raw != shape_lbl:
                print(f"MISMATCH {row['CaseID']}:")
                print(f"  Raw:   {shape_raw} ({raw_path.name})")
                print(f"  Label: {shape_lbl} ({label_path.name})")
                mismatches.append(row['CaseID'])
        except Exception as e:
            print(f"Error checking {row['CaseID']}: {e}")

    print(f"Found {len(mismatches)} mismatches in {dataset_name}.")

def main():
    check_dataset(NNUNET_MAPPING_ACTIN, "Actin")
    check_dataset(NNUNET_MAPPING_MITO, "Mito")

if __name__ == "__main__":
    main()
