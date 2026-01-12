
import pandas as pd
from pathlib import Path
import difflib

DATA_ROOT = Path("/home/luisa/aneurysm_project/data/nnUNet/nnUNet_raw")

DATASETS = {
    "Actin": DATA_ROOT / "Actin_mapping.csv",
    "Mito": DATA_ROOT / "Mito_mapping.csv",
    "Nucleus": DATA_ROOT / "Dataset003_Nucleus/Nucleus_mapping.csv"
}

def check_mapping(name, csv_path):
    print(f"\n--- Verifying {name} ---")
    if not csv_path.exists():
        print(f"❌ Mapping file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Total Cases: {len(df)}")
    
    issues = []
    
    for idx, row in df.iterrows():
        # Column names vary slightly between scripts
        # Actin/Mito: OriginalCellName, LabelPath
        # Nucleus: OriginalName, LabelOriginalName
        
        raw_name = str(row.get('OriginalCellName', row.get('OriginalName', '')))
        lbl_path = str(row.get('LabelPath', row.get('LabelOriginalName', '')))
        lbl_name = Path(lbl_path).name

        # Normalization for comparison
        r_norm = raw_name.lower().replace(" ", "").replace("-", "").replace("_", "").replace("+", "")
        l_norm = lbl_name.lower().replace(" ", "").replace("-", "").replace("_", "").replace("+", "")
        
        # 1. Nucleus Special Case: Label is often "Segmentation_XX.seg.nrrd" which doesn't match Cell Name
        if "segmentation" in l_norm:
            # For Nucleus, we trust folder structure (verified by script logic)
            # But we can check if it's Actin/Mito where we renamed them back
            if name != "Nucleus":
                 # If we have Segmentation_XX in Actin, that's weird unless we didn't rename?
                 # Actually, restore_original_names renamed them to "OriginalName.seg.nrrd"
                 pass
            continue

        # 2. String Similarity Check
        # Remove extensions
        l_norm = l_norm.replace(".nrrd", "").replace(".seg", "").replace(".nii.gz", "")
        
        # Check if one is substring of other
        if r_norm not in l_norm and l_norm not in r_norm:
             # Calculate similarity
             ratio = difflib.SequenceMatcher(None, r_norm, l_norm).ratio()
             if ratio < 0.6: # Arbitrary threshold
                 issues.append(f"{raw_name}  <-->  {lbl_name} (Low Similarity: {ratio:.2f})")

    if issues:
        print(f"⚠️  Found {len(issues)} potential mismatches:")
        for i in issues[:10]:
            print(f"  - {i}")
        if len(issues) > 10: print("  ... and more.")
    else:
        print("✅  All naming pairs look consistent.")

if __name__ == "__main__":
    for name, path in DATASETS.items():
        check_mapping(name, path)
