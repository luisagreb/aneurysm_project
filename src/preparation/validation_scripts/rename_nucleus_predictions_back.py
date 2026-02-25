from pathlib import Path
import pandas as pd
import nibabel as nib
import nrrd
import numpy as np

# ---- Paths ----
project_root = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project")

# 1) nnUNet predictions copied from GPU
pred_dir = project_root / "nnUnetV2" / "results_nnUNet_ch1"

# 2) Excel mapping file
mapping_path = project_root / "nnUnetV2" / "ch1_nnunet_case_mapping.xlsx"

# 3) Output directory with original names, NRRD only
out_dir = project_root / "data" / "nucleus_segmented_named"
out_dir.mkdir(parents=True, exist_ok=True)

print("Predictions dir :", pred_dir)
print("Mapping file    :", mapping_path)
print("Output dir      :", out_dir)

# ---- Load mapping from Excel ----
df = pd.read_excel(mapping_path)
# Must have columns: nnunet_case, raw_filename
case_to_orig = dict(zip(df["nnunet_case"], df["raw_filename"]))

missing_preds = []
converted = 0

for case_id, orig_name in sorted(case_to_orig.items()):
    # Prediction file from nnUNet
    pred_path = pred_dir / f"{case_id}.nii.gz"
    if not pred_path.exists():
        alt = pred_dir / f"{case_id}_0000.nii.gz"
        if alt.exists():
            pred_path = alt
        else:
            print(f"[MISSING PRED] {case_id}")
            missing_preds.append(case_id)
            continue

    # Load NIfTI segmentation
    img = nib.load(str(pred_path))
    data = img.get_fdata()

    # For label maps: cast to uint8 (0,1 labels)
    data = data.astype(np.uint8)

    # Build output NRRD name from original raw file name
    # e.g. "…nucleus_cell1.nii.gz" -> "…nucleus_cell1.nrrd"
    nrrd_name = orig_name.replace(".nii.gz", ".nrrd")
    nrrd_out_path = out_dir / nrrd_name

    # Write simple NRRD (library fills basic header: dimension, sizes, type)
    nrrd.write(str(nrrd_out_path), data)

    converted += 1
    print(f"[OK] {case_id} -> {nrrd_name}")

print(f"\nConverted to NRRD: {converted}")
if missing_preds:
    print("Missing predictions for cases:", missing_preds)