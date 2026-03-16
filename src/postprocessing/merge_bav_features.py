"""
merge_bav_features.py
---------------------
Merges the per-organelle feature CSVs for new BAV cells into the main
Advanced_Features_Raw_Final.csv dataset.

Inputs:
  - outputs/new_BAV_features/{Actin,Mito,Nucleus}_features.csv   (81 rows each)
  - outputs/new_BAV_inference_mapping.csv                          (cell → case ID)
  - outputs/Advanced_Features_Raw_Final.csv                        (existing 200-row dataset)

Output:
  - outputs/Advanced_Features_Raw_Final.csv  (updated, 200 + 81 = 281 rows)
  - outputs/Advanced_Features_Raw_BAV_only.csv  (just the 81 new rows, for inspection)
"""

import pandas as pd
from pathlib import Path

ROOT = Path("/home/luisa/aneurysm_project")

# ── Load mapping (cell path → case filenames) ──────────────────────────────
mapping = pd.read_csv(ROOT / "outputs/new_BAV_inference_mapping.csv")

def case_id(filename_with_modality: str) -> str:
    """'Actin_Test_0001_0000.nii.gz' → 'Actin_Test_0001.nii.gz'"""
    return filename_with_modality.replace("_0000.nii.gz", ".nii.gz")

mapping["Actin_fn"]   = mapping["Actin_case"].apply(case_id)
mapping["Mito_fn"]    = mapping["Mito_case"].apply(case_id)
mapping["Nucleus_fn"] = mapping["Nucleus_case"].apply(case_id)

# ── Load per-organelle feature CSVs ────────────────────────────────────────
actin_df   = pd.read_csv(ROOT / "outputs/new_BAV_features/Actin_features.csv")
mito_df    = pd.read_csv(ROOT / "outputs/new_BAV_features/Mito_features.csv")
nucleus_df = pd.read_csv(ROOT / "outputs/new_BAV_features/Nucleus_features.csv")

# Index by Filename for fast lookup
actin_df   = actin_df.set_index("Filename")
mito_df    = mito_df.set_index("Filename")
nucleus_df = nucleus_df.set_index("Filename")

# ── Column rename maps (new name → existing dataset name) ─────────────────
ACTIN_COLS = {
    "Volume":                "Actin_Volume_µm³",
    "Skeleton_Length":       "Actin_Skeleton_Length_µm",
    "Convex_Hull_Volume":    "Actin_Convex_Hull_Volume_µm³",
    "Solidity":              "Actin_Solidity_ratio",
    "Extent":                "Actin_Extent_ratio",
    "Fractional_Anisotropy": "Actin_Fractional_Anisotropy_ratio",
    "Major_Axis":            "Actin_Major_Axis_µm",
    "Intermediate_Axis":     "Actin_Intermediate_Axis_µm",
    "Minor_Axis":            "Actin_Minor_Axis_µm",
}
MITO_COLS = {
    "Volume":                   "Mito_Volume_µm³",
    "Surface_Area":             "Mito_Surface_Area_µm²",
    "Sphericity":               "Mito_Sphericity_ratio",
    "Fragment_Count":           "Mito_Fragment_Count_n",
    "Mean_Fragment_Sphericity": "Mito_Mean_Fragment_Sphericity_ratio",
    "Std_Fragment_Sphericity":  "Mito_Std_Fragment_Sphericity_ratio",
    "Min_Fragment_Sphericity":  "Mito_Min_Fragment_Sphericity_ratio",
    "Max_Fragment_Sphericity":  "Mito_Max_Fragment_Sphericity_ratio",
    "Mean_Fragment_Volume":     "Mito_Mean_Fragment_Volume_µm³",
    "Junction_Count":           "Mito_Junction_Count_n",
    "Branch_Count":             "Mito_Branch_Count_n",
    "Mean_Branch_Length":       "Mito_Mean_Branch_Length_µm",
    "Total_Network_Length":     "Mito_Total_Network_Length_µm",
    "Mean_Tortuosity":          "Mito_Mean_Tortuosity_ratio",
    "Cyclomatic_Number":        "Mito_Cyclomatic_Number_n",
}
NUCLEUS_COLS = {
    "Volume":      "Nucleus_Volume_µm³",
    "Sphericity":  "Nucleus_Sphericity_ratio",
    "Elongation":  "Nucleus_Elongation_ratio",
    "Flatness":    "Nucleus_Flatness_ratio",
    "Solidity":    "Nucleus_Solidity_ratio",
    # New column — will be NaN in existing rows
    "Circularity": "Nucleus_Circularity_ratio",
}

# ── Build merged rows ──────────────────────────────────────────────────────
rows = []
skipped = []

for _, row in mapping.iterrows():
    cell_name = str(row["cell"])
    af = row["Actin_fn"]
    mf = row["Mito_fn"]
    nf = row["Nucleus_fn"]

    if af not in actin_df.index or mf not in mito_df.index or nf not in nucleus_df.index:
        print(f"  ⚠  SKIP {cell_name}: missing segmentation features")
        skipped.append(cell_name)
        continue

    a = actin_df.loc[af]
    m = mito_df.loc[mf]
    n = nucleus_df.loc[nf]

    record = {
        "CellName":   cell_name,
        "DataSource": "BAV",
        "Voxel_Z_µm": a["Voxel_Z"],
        "Voxel_Y_µm": a["Voxel_Y"],
        "Voxel_X_µm": a["Voxel_X"],
    }
    for src, dst in ACTIN_COLS.items():
        record[dst] = a.get(src, float("nan"))
    for src, dst in MITO_COLS.items():
        record[dst] = m.get(src, float("nan"))
    for src, dst in NUCLEUS_COLS.items():
        record[dst] = n.get(src, float("nan"))

    rows.append(record)

bav_df = pd.DataFrame(rows)
print(f"Built {len(bav_df)} BAV rows  ({len(skipped)} skipped)")

# ── Save BAV-only CSV for inspection ──────────────────────────────────────
bav_df.to_csv(ROOT / "outputs/Advanced_Features_BAV_only.csv", index=False)
print("Saved: outputs/Advanced_Features_BAV_only.csv")

# ── Append to existing master dataset ─────────────────────────────────────
existing_path = ROOT / "outputs/Advanced_Features_Raw_Final.csv"
existing = pd.read_csv(existing_path)
print(f"Existing dataset: {len(existing)} rows")

combined = pd.concat([existing, bav_df], ignore_index=True, sort=False)
combined.to_csv(existing_path, index=False)
print(f"Updated {existing_path.name}: {len(combined)} rows total")
print(f"  DataSource breakdown:\n{combined['DataSource'].value_counts().to_string()}")
