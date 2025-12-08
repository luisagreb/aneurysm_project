import re
import shutil
from pathlib import Path
from typing import Optional
import pandas as pd

# ---------------- CONFIG ----------------

# Project root
project_root = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project")

# 1) All subject folders live here
smc_root = project_root / "SMCs_Zstacks"

# 2) Where to gather all ch1 nucleus volumes (nice human names)
collect_dir = project_root / "data" / "raw_ch1_nucleus_all"

# 3) Where to write nnU-Net-ready inputs
nnunet_dir = project_root / "data" / "ch1_nnunet"

# 4) Mapping CSV for later analysis
mapping_csv = project_root / "data" / "nnunet_case_mapping.csv"

# 5) Excel file with aneurysmal vs non-aneurysmal subjects
#    Put Book1_FR.xlsx at the root of your project or update this path.
excel_path = project_root / "Book1_FR.xlsx"

collect_dir.mkdir(parents=True, exist_ok=True)
nnunet_dir.mkdir(parents=True, exist_ok=True)
mapping_csv.parent.mkdir(parents=True, exist_ok=True)

print("SMCs_Zstacks root :", smc_root)
print("Collect dir       :", collect_dir)
print("nnU-Net dir       :", nnunet_dir)
print("Excel path        :", excel_path)


# ---------------- LOAD ANEURYSM / NON-ANEURYSM FROM EXCEL ----------------

df = pd.read_excel(excel_path)

# Find the row where the two group headers are
idx = df[df.iloc[:, 0].astype(str).str.strip() == "Non-aneurysmal"].index[0]

non_aneur = (
    df.iloc[idx + 1 :, 0].dropna().astype(str).str.strip().tolist()
)  # column 0
aneur = (
    df.iloc[idx + 1 :, 1].dropna().astype(str).str.strip().tolist()
)  # column 1

non_aneur_set = set(non_aneur)
aneur_set = set(aneur)

print("\nNon-aneurysmal subjects from Excel:", sorted(non_aneur_set))
print("Aneurysmal subjects from Excel     :", sorted(aneur_set))


def get_aneurysm_status(subject_id: str) -> str:
    """Return 'Aneurysm' or 'nonAneurysm' based on Excel, warn if unknown."""
    sid = subject_id.strip()
    if sid in non_aneur_set:
        return "nonAneurysm"
    if sid in aneur_set:
        return "Aneurysm"
    print(f"⚠️  Subject {sid} not found in Excel lists, defaulting to 'nonAneurysm'")
    return "nonAneurysm"


def get_coll_status(condition_name: str) -> str:
    """Infer 'coll' vs 'noColl' from the condition folder name."""
    name = condition_name.lower()
    if "no" in name:
        return "noColl"
    return "coll"  # '+ coll', 'coll', etc.


def extract_cell_index(filename: str) -> Optional[int]:
    """Extract the cell number from something like 'Cell1_ch1.nii.gz'."""
    m = re.search(r"[Cc]ell(\d+)", filename)
    if m:
        return int(m.group(1))
    return None

# ---------------- STEP 1: collect all ch1 files & rename nicely ----------------

ch1_paths = []

for f in smc_root.rglob("*.nii*"):
    name_lower = f.name.lower()
    # Keep only nucleus channel 1 volumes: contain "ch1", but not "allchannels"
    if "ch1" in name_lower and "allchannels" not in name_lower:
        ch1_paths.append(f)

print(f"\nFound {len(ch1_paths)} ch1 volumes in SMCs_Zstacks")

rows = []  # for mapping


for src in sorted(ch1_paths):
    # Path structure assumed:
    # SMCs_Zstacks / SubjectID / ( '+ coll' | 'no coll' | similar ) / filename.nii.gz
    subject_id = src.parents[1].name         # e.g. "01Asc-230"
    condition_folder = src.parents[0].name   # e.g. "+ coll" or "no coll"

    aneur_status = get_aneurysm_status(subject_id)        # "Aneurysm" / "nonAneurysm"
    coll_status = get_coll_status(condition_folder)       # "coll" / "noColl"
    cell_idx = extract_cell_index(src.name)               # 1,2,3,...

    if cell_idx is None:
        print(f"⚠️  Could not parse cell index from {src.name}, skipping")
        continue

    # Target human-readable name:
    # SubjectID_Aneurysm/nonAneurysm_coll/noColl_nucleus_cell{k}.nii.gz
    new_name = (
        f"{subject_id}_{aneur_status}_{coll_status}_nucleus_cell{cell_idx}.nii.gz"
    )
    dst = collect_dir / new_name

    print(f"  {src.relative_to(smc_root)} -> {new_name}")
    shutil.copy2(src, dst)

    rows.append(
        {
            "original_path": str(src),
            "raw_name": src.name,
            "renamed_raw": new_name,
            "subject_id": subject_id,
            "aneurysm_status": aneur_status,
            "coll_status": coll_status,
            "cell_index": cell_idx,
        }
    )

print(f"\n Collected and renamed {len(rows)} ch1 volumes into {collect_dir}")


# ---------------- STEP 2: make nnU-Net-style case_XXXX_0000.nii.gz ----------------

raw_ch1_files = sorted(collect_dir.glob("*.nii*"))

from math import prod

nnunet_rows = []

print(f"\nPreparing nnU-Net files from {len(raw_ch1_files)} collected volumes")

for idx, src in enumerate(raw_ch1_files):
    case_id = f"case_{idx:04d}"
    nnunet_name = f"{case_id}_0000.nii.gz"
    dst = nnunet_dir / nnunet_name

    print(f"  {src.name} -> {nnunet_name}")
    shutil.copy2(src, dst)

    # find the row we created above for this file
    meta = next(r for r in rows if r["renamed_raw"] == src.name)

    nnunet_rows.append(
        {
            "case_id": case_id,
            "nnunet_filename": nnunet_name,
            "raw_filename": src.name,
            "subject_id": meta["subject_id"],
            "aneurysm_status": meta["aneurysm_status"],
            "coll_status": meta["coll_status"],
            "cell_index": meta["cell_index"],
        }
    )

# save mapping CSV
mapping_df = pd.DataFrame(nnunet_rows)
mapping_df.to_csv(mapping_csv, index=False)

print("\n Done.")
print("nnU-Net-ready files in :", nnunet_dir)
print("Case mapping saved to  :", mapping_csv)