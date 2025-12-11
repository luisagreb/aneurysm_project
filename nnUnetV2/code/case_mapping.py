from pathlib import Path
import shutil
import re
import pandas as pd

project_root = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project")

raw_dir    = project_root / "data" / "raw_ch1_nucleus_all"
nnunet_dir = project_root / "data" / "ch1_nnunet"
excel_out  = project_root / "data" / "ch1_nnunet_case_mapping.xlsx"

print("Raw nuclei dir   :", raw_dir)
print("nnUNet out dir   :", nnunet_dir)
print("Excel will be at :", excel_out)

raw_files = sorted(raw_dir.glob("*.nii.gz"))
print(f"Found {len(raw_files)} raw ch1 files")

nnunet_dir.mkdir(parents=True, exist_ok=True)

# --- filename parser: subject / aneurysm / coll / (optional) set / cell ---
# Examples:
#   01Asc-222_nonAneurysm_coll_nucleus_cell1.nii.gz
#   03Asc-24_aneurysm_coll_nucleus_set3_cell5.nii.gz
pattern = re.compile(
    r"(?P<subject>[A-Za-z0-9\-]+)_"          # subject ID
    r"(?P<aneurysm>aneurysm|nonAneurysm)_"   # aneurysm status
    r"(?P<coll>coll|noColl)_"                # coll / noColl
    r"nucleus_(?:set(?P<set>\d+)_)?cell(?P<cell>\d+)\.nii\.gz$",  # optional set, then cell
    re.IGNORECASE,
)

rows = []

for idx, src in enumerate(raw_files):
    case_name = f"case_{idx:04d}_0000.nii.gz"
    dst = nnunet_dir / case_name
    shutil.copy2(src, dst)
    print(f"{idx:03d}: {src.name}  ->  {case_name}")

    # --- parse metadata from original filename ---
    raw_name = src.name
    m = pattern.match(raw_name)

    subject = aneurysm_status = coll_status = ""
    set_idx = None
    cell_idx = None
    note = ""

    if not m:
        note = "could not parse filename pattern"
        print(f"⚠️  Could not parse: {raw_name}")
    else:
        subject = m.group("subject")
        aneurysm_status = m.group("aneurysm")
        coll_status = m.group("coll")
        set_idx = m.group("set")
        cell_idx = int(m.group("cell"))

        # normalize fields a bit
        aneurysm_status = "aneurysm" if aneurysm_status.lower() == "aneurysm" else "nonAneurysm"
        coll_status = "coll" if coll_status.lower() == "coll" else "noColl"

    case_id = f"{idx:04d}"        # 0000, 0001, ...
    case_base = f"case_{case_id}" # case_0000

    rows.append(
        {
            "case_index": idx,
            "case_id": case_id,
            "nnunet_case": case_base,
            "nnunet_input_file": str(dst),
            "raw_filename": raw_name,
            "raw_path": str(src),
            "subject_id": subject,
            "aneurysm_status": aneurysm_status,
            "coll_status": coll_status,
            "set_index": int(set_idx) if set_idx is not None else None,
            "cell_index": cell_idx,
            "note": note,
        }
    )

print("\n✅ Rebuilt ch1_nnunet with", len(raw_files), "cases.")

# --- save mapping to Excel ---
df = pd.DataFrame(rows).sort_values("case_index")
excel_out.parent.mkdir(parents=True, exist_ok=True)
df.to_excel(excel_out, index=False)

print(f"✅ Saved case → cell mapping to:\n{excel_out}")