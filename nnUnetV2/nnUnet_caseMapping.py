from pathlib import Path
import shutil
import re
import pandas as pd

# ---------- CONFIG ----------
project_root = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project")
raw_dir   = project_root / "data" / "raw_ch1_nucleus_all"
out_dir   = project_root / "data" / "ch1_nnunet"      # nnU-Net-style cases
mapping_csv = project_root / "data" / "ch1_nnunet_case_mapping.csv"

out_dir.mkdir(parents=True, exist_ok=True)
mapping_csv.parent.mkdir(parents=True, exist_ok=True)

print("Raw ch1 dir :", raw_dir)
print("nnU-Net dir :", out_dir)

# Filenames look like:
# 03Asc-24_aneurysm_coll_nucleus_set1_cell2.nii.gz
# 01C-0202_nonAneurysm_noColl_nucleus_cell5.nii.gz
#
# Optional "setX" part; always ends with "_cellY"
pattern = re.compile(
    r"^(?P<subject>[^_]+)_"                   # 03Asc-24
    r"(?P<aneurysm>aneurysm|nonAneurysm)_"    # aneurysm / nonAneurysm
    r"(?P<coll>coll|noColl)_"                 # coll / noColl
    r"nucleus"
    r"(?:_set(?P<set>\d+))?"                  # optional _set1 / _set2
    r"_cell(?P<cell>\d+)\.nii(\.gz)?$",
    re.IGNORECASE,
)

rows = []

files = sorted(raw_dir.glob("*.nii*"))
print(f"\nFound {len(files)} NIfTI files in {raw_dir}")

for idx, src in enumerate(files):
    m = pattern.match(src.name)
    if not m:
        print(f"⚠️  Filename does not match expected pattern, skipping: {src.name}")
        continue

    subject = m.group("subject")
    aneurysm_raw = m.group("aneurysm")
    coll_raw = m.group("coll")
    set_id = m.group("set")
    cell_idx = int(m.group("cell"))

    # Normalize labels a bit
    aneurysm_status = "Aneurysm" if aneurysm_raw.lower() == "aneurysm" else "nonAneurysm"
    coll_status = "coll" if coll_raw.lower() == "coll" else "noColl"

    case_id = f"case_{idx:04d}"
    nnunet_name = f"{case_id}_0000.nii.gz"
    dst = out_dir / nnunet_name

    print(f"{src.name}  ->  {nnunet_name}")
    shutil.copy2(src, dst)

    rows.append(
        {
            "case_id": case_id,
            "nnunet_filename": nnunet_name,
            "original_filename": src.name,
            "subject_id": subject,
            "aneurysm_status": aneurysm_status,
            "coll_status": coll_status,
            "set_id": set_id,
            "cell_index": cell_idx,
        }
    )

# Save mapping CSV for later analysis
df = pd.DataFrame(rows)
df.to_csv(mapping_csv, index=False)

print("\n✅ Done.")
print("nnU-Net-ready files in :", out_dir)
print("Case mapping saved to  :", mapping_csv)