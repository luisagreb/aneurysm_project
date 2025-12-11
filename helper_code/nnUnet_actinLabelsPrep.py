from pathlib import Path
import re
import csv
import SimpleITK as sitk

# -------- Paths --------
project_root = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project")

raw_dir   = project_root / "data" / "raw_ch2_actin_all"
label_dir = project_root / "data" / "labels_actin"
out_dir   = project_root / "data" / "nnunet_labels_actin"

print("Actin RAW images dir :", raw_dir)
print("Actin label dir      :", label_dir)
print("nnUNet labels out    :", out_dir)

out_dir.mkdir(parents=True, exist_ok=True)

raw_paths   = sorted(raw_dir.glob("*.nii.gz"))
label_paths = sorted(label_dir.glob("*.nrrd"))

print(f"Found {len(raw_paths)} raw actin images")
print(f"Found {len(label_paths)} actin label files")

# -------- Map RAW filename -> case index (same order as ch2_nnunet) --------
raw_name_to_idx = {}
for idx, p in enumerate(raw_paths):
    raw_name_to_idx[p.name] = idx
print(f"Indexed {len(raw_name_to_idx)} RAW actin images.\n")

exported = 0
skipped  = 0

mapping_csv = out_dir / "actin_label_case_mapping.csv"
rows = []

for lbl in label_paths:
    # Example label name:
    #   01Asc-180_nonAneurysm_coll_actin_cell1.nrrd
    stem = lbl.stem

    # split subject vs rest
    # subject: 01Asc-180
    # rest   : nonAneurysm_coll_actin_cell1
    parts = stem.split("_", 1)
    if len(parts) != 2:
        print(f"⚠️  Cannot parse label name {lbl.name}, skipping")
        skipped += 1
        continue

    subject, suffix = parts

    # We expect subject like "01Asc-180"
    m = re.match(r"^(\d+)(Asc-)(\d+)$", subject)
    if not m:
        print(f"⚠️  Subject pattern not recognized in {lbl.name}, skipping")
        skipped += 1
        continue

    prefix_num, asc_str, num_str = m.groups()

    # RAW uses upper "ASC" and 4-digit number with leading zero:
    #  -> "01ASC-0180"
    raw_subject = f"{prefix_num}{asc_str.upper()}{int(num_str):04d}"

    # RAW base name:
    raw_base = f"{raw_subject}_{suffix}"       # e.g. 01ASC-0180_nonAneurysm_coll_actin_cell1
    raw_name = raw_base + ".nii.gz"

    if raw_name not in raw_name_to_idx:
        print(f"⚠️  No matching RAW actin image for label {lbl.name} (expected {raw_name}), skipping")
        skipped += 1
        continue

    case_idx = raw_name_to_idx[raw_name]
    out_path = out_dir / f"case_{case_idx:04d}.nii.gz"

    # read NRRD label and save as NIfTI
    img = sitk.ReadImage(str(lbl))
    sitk.WriteImage(img, str(out_path))

    print(f"✅ {lbl.name}  ->  {out_path.name}  (RAW: {raw_name})")
    exported += 1

    rows.append({
        "case_idx": case_idx,
        "case_name": f"case_{case_idx:04d}",
        "raw_image": raw_name,
        "label_file": lbl.name
    })

# -------- Save mapping CSV --------
with open(mapping_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["case_idx", "case_name", "raw_image", "label_file"])
    writer.writeheader()
    writer.writerows(rows)

print("\n==============================")
print(f"✅ Exported : {exported}")
print(f"❌ Skipped  : {skipped}")
print("==============================")
print(f"📄 Case mapping CSV saved to: {mapping_csv}")