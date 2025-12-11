from pathlib import Path
import re
import csv
import SimpleITK as sitk

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

def make_key_from_stem(stem: str):
    """
    Turn a filename stem like
      01ASC-0180_nonAneurysm_coll_actin_cell1
    or
      01Asc-180_nonAneurysm_coll_actin_cell1
    into a normalized key:
      01_asc_180_nonaneurysm_coll_actin_cell1
    (lower-case, typo-fixed, zero-padding ignored)
    """
    m = re.match(r"^(\d+)([Aa][Ss][Cc])-(\d+)_(.+)$", stem)
    if not m:
        return None
    prefix_num, asc_str, num_str, rest = m.groups()
    num_int = int(num_str)  # 180 and 0180 → 180

    rest_norm = rest.lower()
    rest_norm = rest_norm.replace("aneursym", "aneurysm")  # fix typo if present

    key = f"{prefix_num}_asc_{num_int}_{rest_norm}"
    return key

# -------- Build map from normalized key -> (case_idx, raw_name) ----------
key_to_case = {}
for idx, p in enumerate(raw_paths):
    stem = p.name.replace(".nii.gz", "")
    key = make_key_from_stem(stem)
    if key is None:
        print(f"⚠️  RAW name pattern not recognized: {p.name}")
        continue
    if key in key_to_case:
        print(f"⚠️  Duplicate RAW key {key} for {p.name} (already {key_to_case[key][1]})")
    key_to_case[key] = (idx, p.name)

print(f"Built normalized key mapping for {len(key_to_case)} RAW images.\n")

exported = 0
skipped  = 0
rows = []
mapping_csv = out_dir / "actin_label_case_mapping.csv"

# -------- Match each label to a RAW and write case_XXXX.nii.gz ----------
for lbl in label_paths:
    stem = lbl.stem
    key = make_key_from_stem(stem)
    if key is None:
        print(f"⚠️  Label name pattern not recognized: {lbl.name}, skipping")
        skipped += 1
        continue

    if key not in key_to_case:
        print(f"⚠️  No matching RAW actin image for label {lbl.name} (key={key}), skipping")
        skipped += 1
        continue

    case_idx, raw_name = key_to_case[key]
    out_path = out_dir / f"case_{case_idx:04d}.nii.gz"

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

# -------- Save mapping CSV ----------
with open(mapping_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["case_idx", "case_name", "raw_image", "label_file"])
    writer.writeheader()
    writer.writerows(rows)

print("\n==============================")
print(f"✅ Exported : {exported}")
print(f"❌ Skipped  : {skipped}")
print("==============================")
print(f"📄 Case mapping CSV saved to: {mapping_csv}")