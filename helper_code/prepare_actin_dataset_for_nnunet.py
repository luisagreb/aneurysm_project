from pathlib import Path
import shutil
import json
import nibabel as nib

# --- Paths ---
project_root = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project")

actin_img_dir   = project_root / "data" / "ch2_nnunet"           # case_xxxx_0000.nii.gz
actin_lbl_dir   = project_root / "data" / "nnunet_labels_actin_resampled"  # case_xxxx.nii.gz
dataset_root    = project_root / "data" / "actin_nnunet"

imagesTr = dataset_root / "imagesTr"
labelsTr = dataset_root / "labelsTr"
imagesTs = dataset_root / "imagesTs"

print("Actin nnUNet images dir:", actin_img_dir)
print("Actin nnUNet labels dir:", actin_lbl_dir)
print("Dataset root           :", dataset_root)

# Clean + recreate output dirs
if dataset_root.exists():
    print("⚠️  Removing existing", dataset_root)
    shutil.rmtree(dataset_root)

imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)
imagesTs.mkdir(parents=True, exist_ok=True)

# --- Collect label files ---
label_files = sorted(actin_lbl_dir.glob("case_*.nii.gz"))
print(f"Found {len(label_files)} label files in nnunet_labels_actin")

train_cases = []
skipped_shape = []

# --- Build training set: only keep matching shapes ---
for lbl_path in label_files:
    # strip the full ".nii.gz" extension to get "case_0000"
    case_id = lbl_path.name.replace(".nii.gz", "")         # "case_0000"
    img_path = actin_img_dir / f"{case_id}_0000.nii.gz"

    if not img_path.exists():
        print(f"[MISSING IMG] {case_id}: {img_path.name} not found, skipping")
        continue

    # Shape check
    img = nib.load(str(img_path))
    lbl = nib.load(str(lbl_path))

    if img.shape != lbl.shape:
        print(f"[SHAPE MISMATCH] {case_id}: img {img.shape} vs lbl {lbl.shape}, skipping")
        skipped_shape.append(case_id)
        continue

    # Copy to imagesTr / labelsTr
    dst_img = imagesTr / img_path.name   # case_0000_0000.nii.gz
    dst_lbl = labelsTr / lbl_path.name   # case_0000.nii.gz

    shutil.copy2(img_path, dst_img)
    shutil.copy2(lbl_path, dst_lbl)

    train_cases.append(case_id)
    print(f"[OK] {case_id} -> imagesTr/labelsTr")

print("\nKept training cases:", train_cases)
print("Skipped (shape mismatch):", skipped_shape)
print(f"Total training cases: {len(train_cases)}")

# --- Build test set: all actin images not used for training ---
all_img_files = sorted(actin_img_dir.glob("case_*_0000.nii.gz"))
used_set = set(train_cases)

test_cases = []

for img_path in all_img_files:
    # name: case_0000_0000.nii.gz -> stem: case_0000_0000 -> case_0000
    case_id = img_path.stem.rsplit("_", 1)[0]

    if case_id in used_set:
        continue

    dst_img = imagesTs / img_path.name
    shutil.copy2(img_path, dst_img)
    test_cases.append(case_id)

print(f"Total test (imagesTs) cases: {len(test_cases)}")

# --- Build dataset.json for nnUNetv2 ---
dataset_json = {
    "name": "ActinSeg",
    "description": "Actin channel cell segmentation",
    "tensorImageSize": "3D",
    "reference": "",
    "licence": "",
    "release": "1.0",
    "modality": {
        "0": "actin"
    },
    "labels": {
        "0": "background",
        "1": "actin"
    },
    "numTraining": len(train_cases),
    "numTest": len(test_cases),
    "training": [],
    "test": [],
    "channel_names": {
        "0": "actin"
    },
    "file_ending": ".nii.gz"
}

# Fill training entries
for case_id in sorted(train_cases):
    dataset_json["training"].append({
        "image": f"./imagesTr/{case_id}_0000.nii.gz",
        "label": f"./labelsTr/{case_id}.nii.gz"
    })

# Fill test entries (only images)
for img_path in sorted(imagesTs.glob("case_*_0000.nii.gz")):
    rel = f"./imagesTs/{img_path.name}"
    dataset_json["test"].append(rel)

out_json = dataset_root / "dataset.json"
out_json.write_text(json.dumps(dataset_json, indent=4))
print("\nSaved dataset.json to:", out_json)