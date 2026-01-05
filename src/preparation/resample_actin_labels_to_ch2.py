from pathlib import Path
import SimpleITK as sitk

project_root = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project")

# ch2 nnUNet-style images (case_xxxx_0000.nii.gz)
actin_img_dir = project_root / "data" / "ch2_nnunet"

# original actin labels (case_xxxx.nii.gz)
orig_lbl_dir  = project_root / "data" / "nnunet_labels_actin"

# new folder with resampled labels
res_lbl_dir   = project_root / "data" / "nnunet_labels_actin_resampled"
res_lbl_dir.mkdir(parents=True, exist_ok=True)

print("Images dir :", actin_img_dir)
print("Labels in  :", orig_lbl_dir)
print("Resampled ->", res_lbl_dir)

for lbl_path in sorted(orig_lbl_dir.glob("case_*.nii.gz")):
    case_id = lbl_path.name.replace(".nii.gz", "")  # "case_0000"
    img_path = actin_img_dir / f"{case_id}_0000.nii.gz"

    if not img_path.exists():
        print(f"[MISSING IMG] {case_id}: {img_path.name} not found, skipping")
        continue

    print(f"[RESAMPLE] {case_id}")

    # read image + label
    img = sitk.ReadImage(str(img_path))
    lbl = sitk.ReadImage(str(lbl_path))

    # resample label to image grid, nearest neighbor (important for labels!)
    resampled_lbl = sitk.Resample(
        lbl,
        img,                                # reference image (size, spacing, origin, direction)
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,                                  # default background value
        lbl.GetPixelID(),
    )

    out_path = res_lbl_dir / lbl_path.name
    sitk.WriteImage(resampled_lbl, str(out_path))

print("Done resampling.")