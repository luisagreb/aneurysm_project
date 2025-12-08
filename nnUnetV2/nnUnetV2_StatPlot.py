import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project root = .../aneurysm_project
project_root = Path(__file__).resolve().parents[1]

gt_dir   = project_root / "data" / "nnunet_labels"
pred_dir = project_root / "data" / "nnunet_preds_val"

cases = sorted(pred_dir.glob("case_*.nii.gz"))

dice_scores = []
case_names = []

print(f"Found {len(cases)} prediction files in {pred_dir}")

for pred_path in cases:
    case_id = pred_path.stem.split(".")[0]   # "case_0000" from "case_0000.nii.gz"
    gt_path = gt_dir / f"{case_id}.nii.gz"

    if not gt_path.exists():
        print(f"⚠️  Missing GT for {case_id}, skipping")
        continue

    # Read images and binarize
    pred_img = sitk.ReadImage(str(pred_path))
    gt_img   = sitk.ReadImage(str(gt_path))

    pred = sitk.GetArrayFromImage(pred_img) > 0
    gt   = sitk.GetArrayFromImage(gt_img) > 0

    intersection = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum   = gt.sum()

    if pred_sum + gt_sum == 0:
        dice = 1.0  # both empty
    else:
        dice = 2.0 * intersection / (pred_sum + gt_sum)

    dice_scores.append(dice)
    case_names.append(case_id)

    print(f"{case_id}: Dice = {dice:.4f}")

dice_scores = np.array(dice_scores)

print("\nSummary:")
print(f"Mean Dice: {dice_scores.mean():.4f}")
print(f"Std  Dice: {dice_scores.std():.4f}")
print(f"Min  Dice: {dice_scores.min():.4f}")
print(f"Max  Dice: {dice_scores.max():.4f}")

# Save bar plot
out_dir = project_root / "nnUnetV2/model_analysis"
out_dir.mkdir(exist_ok=True)

plt.figure(figsize=(8, 4))
plt.bar(range(len(dice_scores)), dice_scores)
plt.xticks(range(len(dice_scores)), case_names, rotation=90)
plt.ylim(0, 1.0)
plt.ylabel("Dice coefficient")
plt.title("nnU-Net nucleus segmentation – per-case Dice")
plt.tight_layout()
plt.savefig(out_dir / "nnunet_val_dice_per_case.png", dpi=200)
plt.close()

print("\nSaved plot to:", out_dir / "nnunet_val_dice_per_case.png")
