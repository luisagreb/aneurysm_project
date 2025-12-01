import os
import SimpleITK as sitk
import numpy as np
from surface_distance import metrics


# -----------------------------
# Paths (edit these)
# -----------------------------
GT_DIR = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/label_nucleus"
PRED_DIR = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/predicted_nucleusMask_Unet"

# Output CSV
OUT_FILE = "unet_vs_manual_comparison.csv"


def load_mask(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # shape: (Z, Y, X)
    arr = (arr > 0).astype(np.uint8)
    spacing = img.GetSpacing()[::-1]  # convert to Z,Y,X order
    return arr, spacing


def dice_score(gt, pred):
    intersection = np.sum(gt * pred)
    return 2.0 * intersection / (np.sum(gt) + np.sum(pred) + 1e-8)


def compute_surface_metrics(gt, pred, spacing):
    # Required bool arrays
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    results = metrics.compute_all_metrics(gt, pred, spacing)

    return {
        "Hausdorff": results["hd"],
        "Assd": results["assd"],
    }


def main():
    results = []
    gt_files = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".nrrd")])

    for gt_file in gt_files:
        basename = gt_file.replace(".nrrd", "")
        pred_file = basename + ".nii_unetMask.nrrd"
        pred_path = os.path.join(PRED_DIR, pred_file)

        if not os.path.exists(pred_path):
            print(f"❌ Missing prediction for: {gt_file}")
            continue

        gt_mask, gt_spacing = load_mask(os.path.join(GT_DIR, gt_file))
        pred_mask, _ = load_mask(pred_path)

        # Metrics
        dice = dice_score(gt_mask, pred_mask)
        surf = compute_surface_metrics(gt_mask, pred_mask, gt_spacing)

        results.append((basename, dice, surf["Hausdorff"], surf["Assd"]))

        print(f"{basename} → Dice={dice:.4f}, HD={surf['Hausdorff']:.2f}, ASSD={surf['Assd']:.2f}")

    # Save CSV
    import csv
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Dice", "Hausdorff (mm)", "ASSD (mm)"])
        writer.writerows(results)

    print(f"\nResults saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()