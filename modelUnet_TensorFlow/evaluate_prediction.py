import os
import numpy as np
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from glob import glob

# ---------------------------------------------
# Helper functions
# ---------------------------------------------

def load_nrrd(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.uint8)
    return arr

def dice_coef(gt, pred):
    intersection = np.sum(gt * pred)
    return (2 * intersection) / (gt.sum() + pred.sum() + 1e-8)

def jaccard(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt + pred) - intersection
    return intersection / (union + 1e-8)

def precision(gt, pred):
    tp = np.sum((gt == 1) & (pred == 1))
    fp = np.sum((gt == 0) & (pred == 1))
    return tp / (tp + fp + 1e-8)

def recall(gt, pred):
    tp = np.sum((gt == 1) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    return tp / (tp + fn + 1e-8)

def hausdorff_95(gt, pred):
    gt_pts = np.argwhere(gt == 1)
    pred_pts = np.argwhere(pred == 1)
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return np.nan
    d1 = directed_hausdorff(gt_pts, pred_pts)[0]
    d2 = directed_hausdorff(pred_pts, gt_pts)[0]
    return np.percentile([d1, d2], 95)

def volume_voxels(mask):
    return np.sum(mask)

# ---------------------------------------------
# Paths
# ---------------------------------------------
GT_DIR = "C:/Users/Luisa/Documents/aneurysm_project/Data/label_nucleus"
PRED_DIR = "C:/Users/Luisa/Documents/aneurysm_project/predictions_nrrd"

gt_files = sorted(glob(os.path.join(GT_DIR, "*.nrrd")))
pred_files = sorted(glob(os.path.join(PRED_DIR, "*.nrrd")))

# Match by base name
pairs = []
for gt in gt_files:
    base = os.path.basename(gt).replace("Label", "")  # adapt match rules if needed
    pred = os.path.join(PRED_DIR, base)
    if os.path.exists(pred):
        pairs.append((gt, pred))
    else:
        print(f"⚠️ Warning: missing prediction for {gt}")

print(f"\nFound {len(pairs)} matched GT–Prediction pairs.\n")

# ---------------------------------------------
# Evaluation loop
# ---------------------------------------------
results = []

for gt_path, pred_path in pairs:

    gt = load_nrrd(gt_path)
    pred = load_nrrd(pred_path)

    pred = (pred > 0.5).astype(np.uint8)  # binarize if needed

    d = dice_coef(gt, pred)
    j = jaccard(gt, pred)
    p = precision(gt, pred)
    r = recall(gt, pred)
    hd95 = hausdorff_95(gt, pred)
    v_gt = volume_voxels(gt)
    v_pred = volume_voxels(pred)

    results.append([gt_path, d, j, p, r, hd95, v_gt, v_pred])

    print(f"--- {os.path.basename(gt_path)} ---")
    print(f"Dice:     {d:.4f}")
    print(f"Jaccard:  {j:.4f}")
    print(f"Precision:{p:.4f}")
    print(f"Recall:   {r:.4f}")
    print(f"HD95:     {hd95:.2f}")
    print(f"Volume GT:{v_gt}  |  Pred:{v_pred}")
    print("-------------------------------\n")

# ---------------------------------------------
# Save CSV summary
# ---------------------------------------------
import pandas as pd

df = pd.DataFrame(results, columns=[
    "Sample", "Dice", "Jaccard", "Precision", "Recall",
    "HD95", "Volume_GT", "Volume_Pred"
])

df.to_csv("segmentation_evaluation.csv", index=False)
print("\nSaved results to segmentation_evaluation.csv")