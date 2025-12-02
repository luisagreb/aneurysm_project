import os
import numpy as np
import SimpleITK as sitk
from glob import glob
import pandas as pd
from scipy.spatial.distance import directed_hausdorff

# -------------------------------------------------------------------
# CONFIGURE THESE PATHS
# -------------------------------------------------------------------
GT_DIR   = r"C:\Users\Luisa\Documents\aneurysm_project\Data\label_nucleus"
PRED_DIR = r"C:\Users\Luisa\Documents\aneurysm_project\Data\predicted_nucleusMask_Unet"

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def load_nrrd(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.uint8)
    return arr

def dice(gt, pred):
    inter = np.sum(gt * pred)
    return (2 * inter) / (gt.sum() + pred.sum() + 1e-8)

def jaccard(gt, pred):
    inter = np.sum(gt * pred)
    union = gt.sum() + pred.sum() - inter
    return inter / (union + 1e-8)

def precision(gt, pred):
    tp = np.sum((gt==1) & (pred==1))
    fp = np.sum((gt==0) & (pred==1))
    return tp / (tp + fp + 1e-8)

def recall(gt, pred):
    tp = np.sum((gt==1) & (pred==1))
    fn = np.sum((gt==1) & (pred==0))
    return tp / (tp + fn + 1e-8)

def hd95(gt, pred):
    gt_pts = np.argwhere(gt == 1)
    pr_pts = np.argwhere(pred == 1)
    if len(gt_pts)==0 or len(pr_pts)==0:
        return np.nan
    d1 = directed_hausdorff(gt_pts, pr_pts)[0]
    d2 = directed_hausdorff(pr_pts, gt_pts)[0]
    return np.percentile([d1, d2], 95)

def volume(mask):
    return int(mask.sum())

# -------------------------------------------------------------------
# THE IMPORTANT PART: reliable matching
# -------------------------------------------------------------------

def core_id_gt(path):
    """Convert GT filename into standard ID."""
    name = os.path.basename(path).replace("nucleusLabel_", "nucleus_")
    name = name.replace(".nrrd", "")
    return name

def core_id_pred(path):
    """Convert predicted filename into standard ID."""
    name = os.path.basename(path).replace("_nucleusMask_unet", "")
    name = name.replace(".nrrd", "")
    return name

# Load files
gt_files   = sorted(glob(os.path.join(GT_DIR, "*.nrrd")))
pred_files = sorted(glob(os.path.join(PRED_DIR, "*.nrrd")))

# Build pred dictionary
pred_dict = {core_id_pred(p): p for p in pred_files}

# Match pairs
pairs = []
for gt in gt_files:
    cid = core_id_gt(gt)
    if cid in pred_dict:
        pairs.append((gt, pred_dict[cid]))
    else:
        print(f"⚠ Missing prediction for {os.path.basename(gt)}")

print(f"\n✓ Found {len(pairs)} matched GT–Prediction pairs\n")

# -------------------------------------------------------------------
# Compute metrics
# -------------------------------------------------------------------

results = []

for gt_path, pr_path in pairs:
    gt = load_nrrd(gt_path)
    pr = load_nrrd(pr_path)
    pr = (pr > 0.5).astype(np.uint8)

    fn = os.path.basename(gt_path)

    d  = dice(gt, pr)
    j  = jaccard(gt, pr)
    p  = precision(gt, pr)
    r  = recall(gt, pr)
    h  = hd95(gt, pr)
    vg = volume(gt)
    vp = volume(pr)

    results.append([fn, d, j, p, r, h, vg, vp])

    print(f"{fn}: Dice={d:.3f}, Jaccard={j:.3f}, Precision={p:.3f}, Recall={r:.3f}, HD95={h:.2f}, VolGT={vg}, VolPred={vp}")

# Save CSV
df = pd.DataFrame(results, columns=["Name", "Dice", "Jaccard", "Precision", "Recall", "HD95", "Vol_GT", "Vol_Pred"])
df.to_csv("prediction_stats.csv", index=False)
print("\nSaved: prediction_stats.csv")