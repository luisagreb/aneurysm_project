from pathlib import Path
import numpy as np
import nrrd

# Folders – adapt to your structure
GT_DIR   = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/segmented_auto_nucleus")  # manual
PRED_DIR = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/Data/predicted_nucleusMask_Unet")

def dice(a, b):
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    return (2.0 * inter) / (a.sum() + b.sum() + 1e-8)

scores = []

for gt_path in sorted(GT_DIR.glob("*.nrrd")):
    key = gt_path.stem  # adapt if needed

    # Find matching prediction – adjust pattern to your filenames
    pred_path = PRED_DIR / f"{key}.nrrd"
    if not pred_path.exists():
        print(f"⚠️ No prediction for {gt_path.name}")
        continue

    gt, _   = nrrd.read(str(gt_path))
    pred, _ = nrrd.read(str(pred_path))

    # Make sure shapes match
    if gt.shape != pred.shape:
        print(f"⚠️ Shape mismatch for {key}: gt {gt.shape}, pred {pred.shape}")
        continue

    d = dice(gt, pred)
    scores.append(d)
    print(f"{key}: Dice = {d:.3f}")

if scores:
    print("\nMean Dice:", np.mean(scores))
    print("Median Dice:", np.median(scores))