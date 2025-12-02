import os
import csv
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from surface_distance import metrics

# ----------------- PATHS -----------------
GT_DIR = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/Data/label_nucleus"
PRED_DIR = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/Data/predicted_nucleusMask_Unet"

OUT_FILE = "unet_vs_manual_comparison.csv"
PLOT_DIR = "unet_vs_manual_plots"



def load_image(path):
    """
    Read image with SimpleITK, with a clear error if it fails.
    """
    try:
        return sitk.ReadImage(path)
    except Exception as e:
        print(f"   ❌ SimpleITK failed to read: {path}")
        print(f"      Reason: {e}")
        return None


def image_to_bool_array(img):
    """
    Convert a SimpleITK image to a boolean numpy array (Z, Y, X).
    """
    arr = sitk.GetArrayFromImage(img)      # (Z, Y, X)
    arr = arr > 0                          # boolean mask
    return arr.astype(np.bool_)            # explicitly bool dtype


def dice_score(gt_bool, pred_bool):
    """
    Dice on boolean arrays.
    """
    intersection = np.logical_and(gt_bool, pred_bool).sum()
    denom = gt_bool.sum() + pred_bool.sum() + 1e-8
    return 2.0 * intersection / denom


def resample_mask_to_target(gt_img, target_img):
    """
    Resample ground-truth mask (gt_img) to the geometry of target_img
    using nearest-neighbor, then return a boolean numpy array and spacing.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())  # identity
    resampler.SetDefaultPixelValue(0)

    gt_resampled = resampler.Execute(gt_img)

    # convert to bool array
    arr = image_to_bool_array(gt_resampled)
    spacing = target_img.GetSpacing()[::-1]  # (z,y,x)

    return arr, spacing


def compute_surface_metrics(gt_bool, pred_bool, spacing):
    """
    Compute HD95 and ASSD with surface_distance.metrics.
    Inputs must be boolean arrays (Z,Y,X).
    """
    gt_bool = np.asarray(gt_bool).astype(np.bool_)
    pred_bool = np.asarray(pred_bool).astype(np.bool_)

    # Debug: check dtypes before calling library (can comment out later)
    # print("   dtypes before metrics:", gt_bool.dtype, pred_bool.dtype)

    sd = metrics.compute_surface_distances(gt_bool, pred_bool, spacing)

    # robust hausdorff (HD95)
    try:
        hd95 = metrics.compute_robust_hausdorff(sd, 95.0)
    except Exception:
        # fallback if something weird happens
        hd95 = max(
            sd["distances_gt_to_pred"].max(),
            sd["distances_pred_to_gt"].max(),
        )

    # average symmetric surface distance (ASSD)
    assd_pair = metrics.compute_average_surface_distance(sd)  # len=2 array
    assd = float(np.mean(assd_pair))

    return float(hd95), assd


# ----------------- MAIN -----------------
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    gt_files = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".nrrd")])
    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith(".nrrd")])

    print("\n===== Mask Comparison =====\n")
    print(f"GT files   : {len(gt_files)}")
    print(f"Pred files : {len(pred_files)}")

    # match by exact filename
    common = sorted(set(gt_files) & set(pred_files))

    if not common:
        print("❌ No matching filenames between GT_DIR and PRED_DIR.")
        print("GT examples:", gt_files[:5])
        print("Pred examples:", pred_files[:5])
        return

    print(f"→ Found {len(common)} matching GT–Prediction pairs.\n")

    results = []

    for fname in common:
        print(f"• Comparing {fname} …")
        gt_path = os.path.join(GT_DIR, fname)
        pred_path = os.path.join(PRED_DIR, fname)

        # load images
        gt_img = load_image(gt_path)
        pred_img = load_image(pred_path)

        if gt_img is None or pred_img is None:
            print("   → Skipping this pair because one image could not be read.\n")
            continue

        gt_arr = image_to_bool_array(gt_img)
        pred_arr = image_to_bool_array(pred_img)

        # if shapes differ, resample GT to match prediction
        if gt_arr.shape != pred_arr.shape:
            print(f"   ⚠️ Shape mismatch: GT {gt_arr.shape}, PRED {pred_arr.shape}")
            print("     → Resampling GT to prediction shape.")
            gt_arr, spacing = resample_mask_to_target(gt_img, pred_img)
        else:
            spacing = gt_img.GetSpacing()[::-1]  # (z,y,x)

        # ensure still match
        if gt_arr.shape != pred_arr.shape:
            print(f"   ❌ Still mismatch after resampling, skipping: GT {gt_arr.shape}, PRED {pred_arr.shape}")
            continue

        # Dice
        dsc = dice_score(gt_arr, pred_arr)

        # Surface metrics
        hd95, assd = compute_surface_metrics(gt_arr, pred_arr, spacing)

        results.append({
            "name": fname,
            "dice": dsc,
            "hd95": hd95,
            "assd": assd,
        })

        print(f"   Dice={dsc:.4f}, HD95={hd95:.2f} mm, ASSD={assd:.2f} mm\n")

    if not results:
        print("No valid pairs → nothing to save or plot.")
        return

    # ---------- Save CSV ----------
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Dice", "HD95 (mm)", "ASSD (mm)"])
        for r in results:
            writer.writerow([r["name"], r["dice"], r["hd95"], r["assd"]])

    print(f"Results saved to: {OUT_FILE}")

    # ---------- Aggregate & Plots ----------
    names = [r["name"] for r in results]
    dice_vals = np.array([r["dice"] for r in results])
    hd_vals   = np.array([r["hd95"] for r in results])
    assd_vals = np.array([r["assd"] for r in results])

    print("\nSummary statistics:")
    print(f"Dice: mean={dice_vals.mean():.3f}, std={dice_vals.std():.3f}, "
          f"min={dice_vals.min():.3f}, max={dice_vals.max():.3f}")
    print(f"HD95: mean={hd_vals.mean():.3f}, std={hd_vals.std():.3f}")
    print(f"ASSD: mean={assd_vals.mean():.3f}, std={assd_vals.std():.3f}")

    x = np.arange(len(names))

    # Dice per case
    plt.figure(figsize=(10, 4))
    plt.bar(x, dice_vals)
    plt.xticks(x, names, rotation=90)
    plt.ylabel("Dice coefficient")
    plt.title("UNet vs Manual – Dice per cell")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "dice_per_case.png"), dpi=150)
    plt.close()

    # Histograms
    plt.figure()
    plt.hist(dice_vals, bins=10, range=(0, 1.0))
    plt.xlabel("Dice")
    plt.ylabel("Count")
    plt.title("Distribution of Dice scores")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "hist_dice.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.hist(hd_vals, bins=10)
    plt.xlabel("HD95 (mm)")
    plt.ylabel("Count")
    plt.title("Distribution of Hausdorff distance (95%)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "hist_hd95.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.hist(assd_vals, bins=10)
    plt.xlabel("ASSD (mm)")
    plt.ylabel("Count")
    plt.title("Distribution of ASSD")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "hist_assd.png"), dpi=150)
    plt.close()

    # Scatter plots
    plt.figure()
    plt.scatter(dice_vals, hd_vals)
    plt.xlabel("Dice")
    plt.ylabel("HD95 (mm)")
    plt.title("Dice vs HD95")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scatter_dice_vs_hd95.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.scatter(dice_vals, assd_vals)
    plt.xlabel("Dice")
    plt.ylabel("ASSD (mm)")
    plt.title("Dice vs ASSD")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scatter_dice_vs_assd.png"), dpi=150)
    plt.close()

    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()