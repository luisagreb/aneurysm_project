import os
import glob
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
import tensorflow as tf

# --- Imports from your project ---
from uNet_model import unet_model
from dataProcessing import load_and_preprocess_volume

# ---------------- CONFIG ----------------
# Same as in train_model.py
BASE_DATA_DIRECTORY = r"C:\Users\Luisa\Documents\aneurysm_project\Data"
RAW_CHANNEL_DIR     = "raw_ch1_nucleus"
RAW_EXTENSION       = ".nii"     # or ".nii.gz" if needed

MODEL_PATH          = "3d_unet_nucleus_seg.h5"
OUTPUT_DIR          = os.path.join(BASE_DATA_DIRECTORY, "predicted_nucleusMask_Unet")

# Must match training
IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH = 64, 64, 64
TARGET_SHAPE = (IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH)
# ----------------------------------------


def ensure_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_original_image(path):
    """
    Load raw volume with SimpleITK and return:
    - image (SimpleITK object with spacing, direction, origin)
    - numpy volume in (D, H, W) order
    """
    itk_img = sitk.ReadImage(path)
    vol_zyx = sitk.GetArrayFromImage(itk_img)  # (Z, Y, X)
    vol_dhw = vol_zyx.transpose(2, 1, 0).astype(np.float32)  # → (D, H, W)
    return itk_img, vol_dhw


def postprocess_and_save(pred_64, original_img, original_shape, out_path, threshold=0.5):
    """
    pred_64: numpy (64,64,64,1) prediction in [0,1]
    original_img: SimpleITK image of the raw volume
    original_shape: (D,H,W) of the raw volume
    """
    # Remove channel
    pred_64 = pred_64[..., 0]

    # Binarize
    pred_bin = (pred_64 >= threshold).astype(np.float32)

    # Resize back to original (D,H,W)
    zoom_factors = [orig / cur for orig, cur in zip(original_shape, pred_bin.shape)]
    pred_resampled = zoom(pred_bin, zoom_factors, order=0)  # nearest neighbour

    # Convert back to (Z,Y,X) for SimpleITK
    pred_zyx = pred_resampled.transpose(2, 1, 0).astype(np.uint8)

    # Create SITK image and copy geometry
    seg_img = sitk.GetImageFromArray(pred_zyx)
    seg_img.SetSpacing(original_img.GetSpacing())
    seg_img.SetOrigin(original_img.GetOrigin())
    seg_img.SetDirection(original_img.GetDirection())

    # Save as NRRD
    sitk.WriteImage(seg_img, out_path)
    print(f"  → Saved prediction to: {out_path}")


def main():
    # 1. Load model
    print("Loading model...")
    model = unet_model(
        input_shape=(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, 1),
        num_classes=1
    )
    model.load_weights(MODEL_PATH)
    print(f"Loaded weights from: {MODEL_PATH}")

    # 2. Prepare output dir
    ensure_output_dir(OUTPUT_DIR)

    # 3. Collect raw volumes
    raw_pattern = os.path.join(BASE_DATA_DIRECTORY, RAW_CHANNEL_DIR, f"*{RAW_EXTENSION}")
    raw_paths = sorted(glob.glob(raw_pattern))

    if not raw_paths:
        print(f"No raw files found at {raw_pattern}")
        return

    print(f"Found {len(raw_paths)} raw volumes for prediction.")

    # 4. Loop over volumes
    for i, raw_path in enumerate(raw_paths, 1):
        base_name = os.path.basename(raw_path)
        print(f"\n[{i}/{len(raw_paths)}] Predicting for: {base_name}")

        # 4.1 Load original image & shape
        original_img, vol_dhw = load_original_image(raw_path)
        original_shape = vol_dhw.shape  # (D,H,W)

        # 4.2 Preprocess to 64^3 as in training
        vol_input = load_and_preprocess_volume(
            file_path=raw_path,
            target_shape=TARGET_SHAPE,
            is_mask=False
        )  # returns shape (64,64,64,1)

        # Add batch dimension: (1, D,H,W,1)
        x = np.expand_dims(vol_input, axis=0)

        # 4.3 Run prediction
        pred = model.predict(x, verbose=0)[0]  # back to (64,64,64,1)

        # 4.4 Postprocess & save as NRRD in original space
        out_name = base_name.replace(RAW_EXTENSION, "_nucleusMask_unet.nrrd")
        out_path = os.path.join(OUTPUT_DIR, out_name)
        postprocess_and_save(pred, original_img, original_shape, out_path)

    print("\nDone. All prediction masks saved.")


if __name__ == "__main__":
    main()