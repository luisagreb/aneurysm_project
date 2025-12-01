from pathlib import Path
import numpy as np
import nibabel as nib
import tensorflow as tf
import nrrd
from skimage.transform import resize


# --------------------
# Paths
# --------------------
MODEL_PATH = Path(__file__).resolve().parent / "3d_unet_nucleus_seg.h5"

INPUT_DIR = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/raw_ch1_nucleus_all")
OUTPUT_DIR = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/Data/predicted_nucleusMask_Unet")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TARGET_SHAPE = (64, 64, 64)   # must match your model input


# --------------------
# Load model (compile=False is correct)
# --------------------
print(f"Loading model from: {MODEL_PATH}")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={}
)

print("Model loaded.")


# --------------------
# Process each file
# --------------------
files = sorted(INPUT_DIR.glob("*.nii*"))
print(f"Found {len(files)} volumes to segment.")

for i, file in enumerate(files, 1):
    print(f"\n[{i}] Processing: {file.name}")

    # Load original volume
    nii = nib.load(str(file))
    vol = nii.get_fdata().astype(np.float32)

    # Normalization
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

    # Resize to model input
    vol_resized = resize(vol, TARGET_SHAPE, order=1, preserve_range=True).astype(np.float32)

    # Add batch + channel dims
    inp = vol_resized[np.newaxis, ..., np.newaxis]  # shape (1,64,64,64,1)

    # Predict mask
    pred = model.predict(inp, verbose=0)[0, ..., 0]   # shape (64,64,64)

    # Threshold
    mask = (pred > 0.5).astype(np.uint8)

    # Use original voxel spacing
    spacing = nii.header.get_zooms()  # (sx, sy, sz)

    # Save NRRD
    out_path = OUTPUT_DIR / (file.stem + "_unetMask.nrrd")

    header = {
        "space": "left-posterior-superior",
        "spacings": list(spacing)
    }

    nrrd.write(str(out_path), mask, header)

    print(f" Saved mask → {out_path.name}")

print("\nAll masks saved in:", OUTPUT_DIR)