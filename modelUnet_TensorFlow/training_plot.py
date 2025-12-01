import numpy as np
import nibabel as nib
import nrrd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize


# -------- CONFIG --------
RAW_FILE = Path("Users/Luisa/Documents/aneurysm_project/Data/raw_ch1_nucleus/01ASC-0180_nonAneurysm_noColl_nucleus_cell1.nii")
MASK_FILE = Path("Users/Luisa/Documents/aneurysm_project/Data/label_nucleus\/1ASC-0180_nonAneurysm_noColl_nucleusLabel_cell1.nrrd")

TARGET_SHAPE = (64, 64, 64)



# -------- LOAD RAW VOLUME --------
print("\n--- Loading RAW volume ---")
raw = nib.load(str(RAW_FILE)).get_fdata().astype(np.float32)
print("Original raw shape:", raw.shape)

# Normalize to 0–1
raw_norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

# Resize to model shape (same as training & prediction)
raw_resized = resize(raw_norm, TARGET_SHAPE, order=1, preserve_range=True).astype(np.float32)
print("Resized raw shape:", raw_resized.shape)



# -------- LOAD MASK --------
print("\n--- Loading MASK (NRRD) ---")
mask, header = nrrd.read(str(MASK_FILE))
mask = mask.astype(np.uint8)

print("Original mask shape:", mask.shape)
print("Mask values:", np.unique(mask))

# Resize mask (NEAREST interpolation)
mask_resized = resize(mask, TARGET_SHAPE, order=0, preserve_range=True).astype(np.uint8)
print("Resized mask shape:", mask_resized.shape)
print("Mask values after resize:", np.unique(mask_resized))



# -------- VISUALIZATION --------
# Pick central slice from each axis
z = TARGET_SHAPE[0] // 2
y = TARGET_SHAPE[1] // 2
x = TARGET_SHAPE[2] // 2

plt.figure(figsize=(12, 8))

# --- RAW ---
plt.subplot(2, 3, 1)
plt.title("RAW - axial")
plt.imshow(raw_resized[z], cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("RAW - coronal")
plt.imshow(raw_resized[:, y, :], cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("RAW - sagittal")
plt.imshow(raw_resized[:, :, x], cmap='gray')
plt.axis("off")


# --- MASK ---
plt.subplot(2, 3, 4)
plt.title("MASK - axial")
plt.imshow(mask_resized[z], cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("MASK - coronal")
plt.imshow(mask_resized[:, y, :], cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("MASK - sagittal")
plt.imshow(mask_resized[:, :, x], cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()