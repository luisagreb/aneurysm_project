# convert_image.py
# Loads and inspects .nii.gz microscopy data (already converted)

import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Example path — replace with any cell folder you want to analyze
cell_path = Path(
    "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/segmented_training/"
    "01Asc-180/+coll/01ASC-0180 +coll 60x DMSO48h-Zstack cell1/"
    "01ASC-0180 +coll 60x DMSO48h-Zstack cell1_allchannels.nii.gz"
)

# Load the image
nii = nib.load(str(cell_path))
data = nii.get_fdata()
print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")

# If it’s a multi-channel stack (e.g., last dimension = channels)
if data.ndim == 4:
    n_channels = data.shape[-1]
    print(f"Number of channels: {n_channels}")
    for c in range(n_channels):
        mid_z = data.shape[2] // 2
        plt.imshow(data[:, :, mid_z, c], cmap="gray")
        plt.title(f"Channel {c+1} — mid Z slice")
        plt.axis("off")
        plt.show()
else:
    print("Single-channel data")
    mid_z = data.shape[2] // 2
    plt.imshow(data[:, :, mid_z], cmap="gray")
    plt.title("Mid Z slice")
    plt.axis("off")
    plt.show()