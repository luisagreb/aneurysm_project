# from pathlib import Path
# import numpy as np
# import nibabel as nib
# import imageio.v2 as iio
# from skimage.exposure import rescale_intensity

# P = Path('/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/segmented_training/01Asc-180/no coll/01ASC-0180 -coll 60x DMSO48H-Zstack cell1/01ASC-0180 -coll 60x DMSO48H-Zstack cell1_allchannels.nii.gz')

# img = nib.load(str(P))
# arr = img.get_fdata(dtype=np.float32)      # (C, Z, Y, X)
# print("shape:", arr.shape, "min/max:", float(arr.min()), float(arr.max()))

# assert arr.ndim == 4 and arr.shape[0] in (3,4), "Expected (C,Z,Y,X)"

# for c in range(arr.shape[0]):
#     vol = arr[c]                               # (Z,Y,X)
#     mip = vol.max(axis=0)                      # (Y,X)
#     mid = vol[vol.shape[0]//2]                 # mid-Z slice (Y,X)
#     mip = rescale_intensity(mip, in_range="image", out_range=(0,255)).astype(np.uint8)
#     mid = rescale_intensity(mid, in_range="image", out_range=(0,255)).astype(np.uint8)
#     iio.imwrite(f"debug_mip_C{c}.png", mip)
#     iio.imwrite(f"debug_mid_C{c}.png", mid)
#     print(f"C{c} saved (MIP + mid).")

# for idx, v in enumerate(vols):
#     v_zyx = to_zyx(v)
#     show = rescale_intensity(v_zyx, in_range="image", out_range=(0, 1))
#     mip = (show.max(axis=0) * 255).astype(np.uint8)
#     iio.imwrite(f"debug_mip_ch{idx}.png", mip)
#     midz = v_zyx.shape[0]//2
#     iio.imwrite(f"debug_midslice_ch{idx}.png",
#                 (rescale_intensity(v_zyx[midz], in_range="image", out_range=(0,1))*255).astype(np.uint8))
# print("Saved debug_mip_ch*.png and debug_midslice_ch*.png")
#!/usr/bin/env python3

import nibabel as nib
import nrrd
import numpy as np
import os
from skimage.filters import threshold_multiotsu
from scipy.ndimage import label, gaussian_filter

def segment_nucleus(data: np.ndarray) -> np.ndarray:
    """
    Performs robust 3D segmentation using Multi-Otsu to focus on the highest-intensity 
    (darker red) region, which corresponds to the nucleus. Returns a clear 0/255 mask.
    """
    print("Starting robust nucleus segmentation using Multi-Otsu (0/255 Binary Mask).")
    
    # 1. Preprocessing and Smoothing
    GAUSSIAN_SIGMA = 0.8
    data_float = data.astype(np.float32)
    smoothed_data = gaussian_filter(data_float, sigma=GAUSSIAN_SIGMA)

    # 2. Focus on non-zero, high-intensity regions for threshold calculation
    # We only care about voxels that are brighter than a low noise floor (e.g., 5% of the max)
    max_val = np.max(smoothed_data)
    MIN_INTENSITY_FLOOR = max_val * 0.05
    
    analysis_data = smoothed_data[smoothed_data > MIN_INTENSITY_FLOOR]
    
    if analysis_data.size < 100:
        print("Warning: Insufficient bright voxels for robust Multi-Otsu. Returning empty mask.")
        return np.zeros_like(data, dtype=np.uint8)

    # 3. Apply Multi-Otsu thresholding to find 3 distinct groups (background, cell, nucleus)
    try:
        # We expect 3 classes: background/noise, cytoplasm, nucleus
        thresholds = threshold_multiotsu(analysis_data, classes=3)
        # The highest threshold should separate the nucleus (highest intensity) from the cell/background
        nucleus_threshold = thresholds[-1] 
    except Exception as e:
        print(f"Error in Multi-Otsu calculation: {e}. Falling back to single Otsu.")
        from skimage.filters import threshold_otsu
        nucleus_threshold = threshold_otsu(analysis_data)
    
    # 4. Create binary mask based on the selected high threshold
    # We apply the threshold directly to the full smoothed data
    # Note: Multi-Otsu thresholds are absolute values relative to the data range
    binary_mask = smoothed_data > nucleus_threshold
    
    # 5. Label connected components and select the largest (the nucleus)
    labeled_array, num_features = label(binary_mask)
    
    if num_features == 0:
        print("No high-intensity features found after thresholding. Returning empty mask.")
        return np.zeros_like(data, dtype=np.uint8)

    component_sizes = np.bincount(labeled_array.flat)[1:] 
    
    if component_sizes.size == 0:
        return np.zeros_like(data, dtype=np.uint8)

    largest_component_label = np.argmax(component_sizes) + 1 
    
    # Final segmentation mask: 255 for nucleus, 0 for background
    segmentation_mask = (labeled_array == largest_component_label).astype(np.uint8) * 255
    
    print(f"Segmentation complete. Found {num_features} potential features. Largest selected.")
    return segmentation_mask


def convert_nii_to_nrrd(input_nii_path: str, output_nrrd_path: str):
    """
    Loads a NIfTI file, segments the nucleus, and saves the resulting 
    0/255 binary segmentation mask as a NRRD file.
    """
    print(f"Loading NIfTI file: {input_nii_path}")

    try:
        # 1. Load the NIfTI file
        img = nib.load(input_nii_path)
        data = img.get_fdata() # Original intensity data
        affine = img.affine
        
        # 2. Segment the nucleus to get the 0/255 binary mask
        output_data = segment_nucleus(data)

        # Prepare the output data
        output_data = np.ascontiguousarray(output_data).astype(np.uint8)
        
        # 3. Derive space information (use full directions matrix)
        space_directions = affine[:3, :3]
        
        # 4. Prepare the NRRD header
        header = {
            'space': 'left-posterior-superior', 
            'space directions': [tuple(d) for d in space_directions],
            'space origin': tuple(affine[:3, 3]),
            'dimension': output_data.ndim,
            'kind': 'label', 
            'encoding': 'gzip', 
            'data type': 'uint8' # Max value 255
        }

        # 5. Write the NRRD file
        print(f"Writing NRRD file containing the 0/255 binary mask: {output_nrrd_path}")
        nrrd.write(output_nrrd_path, output_data, header=header)

        print("\n--- Process Complete ---")
        print(f"Output saved to: {output_nrrd_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_nii_path}")
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")


# --- Configuration ---
# 

# 1. Input NIfTI file path from your request
INPUT_NII_FILE = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/segmented_training/01Asc-180/+coll/01ASC-0180 +coll 60x DMSO48h-Zstack cell2/01ASC-0180 +coll 60x DMSO48h-Zstack cell2_ch1.nii"

# 2. Output NRRD file path - Final name that matches the structure of your successful example
OUTPUT_NRRD_FILE = "Segmentation_cell2.nucleus.seg.nrrd"


# --- Execute Conversion ---
if __name__ == "__main__":
    convert_nii_to_nrrd(INPUT_NII_FILE, OUTPUT_NRRD_FILE)