import nibabel as nib
import nrrd
import numpy as np
import os
import glob
from skimage.filters import threshold_multiotsu, threshold_otsu
from scipy.ndimage import label, gaussian_filter

# --- Global Configuration ---
# NOTE: Replace these with your actual root directories.
INPUT_ROOT_DIR = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/segmented_training"
OUTPUT_ROOT_DIR = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/segmented_auto_nucleus"

# --- Target Channel Definitions (New Automatic Structure) ---
# Define ALL channels you want to process in one pass.
# Each dictionary represents a segmentation task.
CHANNEL_CONFIGS = [
    {
        "structure": "nucleus",
        "suffix": "_ch1.nii.gz",
        "output_mask_name": "nucleus.seg.nrrd" 
    },
    # Add other channels here if you switch segmentation algorithms:
    # {
    #     "structure": "actin",
    #     "suffix": "_ch2.nii.gz",
    #     "output_mask_name": "actin.seg.nrrd"
    # },
    # {
    #     "structure": "mitochondria",
    #     "suffix": "_ch3.nii.gz",
    #     "output_mask_name": "mitochondria.seg.nrrd"
    # }
]

# --- Segmentation Core Function (UPDATED: Adaptive Threshold Selection) ---
def segment_nucleus_3class(data: np.ndarray) -> np.ndarray:
    """
    Performs robust 3D segmentation using 3-class Multi-Otsu, adaptively selecting 
    the threshold based on the resulting object size. Returns a clear 0/255 mask.
    """
    print("Starting robust nucleus segmentation using 3-Class Multi-Otsu (0/255 Binary Mask).")
    
    # 1. Preprocessing and Smoothing
    GAUSSIAN_SIGMA = 0.8
    data_float = data.astype(np.float32)
    smoothed_data = gaussian_filter(data_float, sigma=GAUSSIAN_SIGMA)

    # 2. Focus on high-intensity regions for threshold calculation
    max_val = np.max(smoothed_data)
    MIN_INTENSITY_FLOOR = max_val * 0.05
    
    analysis_data = smoothed_data[smoothed_data > MIN_INTENSITY_FLOOR]
    
    if analysis_data.size < 100:
        print("Warning: Insufficient bright voxels for robust Multi-Otsu. Returning empty mask.")
        return np.zeros_like(data, dtype=np.uint8)

    # 3. Apply Multi-Otsu thresholding to find 3 distinct groups
    try:
        # We expect 3 classes: background/noise (0), cytoplasm (1), nucleus (2)
        thresholds = threshold_multiotsu(analysis_data, classes=3)
        # Sort thresholds to ensure consistent access: [T1, T2]
        thresholds.sort()
        
        # Define candidate thresholds
        # T_HIGH: Highest threshold, ideally isolating the nucleus core
        T_HIGH = thresholds[-1] 
        # T_MID: Middle threshold, likely capturing the whole cell or dim nucleus
        T_MID = thresholds[-2] if len(thresholds) > 1 else T_HIGH # Fallback if only 1 threshold

    except Exception as e:
        print(f"Error in Multi-Otsu calculation: {e}. Falling back to single Otsu.")
        T_HIGH = threshold_otsu(analysis_data)
        T_MID = T_HIGH # Only one choice on fallback

    
    # 4. Adaptive Threshold Selection
    
    # Set a heuristic minimum acceptable size (e.g., 500 voxels, adjust if needed)
    MIN_NUCLEUS_SIZE = 500 
    selected_threshold = T_HIGH
    
    # Function to test a threshold and get the largest component size
    def get_largest_component_size(thresh):
        binary_mask = smoothed_data > thresh
        labeled_array, _ = label(binary_mask)
        component_sizes = np.bincount(labeled_array.flat)[1:] 
        return np.max(component_sizes) if component_sizes.size > 0 else 0

    # 4a. Test the HIGH threshold first (default for ideal nucleus isolation)
    largest_size_high = get_largest_component_size(T_HIGH)
    
    if largest_size_high < MIN_NUCLEUS_SIZE and T_HIGH != T_MID:
        # If T_HIGH yields an object that is too small, try the MID threshold
        largest_size_mid = get_largest_component_size(T_MID)
        
        if largest_size_mid > largest_size_high:
            selected_threshold = T_MID
            print(f"Adaptive switch: High threshold ({largest_size_high} voxels) too small. Using mid threshold.")
        else:
             print(f"Warning: Both high and mid thresholds yielded small objects.")
    
    # 5. Create final mask based on the selected threshold
    binary_mask = smoothed_data > selected_threshold
    
    # 6. Label connected components and select the largest
    labeled_array, num_features = label(binary_mask)
    
    if num_features == 0 or np.bincount(labeled_array.flat).size <= 1:
        print("No dominant feature found after labeling. Returning empty mask.")
        return np.zeros_like(data, dtype=np.uint8)

    component_sizes = np.bincount(labeled_array.flat)[1:] 
    
    if component_sizes.size == 0:
        return np.zeros_like(data, dtype=np.uint8)

    largest_component_label = np.argmax(component_sizes) + 1 
    
    # Final segmentation mask: 255 for nucleus, 0 for background
    segmentation_mask = (labeled_array == largest_component_label).astype(np.uint8) * 255
    
    print(f"Segmentation complete. Selected threshold resulted in object size: {np.sum(segmentation_mask > 0)} voxels.")
    return segmentation_mask


def process_single_file(input_nii_path: str, output_nrrd_path: str, relative_path: str, structure_name: str):
    """Loads, segments, and saves a single NIfTI file."""
    
    print(f"\n--- Processing {structure_name} for file: {relative_path} ---")
    try:
        # 1. Load the NIfTI file
        img = nib.load(input_nii_path)
        data = img.get_fdata()
        affine = img.affine
        
        # 2. Segment the data
        # Now uses the improved 3-class segmentation method
        if structure_name == "nucleus":
            output_data = segment_nucleus_3class(data)
        else:
            # Fallback or placeholder for other channels like actin/mito (future deep learning steps)
            print(f"Warning: No dedicated segmentation method for {structure_name}. Using nucleus method as fallback.")
            output_data = segment_nucleus_3class(data)

        # Ensure output directory exists and save
        os.makedirs(os.path.dirname(output_nrrd_path), exist_ok=True)
        
        output_data = np.ascontiguousarray(output_data).astype(np.uint8)
        space_directions = affine[:3, :3]
        
        header = {
            'space': 'left-posterior-superior', 
            'space directions': [tuple(d) for d in space_directions],
            'space origin': tuple(affine[:3, 3]),
            'dimension': output_data.ndim,
            'kind': 'label', 
            'encoding': 'gzip', 
            'data type': 'uint8'
        }

        nrrd.write(output_nrrd_path, output_data, header=header)
        print(f"SUCCESS: Mask saved to {output_nrrd_path}. Output size: {output_data.shape}")

    except Exception as e:
        print(f"ERROR: Failed to process {relative_path}. Reason: {e}")


def batch_process_segmentation(input_dir: str, output_dir: str, configs: list):
    """
    Traverses the input directory recursively and processes files based on the provided configs.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    total_files_processed = 0

    for config in configs:
        suffix = config["suffix"]
        output_mask_name = config["output_mask_name"]
        structure_name = config["structure"]
        
        print(f"\n############################################################")
        print(f"### Starting Batch for Structure: {structure_name.upper()} ({suffix}) ###")
        print(f"############################################################")

        # Key to recursive search: glob.glob with "**" and recursive=True
        search_path = os.path.join(input_dir, "**", f"*{suffix}")
        nii_files = glob.glob(search_path, recursive=True)

        if not nii_files:
            print(f"No NIfTI files found matching the suffix '{suffix}' in {input_dir}")
            continue

        print(f"Found {len(nii_files)} NIfTI files for {structure_name}...")

        for nii_path in nii_files:
            # 1. Get relative path (e.g., 'Subject_A/+coll/cell_01_ch1.nii.gz')
            relative_path = os.path.relpath(nii_path, input_dir)
            
            # 2. Construct the output NRRD file path
            # Removes the channel suffix (e.g., removes '_ch1.nii.gz')
            base_name = os.path.basename(nii_path).replace(suffix, "")
            nrrd_filename = base_name + "." + output_mask_name
            
            # Construct the directory structure in the output folder
            output_dir_path = os.path.join(output_dir, os.path.dirname(relative_path))
            output_nrrd_path = os.path.join(output_dir_path, nrrd_filename)
            
            process_single_file(nii_path, output_nrrd_path, relative_path, structure_name)
            total_files_processed += 1

    print(f"\nBatch processing complete. Total files processed: {total_files_processed}")


if __name__ == "__main__":
    batch_process_segmentation(
        input_dir=INPUT_ROOT_DIR,
        output_dir=OUTPUT_ROOT_DIR,
        configs=CHANNEL_CONFIGS
    )