import numpy as np
import nibabel as nib # <-- The critical import for NIfTI files
from scipy.ndimage import gaussian_filter
from time import time
import os
import glob
import logging

# --- New Import for NRRD Output ---
try:
    import nrrd
except ImportError:
    # If the user hasn't installed pynrrd, we cannot proceed.
    raise ImportError("The 'nrrd' library (pynrrd) is required for NRRD output. Please ensure it is installed: pip install pynrrd")

# Import required for Otsu's thresholding.
try:
    from skimage.filters import threshold_otsu 
except ImportError:
    # Define a dummy function if skimage is not available
    def threshold_otsu(image):
        # Fallback to mean threshold
        return np.mean(image)
    logging.warning("scikit-image (for Otsu's thresholding) not found. Falling back to mean threshold.")


# --- Configuration ---
GAUSSIAN_SIGMA = 1.0  
OUTPUT_DIR = "segmented_output"
# This pattern ONLY searches for files containing 'ch1' (nucleus channel).
INPUT_PATTERNS = [
    "./segmented_training/**/*ch1.nii", 
    "./segmented_training/**/*ch1.nii.gz"
] 

# Set up logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_segmentation(data_32):
    """
    Applies Gaussian smoothing and then performs segmentation using Otsu's method 
    (or mean threshold as fallback).
    """
    logging.info(f"Applying 3D Gaussian filter with sigma={GAUSSIAN_SIGMA}...")
    smoothed_data = gaussian_filter(data_32, sigma=GAUSSIAN_SIGMA)
    
    # 1. Otsu's Thresholding
    try:
        threshold = threshold_otsu(smoothed_data) 
        logging.info(f"Calculated Otsu's Threshold: {threshold:.4f}")
    except ValueError as e:
        logging.warning(f"Threshold calculation failed ({e}). Falling back to mean threshold.")
        threshold = np.mean(smoothed_data)

    # 2. Apply Threshold
    # Ensure the output mask is an integer type (uint8 is standard for segmentation masks)
    segmented_data = (smoothed_data > threshold).astype(np.uint8)
    return segmented_data

def process_nii_file(input_filepath):
    """
    Loads a NIfTI file, processes it, and saves the segmented result as NRRD.
    """
    try:
        logging.info(f"Starting processing: {os.path.basename(input_filepath)}")
        start_time = time()

        # 1. Load Data (uses the fixed 'nib' alias)
        img = nib.load(input_filepath)
        data = img.get_fdata()
        data_32 = data.astype(np.float32)
        load_time = time()
        
        # We expect 3D data (Z, Y, X). Skip if not 3D.
        if data_32.ndim != 3:
            logging.warning(f"Skipping file: {os.path.basename(input_filepath)} has {data_32.ndim} dimensions. Expected 3D data.")
            return

        logging.info(f"Data loaded in: {load_time - start_time:.2f}s. Shape: {data_32.shape}, DType: {data_32.dtype}")

        # 2. Perform Segmentation (Gaussian + Threshold)
        segment_start = time()
        segmented_data = perform_segmentation(data_32)
        segment_time = time()
        logging.info(f"Segmentation completed in: {segment_time - segment_start:.2f}s.")

        # 3. Prepare NRRD Header (to preserve spatial orientation/spacing)
        # Calculate voxel size (spacings) from the NIfTI affine matrix
        voxel_size = np.sqrt(np.sum(img.affine[:3, :3]**2, axis=0))
        
        # Set NRRD header information
        nrrd_header = {
            'spacings': voxel_size,
            'space': 'left-posterior-superior' 
        }

        # 4. Save Segmented Image as NRRD
        output_filename = "segmented_" + os.path.basename(input_filepath).replace('.nii.gz', '.nrrd').replace('.nii', '.nrrd')
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        
        # Save the segmented data and header to the NRRD file
        nrrd.write(output_filepath, segmented_data, header=nrrd_header)
        
        save_time = time()
        logging.info(f"Result saved to {output_filename} in: {save_time - segment_time:.2f}s.")
        logging.info(f"Finished processing {os.path.basename(input_filepath)}. Total time: {save_time - start_time:.2f}s.\n")

    except Exception as e:
        logging.error(f"Error processing {input_filepath}: {e}")

def main():
    """Main function to find files and orchestrate processing."""
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    logging.info(f"Scanning for NIfTI files matching *ch1* in: ./segmented_training/...")
    
    input_files = []
    for pattern in INPUT_PATTERNS:
        try:
            found_files = glob.glob(pattern, recursive=True)
            input_files.extend(found_files)
        except TypeError:
            input_files.extend(glob.glob(pattern))

    input_files = sorted(list(set(input_files))) 
    
    if not input_files:
        logging.warning("No *ch1* NIfTI files found. Please verify naming convention in 'segmented_training'.")
        return

    logging.info(f"Found {len(input_files)} file(s) to process.")

    total_pipeline_start = time()
    for file in input_files:
        process_nii_file(file)

    total_pipeline_end = time()
    logging.info(f"--- Pipeline finished processing all files. ---")
    logging.info(f"Total pipeline execution time: {total_pipeline_end - total_pipeline_start:.2f}s.")

if __name__ == "__main__":
    main()
