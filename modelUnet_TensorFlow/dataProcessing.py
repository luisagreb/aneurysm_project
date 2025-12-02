import os
import glob
import numpy as np
import SimpleITK as sitk # Required for loading .nrrd and .nii.gz files
from scipy.ndimage import zoom
import tensorflow as tf
import sys 

# --- Helper Functions for Data Loading ---

def get_data_paths(base_dir, raw_dir, label_dir, raw_ext, label_ext):
    """
    Collects and pairs raw image and label mask file paths.
    
    It assumes the label mask file shares the exact same base name as the raw file,
    only differing in the file extension.
        
    Args:
        base_dir (str): Root directory for the data.
        raw_dir (str): Subdirectory for raw images.
        label_dir (str): Subdirectory for label masks.
        raw_ext (str): Extension for raw images (e.g., '.nii.gz').
        label_ext (str): Extension for label masks (e.g., '.nrrd').
        
    Returns:
        tuple: (list of raw paths, list of label paths)
    """
    # Use glob to find all raw files
    raw_path_pattern = os.path.join(base_dir, raw_dir, f"*{raw_ext}")
    all_raw_paths = sorted(glob.glob(raw_path_pattern))
    
    raw_paths_filtered = []
    corresponding_label_paths = []
    
    # Use a set of existing label file base names for faster lookup (optional optimization)
    # all_label_filenames = set(os.listdir(os.path.join(base_dir, label_dir)))
    
    for raw_path in all_raw_paths:
        raw_filename = os.path.basename(raw_path)
        
        # 1. Extract the base name by removing the extension using slicing
        if raw_filename.endswith(raw_ext):
            base_name = raw_filename[:-len(raw_ext)]
        else:
            base_name = raw_filename.replace(raw_ext, '')
        
        # 2. Construct the expected label filename: BASE_NAME + LABEL_EXTENSION (NO _SEG suffix)
        # We are removing the assumption of a '_SEG' suffix here.
        label_filename = f"{base_name}{label_ext}"
        label_path = os.path.join(base_dir, label_dir, label_filename)
        
        if os.path.exists(label_path):
            raw_paths_filtered.append(raw_path)
            corresponding_label_paths.append(label_path)
        else:
            # --- DEBUG PRINT: Now includes a clear separator for visibility ---
            print("="*60, file=sys.stderr)
            print(f"!!! WARNING: Label mask not found for RAW file: {raw_filename}", file=sys.stderr)
            print(f"!!! EXPECTED PATH: {label_path}", file=sys.stderr)
            print("="*60, file=sys.stderr, flush=True)
            
    return raw_paths_filtered, corresponding_label_paths

def load_and_preprocess_volume(file_path, target_shape, is_mask=False):
    """
    Loads a 3D volume, resizes it to the target shape, and normalizes/binarizes it.
    """
    try:
        # Load the volume using SimpleITK
        itk_image = sitk.ReadImage(file_path)
        # GetArrayFromImage returns numpy array in ZYX order, so we need to transpose to DHW
        volume = sitk.GetArrayFromImage(itk_image).transpose(2, 1, 0).astype(np.float32)

        # Calculate zoom factor for resizing 
        current_shape = volume.shape # (D, H, W)
        zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
        
        # Resize volume (Order 0 for nearest neighbor (masks), Order 3 for cubic spline (images))
        volume_resized = zoom(volume, zoom_factors, order=0 if is_mask else 3)
        
        if is_mask:
            # Binarize mask: convert labels to 0 or 1
            volume_processed = (volume_resized > 0).astype(np.float32)
        else:
            # Normalize raw data to [0, 1]
            min_val = np.min(volume_resized)
            max_val = np.max(volume_resized)
            if max_val > min_val:
                volume_processed = (volume_resized - min_val) / (max_val - min_val)
            else:
                volume_processed = np.zeros_like(volume_resized)


        # Add channel dimension (D, H, W, 1)
        return np.expand_dims(volume_processed, axis=-1)

    except Exception as e:
        print(f"Error loading or preprocessing {file_path}: {e}", file=sys.stderr)
        # Return a zero array of the target shape in case of error
        return np.zeros(target_shape + (1,), dtype=np.float32)

# --- Data Generator Function (Not changed) ---

def prepare_data_generator(raw_paths, label_paths, target_shape, batch_size, num_classes, shuffle=True):
    """
    A generator that yields batches of 3D image and mask data.
    """
    
    data_size = len(raw_paths)
    indices = np.arange(data_size)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, data_size, batch_size):
            end_idx = min(start_idx + batch_size, data_size)
            batch_indices = indices[start_idx:end_idx]
            
            # Initialize empty arrays for the batch
            batch_raw = np.zeros((len(batch_indices),) + target_shape + (1,), dtype=np.float32)
            batch_label = np.zeros((len(batch_indices),) + target_shape + (num_classes,), dtype=np.float32)
            
            for i, data_idx in enumerate(batch_indices):
                raw_path = raw_paths[data_idx]
                label_path = label_paths[data_idx]
                
                # Load and preprocess image
                raw_volume = load_and_preprocess_volume(raw_path, target_shape, is_mask=False)
                batch_raw[i] = raw_volume
                
                # Load and preprocess mask
                label_volume = load_and_preprocess_volume(label_path, target_shape, is_mask=True)
                
                # NOTE: Assuming NUM_CLASSES=1 in train_model.py, we don't one-hot encode.
                # If you change NUM_CLASSES > 1, this needs to be updated for categorical encoding.
                batch_label[i] = label_volume 

            yield (batch_raw, batch_label)