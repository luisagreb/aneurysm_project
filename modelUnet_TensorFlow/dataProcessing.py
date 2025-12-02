import os
import glob
import numpy as np
import SimpleITK as sitk # Required for loading .nrrd and .nii.gz files
from scipy.ndimage import zoom
import tensorflow as tf

# --- Helper Functions for Data Loading ---

def get_data_paths(base_dir, raw_dir, label_dir, raw_ext, label_ext):
    """
    Collects and pairs raw image and label mask file paths.
    
    Assumes matching filenames (excluding extension) between raw and label folders.
    
    Returns:
        tuple: (list of raw paths, list of label paths)
    """
    # 1. Collect all raw files
    raw_path_pattern = os.path.join(base_dir, raw_dir, f"*{raw_ext}")
    all_raw_paths = sorted(glob.glob(raw_path_pattern))
    
    all_label_paths = []
    raw_paths_filtered = []
    
    for raw_path in all_raw_paths:
        # Extract the base filename by safely removing the raw extension
        base_name = os.path.basename(raw_path)
        if base_name.endswith(raw_ext):
            # This handles both .nii.gz and .nii by matching raw_ext exactly
            base_name = base_name[:-len(raw_ext)]
        else:
            # Fallback for unexpected case, though unlikely
            continue
        
        # Construct the expected label path
        label_path = os.path.join(base_dir, label_dir, f"{base_name}{label_ext}")
        
        if os.path.exists(label_path):
            all_label_paths.append(label_path)
            raw_paths_filtered.append(raw_path)
        else:
            print(f"Warning: Corresponding label mask not found for {os.path.basename(raw_path)}")
    
    return raw_paths_filtered, all_label_paths


def load_and_preprocess_volume(file_path, target_shape, is_mask=False):
    """
    Loads a 3D volume, handles potential SimpleITK spacing errors, normalizes,
    and resamples to the target shape.
    """
    try:
        # 1. Attempt standard read, which should capture the metadata
        image = sitk.ReadImage(file_path)
        volume = sitk.GetArrayFromImage(image).astype(np.float32)

    except RuntimeError as e:
        # 2. Handle known NRRD spacing error by returning a zeroed volume
        if "nrrd spacing" in str(e) and file_path.lower().endswith('.nrrd'):
            print(f"    ⚠️ CRITICAL NRRD ERROR: SimpleITK failed to read {os.path.basename(file_path)} due to spacing metadata.")
            # Since the image is unreadable, we must skip it by returning an empty, correctly shaped array.
            # This ensures the generator doesn't crash, but the data is lost.
            # This usually means the predicted .nrrd file is corrupted (which happens if the model saves data incorrectly).
            return np.zeros(target_shape + (1,), dtype=np.float32)

        else:
            # Re-raise other unexpected errors
            raise e

    # --- Preprocessing after successful load ---
    
    # Resample
    current_shape = volume.shape
    zoom_factors = np.array(target_shape) / np.array(current_shape)
    
    # Resampling function: nearest for mask, linear/cubic for raw
    order = 0 if is_mask else 3 
    
    volume_resampled = zoom(volume, zoom_factors, order=order, prefilter=True)

    # Normalization (for raw images only)
    if not is_mask:
        # Z-score normalization
        if np.std(volume_resampled) > 1e-6:
            volume_resampled = (volume_resampled - np.mean(volume_resampled)) / np.std(volume_resampled)
        else:
            volume_resampled = volume_resampled - np.mean(volume_resampled)
            
    else:
        # For masks, ensure the output is binary (0 or 1)
        volume_resampled = (volume_resampled > 0.5).astype(np.float32)

    # Expand dimensions for the channel dimension (TensorFlow requires [D, H, W, C])
    return np.expand_dims(volume_resampled, axis=-1)


# --- Data Generator Function ---
# (The rest of this file, including prepare_data_generator, remains the same as previously provided)

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
                batch_label[i] = label_volume
            
            yield (batch_raw, batch_label)