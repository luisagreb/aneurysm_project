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

    Raw example : 01ASC-0180_nonAneurysm_coll_nucleus_cell1.nii
    Label       : 01ASC-0180_nonAneurysm_coll_nucleusLabel_cell1.nrrd
    """
    raw_path_pattern = os.path.join(base_dir, raw_dir, f"*{raw_ext}")
    all_raw_paths = sorted(glob.glob(raw_path_pattern))

    raw_paths_filtered = []
    corresponding_label_paths = []

    for raw_path in all_raw_paths:
        raw_filename = os.path.basename(raw_path)

        # strip extension
        if raw_filename.endswith(raw_ext):
            base_name = raw_filename[:-len(raw_ext)]
        else:
            base_name = os.path.splitext(raw_filename)[0]

        # ---- build possible label names ----
        candidates = []

        # 1) same basename (generic case)
        candidates.append(base_name + label_ext)

        # 2) special case: nucleus -> nucleusLabel
        if "nucleus_" in base_name:
            label_base = base_name.replace("nucleus_", "nucleusLabel_")
            candidates.append(label_base + label_ext)

        label_path = None
        for lab_fn in candidates:
            cand_path = os.path.join(base_dir, label_dir, lab_fn)
            if os.path.exists(cand_path):
                label_path = cand_path
                break

        if label_path is not None:
            raw_paths_filtered.append(raw_path)
            corresponding_label_paths.append(label_path)
        else:
            print("=" * 60, file=sys.stderr)
            print(f"!!! WARNING: Label mask not found for RAW file: {raw_filename}", file=sys.stderr)
            print("!!! Tried candidates:", file=sys.stderr)
            for lab_fn in candidates:
                print("   ", os.path.join(base_dir, label_dir, lab_fn), file=sys.stderr)
            print("=" * 60, file=sys.stderr, flush=True)

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
            batch_indices = indiceis[start_idx:end_idx]
            
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