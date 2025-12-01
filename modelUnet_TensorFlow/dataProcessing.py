import os
import glob
import numpy as np
import SimpleITK as sitk # Required for loading .nrrd and .nii.gz files
from scipy.ndimage import zoom
import tensorflow as tf
import sys 

# --- Helper Functions for Data Loading ---

import re

def canonical_key(name_no_ext: str) -> str:
    """
    Normalize key (used to match raw and label filenames).
    """
    s = name_no_ext
    s = s.replace("nonAneursym", "nonAneurysm")
    s = s.replace("nucleusLabel", "nucleus")
    return s

def get_data_paths(base_dir, raw_dir, label_dir, raw_ext, label_ext):
    """
    Collects and pairs raw image and label mask file paths.

    Uses a 'canonical_key' so that:
      - RAW:   01ASC-0180_nonAneurysm_coll_nucleus_cell1.nii.gz
      - LABEL: 01ASC-0180_nonAneurysm_coll_nucleusLabel_cell1.nrrd
    are treated as a matching pair.

    Returns:
        (raw_paths, label_paths) — same length, same order.
    """
    raw_dir_full = os.path.join(base_dir, raw_dir)
    label_dir_full = os.path.join(base_dir, label_dir)

    # 1. Collect raw files
    raw_pattern = os.path.join(raw_dir_full, f"*{raw_ext}")
    raw_files = sorted(glob.glob(raw_pattern))

    # 2. Collect label files
    label_pattern = os.path.join(label_dir_full, f"*{label_ext}")
    label_files = sorted(glob.glob(label_pattern))

    if not raw_files:
        print(f"!!! No RAW files found with pattern: {raw_pattern}", file=sys.stderr)
    if not label_files:
        print(f"!!! No LABEL files found with pattern: {label_pattern}", file=sys.stderr)

    # 3. Build dict: canonical_key -> path
    def build_dict(paths, ext):
        d = {}
        for p in paths:
            fname = os.path.basename(p)
            if fname.endswith(ext):
                base = fname[:-len(ext)]
            else:
                base = os.path.splitext(fname)[0]
            key = canonical_key(base)
            d[key] = p
        return d

    raw_dict = build_dict(raw_files, raw_ext)
    label_dict = build_dict(label_files, label_ext)

    # 4. Intersection of keys
    common_keys = sorted(set(raw_dict.keys()) & set(label_dict.keys()))

    if not common_keys:
        print("!!! ERROR: No overlapping RAW/LABEL keys found.", file=sys.stderr)
        print(f"    Example RAW keys:   {list(raw_dict.keys())[:3]}", file=sys.stderr)
        print(f"    Example LABEL keys: {list(label_dict.keys())[:3]}", file=sys.stderr)
        return [], []

    raw_paths = [raw_dict[k] for k in common_keys]
    label_paths = [label_dict[k] for k in common_keys]

    print(f"✅ Found {len(raw_paths)} paired RAW/LABEL volumes.")
    print("   Example pair:")
    print(f"   RAW:   {raw_paths[0]}")
    print(f"   LABEL: {label_paths[0]}")

    return raw_paths, label_paths

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