import os
import glob
import numpy as np
import SimpleITK as sitk # Required for loading .nrrd and .nii.gz files
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- Helper Function for Data Path Collection ---

def get_data_paths(base_dir, raw_dir, label_dir, raw_ext, label_ext):
    """
    Collects and pairs raw image and label mask file paths.
    
    Assumes matching filenames (excluding extension) between raw and label folders.
    
    Returns:
        tuple: (list of raw paths, list of label paths)
    """
    raw_path_pattern = os.path.join(base_dir, raw_dir, f"*{raw_ext}")
    all_raw_paths = sorted(glob.glob(raw_path_pattern))
    
    raw_paths_filtered = []
    all_label_paths = []
    
    for raw_path in all_raw_paths:
        # Extract the base filename (e.g., '19_nuclei_125')
        base_name = os.path.basename(raw_path).replace(raw_ext, '')
        
        # Construct the expected label path
        label_path = os.path.join(base_dir, label_dir, f"{base_name}{label_ext}")
        
        if os.path.exists(label_path):
            raw_paths_filtered.append(raw_path)
            all_label_paths.append(label_path)
        else:
            print(f"Warning: Corresponding label mask not found for {os.path.basename(raw_path)}")
    
    return raw_paths_filtered, all_label_paths


# --- PyTorch Dataset Class ---

class NucleusSegmentationDataset(Dataset):
    """
    A PyTorch Dataset for loading 3D confocal microscopy volumes (raw image and mask).
    """
    def __init__(self, raw_paths, label_paths, target_shape=(64, 64, 64)):
        """
        Args:
            raw_paths (list): List of file paths for raw 3D images.
            label_paths (list): List of file paths for corresponding segmentation masks.
            target_shape (tuple): The desired output shape (D, H, W).
        """
        self.raw_paths = raw_paths
        self.label_paths = label_paths
        self.target_shape = target_shape
        
        assert len(self.raw_paths) == len(self.label_paths), "Raw and label path lists must have the same length."

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.raw_paths)

    def __getitem__(self, idx):
        """Loads and preprocesses the i-th sample."""
        raw_path = self.raw_paths[idx]
        label_path = self.label_paths[idx]

        # 1. Load and Preprocess Raw Image
        image_volume = self._load_and_preprocess_volume(raw_path, is_mask=False)
        
        # 2. Load and Preprocess Mask
        mask_volume = self._load_and_preprocess_volume(label_path, is_mask=True)

        # 3. Convert to PyTorch Tensor format (C, D, H, W)
        # Original shape is (D, H, W). We add C=1 channel dimension.
        image_tensor = torch.from_numpy(image_volume[np.newaxis, ...]).float()
        mask_tensor = torch.from_numpy(mask_volume[np.newaxis, ...]).float()
        
        return image_tensor, mask_tensor

    def _load_and_preprocess_volume(self, file_path, is_mask):
        """
        Loads volume using SimpleITK, resamples it to target_shape, and normalizes/binarizes.
        """
        try:
            itk_image = sitk.ReadImage(file_path)
            volume = sitk.GetArrayFromImage(itk_image) # NumPy array (Z, Y, X)
        except Exception as e:
            # If loading fails, return a zero volume to prevent training crash
            print(f"Error loading {file_path}: {e}")
            return np.zeros(self.target_shape, dtype=np.float32)

        volume = volume.astype(np.float32)
        current_shape = volume.shape
        
        # Calculate zoom factors for resampling
        zoom_factors = [ts / cs for ts, cs in zip(self.target_shape, current_shape)]
        
        # Resample the volume
        if is_mask:
            # Use nearest neighbor interpolation (order=0) for masks
            resampled_volume = zoom(volume, zoom_factors, order=0, mode='nearest')
            # Binarization: Ensure mask is strictly 0 or 1
            resampled_volume = (resampled_volume > 0.5).astype(np.float32)
        else:
            # Use trilinear interpolation (order=3) for raw images
            resampled_volume = zoom(volume, zoom_factors, order=3, mode='constant')
            # Intensity normalization (Min-Max normalization)
            min_val = resampled_volume.min()
            max_val = resampled_volume.max()
            if (max_val - min_val) > 1e-6:
                resampled_volume = (resampled_volume - min_val) / (max_val - min_val)
            else:
                resampled_volume = np.zeros_like(resampled_volume) # Handle zero-variance
        
        # Ensure exact target shape (in case of minor zoom discrepancies)
        if resampled_volume.shape != self.target_shape:
             # Crop/pad to target shape
             temp_vol = np.zeros(self.target_shape, dtype=np.float32)
             d, h, w = self.target_shape
             rd, rh, rw = resampled_volume.shape
             
             d_slice = slice(0, min(d, rd))
             h_slice = slice(0, min(h, rh))
             w_slice = slice(0, min(w, rw))
             
             temp_vol[d_slice, h_slice, w_slice] = resampled_volume[d_slice, h_slice, w_slice]
             resampled_volume = temp_vol

        return resampled_volume


# --- Main Data Splitter Function ---

def get_train_val_loaders(base_dir, raw_dir, label_dir, raw_ext, label_ext, 
                          target_shape=(64, 64, 64), batch_size=4, val_split=0.15, num_workers=0):
    """
    Creates PyTorch DataLoaders for training and validation sets.
    """
    raw_paths, label_paths = get_data_paths(base_dir, raw_dir, label_dir, raw_ext, label_ext)
    
    if not raw_paths:
        raise FileNotFoundError(f"No paired data files found in {base_dir}. Cannot create DataLoaders.")

    # Split indices
    indices = np.arange(len(raw_paths))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=42)
    
    train_raw_paths = [raw_paths[i] for i in train_indices]
    train_label_paths = [label_paths[i] for i in train_indices]
    val_raw_paths = [raw_paths[i] for i in val_indices]
    val_label_paths = [label_paths[i] for i in val_indices]
    
    # Create Datasets
    train_dataset = NucleusSegmentationDataset(train_raw_paths, train_label_paths, target_shape)
    val_dataset = NucleusSegmentationDataset(val_raw_paths, val_label_paths, target_shape)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader, len(train_dataset), len(val_dataset)