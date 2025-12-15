import os
import numpy as np
import nrrd
from pathlib import Path
from scipy import ndimage
try:
    from skimage.filters import threshold_otsu
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Otsu thresholding will not be available.")

def calculate_dynamic_threshold(channel_data, method='otsu', **kwargs):
    """
    Calculate a dynamic threshold based on the image statistics.
    
    Args:
        channel_data: 3D numpy array (Z, Y, X)
        method: Thresholding method - 'otsu', 'percentile', 'mean_std', or 'fixed'
        **kwargs: Additional parameters for specific methods:
            - percentile: percentile value (default: 95)
            - mean_std_multiplier: multiplier for mean + std (default: 2.0)
            - fixed_value: fixed threshold value (for 'fixed' method)
    
    Returns:
        Calculated threshold value
    """
    if method == 'otsu':
        if not HAS_SKIMAGE:
            raise ImportError("scikit-image required for Otsu thresholding. Install with: pip install scikit-image")
        # Flatten the 3D array for Otsu
        flat_data = channel_data.flatten()
        # Remove zeros to avoid bias
        non_zero = flat_data[flat_data > 0]
        if len(non_zero) == 0:
            return np.max(channel_data)  # Fallback if all zeros
        threshold = threshold_otsu(non_zero)
        return float(threshold)
    
    elif method == 'percentile':
        percentile = kwargs.get('percentile', 95)
        threshold = np.percentile(channel_data, percentile)
        return float(threshold)
    
    elif method == 'mean_std':
        multiplier = kwargs.get('mean_std_multiplier', 2.0)
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        threshold = mean_val + (multiplier * std_val)
        return float(threshold)
    
    elif method == 'fixed':
        return float(kwargs.get('fixed_value', 250))
    
    else:
        raise ValueError(f"Unknown threshold method: {method}. Choose from: 'otsu', 'percentile', 'mean_std', 'fixed'")


def create_mask_with_threshold(channel_data, threshold, remove_small_objects=True, min_size=50):
    """
    Create a binary mask from channel data using a specific threshold.

    Args:
        channel_data: 3D numpy array (Z, Y, X)
        threshold: threshold value
        remove_small_objects: whether to remove small objects from mask
        min_size: minimum size of objects to keep (in voxels)

    Returns:
        Binary mask as uint8 array
    """
    print(f"    Using threshold: {threshold}")

    # Create binary mask
    mask = channel_data > threshold

    # Remove small objects if requested
    if remove_small_objects:
        # Apply morphological opening to clean up noise
        mask = ndimage.binary_opening(mask, structure=np.ones((2, 2, 2)))

        # Label connected components
        labeled_mask, num_features = ndimage.label(mask)

        # Remove small connected components
        for i in range(1, num_features + 1):
            if np.sum(labeled_mask == i) < min_size:
                mask[labeled_mask == i] = 0

        print(f"    Cleaned up small objects (min size: {min_size} voxels)")

    # Convert to uint8 (0 or 255)
    mask = (mask * 255).astype(np.uint8)

    return mask


def process_masks_with_custom_thresholds(input_dir, channel_thresholds,
                                        remove_small_objects=True, min_size=50):
    """
    Process all NRRD files and create masks with custom or dynamic thresholds per channel.

    Args:
        input_dir: Directory containing subdirectories with NRRD files
        channel_thresholds: Dictionary mapping channel index to threshold config.
                          Can be:
                          - Fixed value: {0: 250, 1: 300}
                          - Dynamic method: {0: {'method': 'otsu'}, 
                                            1: {'method': 'percentile', 'percentile': 95},
                                            2: {'method': 'mean_std', 'mean_std_multiplier': 2.0}}
        remove_small_objects: Whether to clean up small objects
        min_size: Minimum object size to keep
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return

    # Find all subdirectories
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"No subdirectories found in {input_dir}")
        return

    print(f"Found {len(subdirs)} subdirectories to process")
    print(f"Channel threshold configs: {channel_thresholds}")
    print(f"Remove small objects: {remove_small_objects} (min size: {min_size})")
    print()

    total_masks = 0

    # Process each subdirectory
    for subdir in sorted(subdirs):
        print(f"Processing: {subdir.name}")

        # Find all channel NRRD files (not masks)
        channel_files = sorted([f for f in subdir.glob("channel_*.nrrd")
                               if not f.name.endswith("_mask.nrrd")])

        if not channel_files:
            print(f"  No channel files found, skipping")
            continue

        # Process each channel
        for channel_file in channel_files:
            # Extract channel number from filename
            channel_num = int(channel_file.stem.split('_')[1])

            # Skip if no threshold defined for this channel
            if channel_num not in channel_thresholds:
                print(f"  Skipping {channel_file.name} (no threshold defined)")
                continue

            print(f"  Processing {channel_file.name}")

            try:
                # Read NRRD file
                channel_data, header = nrrd.read(str(channel_file))
                print(f"    Shape: {channel_data.shape}, dtype: {channel_data.dtype}")
                print(f"    Intensity range: [{np.min(channel_data)}, {np.max(channel_data)}]")

                # Get threshold config for this channel
                threshold_config = channel_thresholds[channel_num]
                
                # Determine if it's a fixed value or dynamic method
                if isinstance(threshold_config, (int, float)):
                    # Fixed threshold
                    threshold = float(threshold_config)
                    print(f"    Using fixed threshold: {threshold}")
                elif isinstance(threshold_config, dict):
                    # Dynamic threshold
                    method = threshold_config.get('method', 'otsu')
                    print(f"    Calculating dynamic threshold using method: {method}")
                    threshold = calculate_dynamic_threshold(channel_data, method=method, **threshold_config)
                    print(f"    Calculated threshold: {threshold:.2f}")
                else:
                    raise ValueError(f"Invalid threshold config for channel {channel_num}: {threshold_config}")

                # Create mask
                mask = create_mask_with_threshold(
                    channel_data,
                    threshold=threshold,
                    remove_small_objects=remove_small_objects,
                    min_size=min_size
                )

                # Create output filename
                mask_filename = channel_file.stem + "_mask.nrrd"
                mask_filepath = subdir / mask_filename

                # Prepare header for mask
                mask_header = header.copy()
                mask_header['type'] = 'uint8'

                # Write mask
                nrrd.write(str(mask_filepath), mask, mask_header)
                print(f"    Saved mask -> {mask_filename}")

                # Calculate statistics
                foreground_voxels = np.sum(mask > 0)
                total_voxels = mask.size
                percentage = (foreground_voxels / total_voxels) * 100
                print(f"    Foreground: {foreground_voxels}/{total_voxels} voxels ({percentage:.2f}%)")

                total_masks += 1

            except Exception as e:
                print(f"    Error processing {channel_file.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

            print()

    print(f"\n✓ Mask creation complete! Created {total_masks} masks")


if __name__ == "__main__":
    # Configuration
    input_directory = r"D:\NewData\nii.gz_files"  # Parent directory containing all subdirectories

    # OPTION 1: Fixed thresholds (same for all images)
    # channel_thresholds = {
    #     0: 250,   # Channel 0 (First channel) threshold
    #     1: 280,   # Channel 1 (BLUE) - captures bright blue structures
    #     2: 400    # Channel 2 (RED) threshold
    # }

    # OPTION 2: Dynamic thresholds (calculated per image)
    # Channel structure:
    # - Channel 0 (index 0): DAPI nucleus - clear foreground/background, Otsu works well
    # - Channel 1 (index 1): F-actin filaments - fine structures, may need percentile
    # - Channel 2 (index 2): Mitochondria (TOM20) - complex structures, may need careful tuning
    
    # Methods available:
    # - 'otsu': Automatic threshold using Otsu's method (best for clear foreground/background)
    # - 'percentile': Use Nth percentile (e.g., 95th = top 5% brightest)
    # - 'mean_std': mean + (multiplier * std_dev)
    
    channel_thresholds = {
        0: {'method': 'otsu'},  # Channel 0 (DAPI nucleus): Otsu - clear separation
        1: {'method': 'otsu'},  # Channel 1 (F-actin): Start with Otsu, try percentile if needed
        2: {'method': 'percentile', 'percentile': 90}  # Channel 2 (Mitochondria): Percentile for complex structures
    }
    
    # Alternative configurations to try:
    # For F-actin (Channel 1) if Otsu misses fine filaments:
    #   1: {'method': 'percentile', 'percentile': 85}  # Lower percentile captures more dim filaments
    
    # For Mitochondria (Channel 2) if too much noise:
    #   2: {'method': 'percentile', 'percentile': 95}  # Higher percentile = more selective
    # Or if missing structures:
    #   2: {'method': 'mean_std', 'mean_std_multiplier': 1.5}  # Lower threshold

    # Adjust these parameters as needed
    cleanup_small_objects = True
    minimum_object_size = 50  # Reduced from 100 to keep more small structures

    print("=" * 70)
    print("DYNAMIC MASK CREATION WITH ADAPTIVE THRESHOLDS")
    print("=" * 70)
    print("\nThreshold methods:")
    print("  - 'otsu': Automatic optimal threshold (best for most cases)")
    print("  - 'percentile': Use Nth percentile (e.g., 95 = top 5% brightest)")
    print("  - 'mean_std': mean + (multiplier * std_dev)")
    print("  - Fixed value: Just use a number (e.g., 250)")
    print("\nCurrent settings will overwrite existing masks!")
    print("=" * 70)
    print()

    # Run the mask creation
    process_masks_with_custom_thresholds(
        input_directory,
        channel_thresholds=channel_thresholds,
        remove_small_objects=cleanup_small_objects,
        min_size=minimum_object_size
    )

    print("\n" + "=" * 70)
    print("TIPS FOR DYNAMIC THRESHOLDING:")
    print("  - Otsu method: Best for images with clear foreground/background separation")
    print("  - Percentile: Higher percentile (e.g., 98) = more selective (fewer voxels)")
    print("               Lower percentile (e.g., 90) = less selective (more voxels)")
    print("  - Mean+Std: Increase multiplier (e.g., 2.5) = more selective")
    print("             Decrease multiplier (e.g., 1.5) = less selective")
    print("\n  If masks have too much noise:")
    print("    - For percentile: INCREASE percentile value (e.g., 95 -> 98)")
    print("    - For mean_std: INCREASE multiplier (e.g., 2.0 -> 2.5)")
    print("\n  If masks miss structures:")
    print("    - For percentile: DECREASE percentile value (e.g., 95 -> 90)")
    print("    - For mean_std: DECREASE multiplier (e.g., 2.0 -> 1.5)")
    print("=" * 70)
