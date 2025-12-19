import os
import numpy as np
import nrrd
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
try:
    from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
    from skimage import restoration
    from skimage.filters import frangi
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Advanced features will not be available.")
    print("Install with: pip install scikit-image")

def preprocess_actin_channel(channel_data, rolling_ball_radius=15, frangi_sigma_range=(2, 7), frangi_beta=2.0):
    """
    Advanced preprocessing for actin channel: Pre-filtering + Rolling Ball + Frangi Filter
    
    Args:
        channel_data: 3D numpy array (Z, Y, X)
        rolling_ball_radius: Radius for rolling ball background subtraction (default: 15)
        frangi_sigma_range: Range of sigma values for Frangi filter (default: (2, 7))
        frangi_beta: Beta parameter for Frangi filter - higher values penalize square shapes (default: 2.0)
    
    Returns:
        Preprocessed 3D array with enhanced fibers
    """
    if not HAS_SKIMAGE:
        print("    Warning: scikit-image required for advanced preprocessing. Using original data.")
        return channel_data
    
    print(f"    Applying advanced preprocessing: Median filter + Gaussian blur + Rolling Ball (radius={rolling_ball_radius}) + Frangi filter (beta={frangi_beta})")
    
    # Process each Z-slice separately (Frangi filter works on 2D)
    processed_slices = []
    
    for z_idx in range(channel_data.shape[0]):
        slice_2d = channel_data[z_idx, :, :]
        
        # Convert to float for processing
        slice_float = slice_2d.astype(np.float64)
        
        # Step 0a: Kill horizontal line noise with median filter (3x1)
        # This specifically targets lines that are 1-pixel high
        slice_float = median_filter(slice_float, size=(3, 1))
        
        # Step 0b: Soften tiling square edges with Gaussian blur
        # Makes the "hard" edges of squares invisible to Frangi
        slice_float = gaussian_filter(slice_float, sigma=1.2)
        
        # Step 1: Rolling Ball Background Subtraction
        # Rolling ball background subtraction
        # Using morphological opening as approximation (rolling ball is similar)
        from scipy.ndimage import uniform_filter
        background = uniform_filter(slice_float, size=rolling_ball_radius * 2 + 1)
        slice_subtracted = slice_float - background
        slice_subtracted = np.clip(slice_subtracted, 0, None)  # Remove negative values
        
        # Step 2: Frangi Filter (enhances tubular structures like actin fibers)
        # Normalize to 0-1 range for Frangi filter
        slice_normalized = (slice_subtracted - np.min(slice_subtracted)) / (np.max(slice_subtracted) - np.min(slice_subtracted) + 1e-10)
        
        # Apply Frangi filter
        frangi_response = frangi(slice_normalized, 
                                 sigmas=frangi_sigma_range,
                                 beta=frangi_beta,
                                 gamma=1.0,
                                 black_ridges=False)
        
        # Convert back to original intensity range
        frangi_scaled = frangi_response * (np.max(slice_subtracted) - np.min(slice_subtracted)) + np.min(slice_subtracted)
        frangi_scaled = np.clip(frangi_scaled, 0, np.max(slice_float))
        
        processed_slices.append(frangi_scaled)
    
    # Stack slices back into 3D
    processed_3d = np.stack(processed_slices, axis=0)
    
    print(f"    Preprocessing complete: enhanced fiber structures")
    return processed_3d.astype(channel_data.dtype)


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


def apply_local_thresholding(channel_data, method='sauvola', window_size=15, k=0.2):
    """
    Apply local adaptive thresholding (Niblack or Sauvola) to 3D data.
    
    Args:
        channel_data: 3D numpy array (Z, Y, X)
        method: 'niblack' or 'sauvola' (default: 'sauvola')
        window_size: Size of local window (default: 15)
        k: Parameter for threshold calculation (default: 0.2)
    
    Returns:
        Binary mask from local thresholding
    """
    if not HAS_SKIMAGE:
        print("    Warning: scikit-image required for local thresholding.")
        return None
    
    print(f"    Applying local {method} thresholding (window={window_size}, k={k})")
    
    # Process each Z-slice separately
    mask_slices = []
    
    for z_idx in range(channel_data.shape[0]):
        slice_2d = channel_data[z_idx, :, :].astype(np.float64)
        
        # Normalize to 0-1 range
        slice_min = np.min(slice_2d)
        slice_max = np.max(slice_2d)
        if slice_max > slice_min:
            slice_normalized = (slice_2d - slice_min) / (slice_max - slice_min)
        else:
            slice_normalized = slice_2d
        
        # Apply local thresholding
        if method == 'sauvola':
            threshold_map = threshold_sauvola(slice_normalized, window_size=window_size, k=k)
        elif method == 'niblack':
            threshold_map = threshold_niblack(slice_normalized, window_size=window_size, k=k)
        else:
            raise ValueError(f"Unknown local thresholding method: {method}")
        
        # Create binary mask
        slice_mask = slice_normalized > threshold_map
        mask_slices.append(slice_mask)
    
    # Stack slices back into 3D
    mask_3d = np.stack(mask_slices, axis=0)
    
    return mask_3d


def create_mask_with_threshold(channel_data, threshold, remove_small_objects=True, min_size=50, channel_num=None, use_local_threshold=False):
    """
    Create a binary mask from channel data using a specific threshold.

    Args:
        channel_data: 3D numpy array (Z, Y, X)
        threshold: threshold value (used if use_local_threshold=False)
        remove_small_objects: whether to remove small objects from mask
        min_size: minimum size of objects to keep (in voxels)
        channel_num: channel number (for channel-specific cleanup, e.g., 1 for actin)
        use_local_threshold: if True, use local adaptive thresholding instead of global

    Returns:
        Binary mask as uint8 array
    """
    # Use local thresholding if requested (for channel 1 with Frangi preprocessing)
    if use_local_threshold:
        # Lower k value = more liberal thresholding (captures more filaments)
        # k=0.01 is very sensitive, captures dim regions between bright dashes
        # Larger window (51) provides better background estimation
        mask = apply_local_thresholding(channel_data, method='sauvola', window_size=51, k=0.01)
        if mask is None:
            # Fallback to global threshold
            print(f"    Falling back to global threshold: {threshold}")
            mask = channel_data > threshold
        else:
            print(f"    Using local Sauvola thresholding (window=51, k=0.01, very sensitive)")
    else:
        print(f"    Using threshold: {threshold}")
        # Create binary mask
        mask = channel_data > threshold

    # Remove small objects if requested
    if remove_small_objects:
        # Channel-specific cleanup parameters
        if channel_num == 1:
            # --- Step A: Clear the Borders (especially bottom for scan lines) ---
            # Scan lines usually sit at the very edge. Zero out bottom 20 pixels.
            mask[:, :10, :] = 0   # Top border
            mask[:, -20:, :] = 0  # Bottom border (extra clearing for scan lines)
            mask[:, :, :10] = 0   # Left border
            mask[:, :, -10:] = 0  # Right border
            print(f"    Zeroed out borders (20px bottom, 10px other sides) to remove edge artifacts")
            
            # --- Step B: Shape-Based Filtering (Remove Horizontal Lines and Squares) ---
            labeled_mask_temp, num_features_temp = ndimage.label(mask)
            removed_shapes = {'horizontal_lines': 0, 'small_squares': 0}
            
            for i in range(1, num_features_temp + 1):
                component_mask = (labeled_mask_temp == i)
                coords = np.argwhere(component_mask)
                
                if len(coords) == 0:
                    continue
                
                # Calculate bounding box dimensions (Y is axis 1, X is axis 2 in 3D)
                y_min, x_min = coords[:, 1].min(), coords[:, 2].min()
                y_max, x_max = coords[:, 1].max(), coords[:, 2].max()
                
                width = x_max - x_min + 1
                height = y_max - y_min + 1
                
                # 1. DELETE Horizontal Scan Lines:
                # If the object is very wide (>40px) but very short (<4px)
                if width > 40 and height < 4:
                    mask[component_mask] = 0
                    removed_shapes['horizontal_lines'] += 1
                    continue
                
                # 2. DELETE Tiling Squares:
                # If the object is roughly a square (ratio close to 1:1) and small
                if width < 15 and height < 15 and abs(width - height) < 5:
                    mask[component_mask] = 0
                    removed_shapes['small_squares'] += 1
                    continue
            
            print(f"    Shape-based filtering: removed {removed_shapes['horizontal_lines']} horizontal lines, {removed_shapes['small_squares']} small squares")
            
            # --- Step C: Connect the "Dashed" Filaments ---
            # Use slightly larger closing to bridge gaps in real filaments
            mask = ndimage.binary_dilation(mask, structure=np.ones((2, 2, 2)))
            mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)))
            print(f"    Applied actin-specific cleanup: dilation + larger closing (connects dashed filaments)")
        else:
            # Standard cleanup for other channels
            mask = ndimage.binary_opening(mask, structure=np.ones((2, 2, 2)))

        # Label connected components
        labeled_mask, num_features = ndimage.label(mask)

        # Remove small connected components with channel-specific min_size
        removed_count = 0
        # Use larger min_size for channel 1 to filter out small artifacts
        effective_min_size = min_size if channel_num == 1 else min_size  # 250 voxels for channel 1
        
        # For channel 1, also remove objects far from the main structure
        if channel_num == 1 and num_features > 1:
            # Find the largest component (main actin structure)
            component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
            largest_component = np.argmax(component_sizes) + 1
            
            # Calculate distance from each component to the largest component
            largest_mask = (labeled_mask == largest_component)
            # Get coordinates of largest component
            largest_coords = np.argwhere(largest_mask)
            
            if len(largest_coords) > 0:
                # Calculate centroid of largest component
                largest_centroid = np.mean(largest_coords, axis=0)
                
                # Remove components that are far from main structure
                for i in range(1, num_features + 1):
                    if i == largest_component:
                        continue
                    component_mask = (labeled_mask == i)
                    component_coords = np.argwhere(component_mask)
                    if len(component_coords) > 0:
                        component_centroid = np.mean(component_coords, axis=0)
                        distance = np.linalg.norm(component_centroid - largest_centroid)
                        # Remove if far away (distance threshold based on image size)
                        # Increased threshold to preserve more actin structures and extensions
                        max_distance = np.max(channel_data.shape) * 0.5  # 50% of max dimension
                        if distance > max_distance:
                            mask[component_mask] = 0
                            removed_count += 1
                            continue
                    
                    # Also remove if component is too small
                    component_size = np.sum(component_mask)
                    if component_size < effective_min_size:
                        mask[component_mask] = 0
                        removed_count += 1
            else:
                # Fallback: just remove small components
                for i in range(1, num_features + 1):
                    component_size = np.sum(labeled_mask == i)
                    if component_size < effective_min_size:
                        mask[labeled_mask == i] = 0
                        removed_count += 1
        else:
            # Standard removal for other channels or if only one component
            for i in range(1, num_features + 1):
                component_size = np.sum(labeled_mask == i)
                if component_size < effective_min_size:
                    mask[labeled_mask == i] = 0
                    removed_count += 1

        if channel_num == 1:
            print(f"    Cleaned up small/distant objects (min size: {effective_min_size} voxels for channel 1, removed {removed_count} objects)")
        else:
            print(f"    Cleaned up small objects (min size: {min_size} voxels, removed {removed_count} objects)")

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

    # Process the first 10 folders
    subdirs = sorted(subdirs)[:10]
    
    print(f"Found {len(subdirs)} subdirectory(ies) to process (processing first 10 folders)")
    print(f"Channel threshold configs: {channel_thresholds}")
    print(f"Remove small objects: {remove_small_objects} (min size: {min_size})")
    print()

    total_masks = 0

    # Process the first 10 subdirectories
    for subdir in subdirs:
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

                # Advanced preprocessing for channel 1 (actin)
                use_advanced_preprocessing = (channel_num == 1)
                processed_data = channel_data
                
                if use_advanced_preprocessing:
                    # Apply Rolling Ball + Frangi Filter preprocessing
                    # Sigma range (2, 7): Starting at 2 ignores 1-pixel-high horizontal noise lines
                    # Higher beta (2.0): Heavily penalizes square/blobby shapes, only keeps long thin structures
                    processed_data = preprocess_actin_channel(
                        channel_data,
                        rolling_ball_radius=15,
                        frangi_sigma_range=(2, 7),
                        frangi_beta=2.0
                    )
                    print(f"    Preprocessed intensity range: [{np.min(processed_data)}, {np.max(processed_data)}]")

                # Get threshold config for this channel
                threshold_config = channel_thresholds[channel_num]
                
                # Determine if it's a fixed value or dynamic method
                use_local_threshold = False
                threshold = None
                
                if isinstance(threshold_config, (int, float)):
                    # Fixed threshold
                    threshold = float(threshold_config)
                    print(f"    Using fixed threshold: {threshold}")
                elif isinstance(threshold_config, dict):
                    method = threshold_config.get('method', 'otsu')
                    
                    # Check if using advanced preprocessing with local thresholding
                    if use_advanced_preprocessing and method in ['niblack', 'sauvola', 'local']:
                        use_local_threshold = True
                        print(f"    Using local adaptive thresholding ({method}) on preprocessed data")
                    elif method == 'percentile' and use_advanced_preprocessing:
                        # Use local thresholding on Frangi-filtered image
                        use_local_threshold = True
                        print(f"    Using local Sauvola thresholding on Frangi-filtered image")
                    else:
                        # Dynamic threshold on preprocessed data
                        print(f"    Calculating dynamic threshold using method: {method}")
                        threshold = calculate_dynamic_threshold(processed_data, method=method, **threshold_config)
                        print(f"    Calculated threshold: {threshold:.2f}")
                else:
                    raise ValueError(f"Invalid threshold config for channel {channel_num}: {threshold_config}")

                # Create mask with channel-specific cleanup
                mask = create_mask_with_threshold(
                    processed_data,  # Use preprocessed data
                    threshold=threshold if threshold is not None else 0,
                    remove_small_objects=remove_small_objects,
                    min_size=min_size,
                    channel_num=channel_num,
                    use_local_threshold=use_local_threshold
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
    input_directory = "/Volumes/LuisaHD/NewData/nrrd_files"

    # OPTION 1: Advanced preprocessing for actin (Rolling Ball + Frangi + Local Thresholding)
    # Channel 1 uses advanced preprocessing: Rolling Ball Background Subtraction + Frangi Filter + Local Adaptive Thresholding
    channel_thresholds = {
        0: 250,   # Channel 0 (First channel) threshold - fixed
        1: {'method': 'local'},  # Channel 1 (actin) - uses Rolling Ball + Frangi + Sauvola local thresholding
        2: 400    # Channel 2 (RED) threshold - fixed
    }
    
    # Advanced preprocessing pipeline for channel 1 (SHAPE-BASED ARTIFACT REMOVAL):
    # 0a. Median Filter (3x1) - kills horizontal line noise
    # 0b. Gaussian Blur (sigma=1.2) - softens tiling square edges (increased from 1.0)
    # 1. Rolling Ball Background Subtraction (radius=15px) - removes diffuse glow
    # 2. Frangi Filter (sigma_range=(2,7), beta=2.0) - HEAVILY penalizes squares, ignores 1-px lines
    # 3. Sauvola Local Adaptive Thresholding (window=51, k=0.01) - very sensitive
    # 4. Border Zeroing (20px bottom, 10px sides) - removes scan lines at edges
    # 5. Shape-Based Filtering - removes horizontal lines (>40px wide, <4px tall) and small squares (<15x15)
    # 6. Morphological: Dilation + Larger Closing - connects dashed filaments
    # 7. Min object size: 250 voxels - real filaments are long, kills small fragments
    
    # Alternative configurations for channel 1:
    # Option A: Advanced preprocessing with local thresholding (RECOMMENDED - current setting)
    #   1: {'method': 'local'}  # Uses Rolling Ball + Frangi + Sauvola
    
    # Option B: Advanced preprocessing with percentile threshold
    #   1: {'method': 'percentile', 'percentile': 85}  # Rolling Ball + Frangi + percentile
    
    # Option C: Otsu method on preprocessed data
    #   1: {'method': 'otsu'}  # Rolling Ball + Frangi + Otsu
    
    # Option D: No preprocessing, just percentile (simpler but less effective)
    #   1: {'method': 'percentile', 'percentile': 85}  # No preprocessing

    # OPTION 2: Dynamic thresholds (calculated per image)
    # Uncomment below to use dynamic thresholding instead
    # Channel structure:
    # - Channel 0 (index 0): DAPI nucleus - clear foreground/background, Otsu works well
    # - Channel 1 (index 1): F-actin filaments - fine structures, may need percentile
    # - Channel 2 (index 2): Mitochondria (TOM20) - complex structures, may need careful tuning
    #
    # Methods available:
    # - 'otsu': Automatic threshold using Otsu's method (best for clear foreground/background)
    # - 'percentile': Use Nth percentile (e.g., 95th = top 5% brightest)
    # - 'mean_std': mean + (multiplier * std_dev)
    #
    # channel_thresholds = {
    #     0: {'method': 'otsu'},  # Channel 0 (DAPI nucleus): Otsu - clear separation
    #     1: {'method': 'otsu'},  # Channel 1 (F-actin): Start with Otsu, try percentile if needed
    #     2: {'method': 'percentile', 'percentile': 90}  # Channel 2 (Mitochondria): Percentile for complex structures
    # }
    #
    # Alternative configurations to try:
    # For F-actin (Channel 1) if Otsu misses fine filaments:
    #   1: {'method': 'percentile', 'percentile': 85}  # Lower percentile captures more dim filaments
    #
    # For Mitochondria (Channel 2) if too much noise:
    #   2: {'method': 'percentile', 'percentile': 95}  # Higher percentile = more selective
    # Or if missing structures:
    #   2: {'method': 'mean_std', 'mean_std_multiplier': 1.5}  # Lower threshold

    # Adjust these parameters as needed
    cleanup_small_objects = True
    minimum_object_size = 250  # Real filaments are long; this kills small square fragments

    print("=" * 70)
    print("IMPROVED MASK CREATION WITH CUSTOM/ADAPTIVE THRESHOLDS")
    print("=" * 70)
    print("\nThreshold options:")
    print("  - Fixed value: Use integer (e.g., 250)")
    print("  - Dynamic 'otsu': Automatic optimal threshold (best for most cases)")
    print("  - Dynamic 'percentile': Use Nth percentile (e.g., 95 = top 5% brightest)")
    print("  - Dynamic 'mean_std': mean + (multiplier * std_dev)")
    print("\nRecommended threshold adjustments:")
    print("  - Increase threshold: Capture only brightest structures")
    print("  - Decrease threshold: Capture more dimmer structures")
    print("  - Channel 1 (BLUE): Try values between 180-350")
    print("  - Channel 2 (RED): Try values between 350-450")
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
    print("TIPS:")
    print("  - If masks have too much noise: INCREASE threshold values")
    print("  - If masks miss structures: DECREASE threshold values")
    print("  - Channel 1 (BLUE): try thresholds 180, 250, 280, 300, 350")
    print("  - Channel 2 (RED): try thresholds 350, 380, 400, 450")
    print("\nFor dynamic thresholding:")
    print("  - Otsu method: Best for images with clear foreground/background separation")
    print("  - Percentile: Higher percentile (e.g., 98) = more selective (fewer voxels)")
    print("               Lower percentile (e.g., 90) = less selective (more voxels)")
    print("  - Mean+Std: Increase multiplier (e.g., 2.5) = more selective")
    print("             Decrease multiplier (e.g., 1.5) = less selective")
    print("=" * 70)
