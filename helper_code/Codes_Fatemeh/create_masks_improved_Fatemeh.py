import os
import numpy as np
import nrrd
from pathlib import Path
from scipy import ndimage

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
    Process all NRRD files and create masks with custom thresholds per channel.

    Args:
        input_dir: Directory containing subdirectories with NRRD files
        channel_thresholds: Dictionary mapping channel index to threshold value
                          e.g., {0: 250, 1: 300, 2: 400}
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
    print(f"Channel thresholds: {channel_thresholds}")
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

                # Get threshold for this channel
                threshold = channel_thresholds[channel_num]

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
    input_directory = "nrrd_output"

    # Set custom thresholds for each channel
    # Based on the analysis:
    # - Channel 0: First channel - Most data 140-242, threshold 250
    # - Channel 1 (BLUE): Most data 140-536, but bright spots up to 4095
    #                     Use threshold ~280 to capture blue fluorescence
    # - Channel 2 (RED): Most data 127-524, use threshold around 400

    channel_thresholds = {
        0: 250,   # Channel 0 (First channel) threshold
        1: 280,   # Channel 1 (BLUE) - captures bright blue structures
        2: 400    # Channel 2 (RED) threshold
    }

    # Adjust these parameters as needed
    cleanup_small_objects = True
    minimum_object_size = 50  # Reduced from 100 to keep more small structures

    print("=" * 70)
    print("IMPROVED MASK CREATION WITH CUSTOM THRESHOLDS")
    print("=" * 70)
    print("\nRecommended threshold adjustments:")
    print("  - Increase threshold: Capture only brightest structures")
    print("  - Decrease threshold: Capture more dimmer structures")
    print("  - Channel 1 (BLUE): Try values between 250-350")
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
    print("  - Channel 1 (BLUE): try thresholds 250, 280, 300, 350")
    print("  - Channel 2 (RED): try thresholds 350, 380, 400, 450")
    print("=" * 70)
