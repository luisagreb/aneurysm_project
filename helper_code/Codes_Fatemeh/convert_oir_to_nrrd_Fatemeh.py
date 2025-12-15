import os
import numpy as np
import nrrd
from bioio import BioImage
from pathlib import Path

def convert_oir_to_nrrd(input_dir, output_dir=None):
    """
    Convert .oir files to NRRD format, preserving channels and z-stacks.

    Args:
        input_dir: Directory containing .oir files
        output_dir: Output directory for NRRD files (default: input_dir/nrrd_output)
    """
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(input_dir, "nrrd_output")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Find all .oir files
    oir_files = list(Path(input_dir).glob("*.oir"))

    if not oir_files:
        print(f"No .oir files found in {input_dir}")
        return

    print(f"Found {len(oir_files)} .oir file(s)")

    # Process each file
    for oir_file in oir_files:
        print(f"\nProcessing: {oir_file.name}")

        try:
            # Read the .oir file
            img = BioImage(oir_file)

            # Get image data
            data = img.data  # Shape: (T, C, Z, Y, X)

            print(f"  Image shape: {data.shape}")
            print(f"  Dimension order: {img.dims}")
            print(f"  Number of channels: {img.dims.C}")
            print(f"  Z-stack size: {img.dims.Z}")

            # Get physical pixel sizes for NRRD spacing
            physical_pixel_sizes = img.physical_pixel_sizes
            spacing = [
                physical_pixel_sizes.Z if physical_pixel_sizes.Z else 1.0,
                physical_pixel_sizes.Y if physical_pixel_sizes.Y else 1.0,
                physical_pixel_sizes.X if physical_pixel_sizes.X else 1.0
            ]

            # Remove time dimension (take first timepoint)
            if img.dims.T > 1:
                print(f"  Multiple timepoints detected ({img.dims.T}), using first timepoint")
            data = data[0]  # Remove T dimension, now shape is (C, Z, Y, X)

            # Create a folder for this .oir file
            base_name = oir_file.stem
            file_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(file_output_dir, exist_ok=True)

            # Process each channel separately
            for channel_idx in range(img.dims.C):
                # Extract channel data (shape will be ZYX)
                channel_data = data[channel_idx]

                # Create output filename in the file's own folder
                output_file = os.path.join(
                    file_output_dir,
                    f"channel_{channel_idx:02d}.nrrd"
                )

                # Prepare NRRD header
                header = {
                    'space': 'left-posterior-superior',
                    'space directions': np.array([
                        [spacing[0], 0, 0],
                        [0, spacing[1], 0],
                        [0, 0, spacing[2]]
                    ]),
                    'kinds': ['domain', 'domain', 'domain'],
                    'encoding': 'gzip'
                }

                # Write NRRD file
                nrrd.write(output_file, channel_data, header)
                print(f"  Saved channel {channel_idx} -> {base_name}/{os.path.basename(output_file)}")
                print(f"    Shape: {channel_data.shape} (Z, Y, X)")
                print(f"    Spacing: {spacing} (Z, Y, X)")

        except Exception as e:
            print(f"  Error processing {oir_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✓ Conversion complete! Files saved to: {output_dir}")


if __name__ == "__main__":
    # Use current directory
    current_dir = os.getcwd()
    print(f"Converting .oir files from: {current_dir}\n")
    convert_oir_to_nrrd(current_dir)
