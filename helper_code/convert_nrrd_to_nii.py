import os
import re
import numpy as np
import nrrd
import nibabel as nib
from pathlib import Path


def convert_nrrd_to_nii(input_folder, output_folder=None, recursive=False):
    """
    Convert all NRRD files from a folder to NII.GZ format, preserving all features and 3D structure.
    
    Args:
        input_folder: Directory containing NRRD files to convert
        output_folder: Output directory for NII.GZ files (default: input_folder/nii_output)
        recursive: If True, search for NRRD files recursively in subdirectories
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Directory {input_folder} does not exist")
        return
    
    # Set up output directory
    if output_folder is None:
        output_folder = input_path / "nii_output"
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_folder}")
    
    # Find all NRRD files
    if recursive:
        nrrd_files = list(input_path.rglob("*.nrrd"))
    else:
        nrrd_files = list(input_path.glob("*.nrrd"))
    
    if not nrrd_files:
        print(f"No NRRD files found in {input_folder}")
        return
    
    print(f"Found {len(nrrd_files)} NRRD file(s)")
    print()
    
    converted_count = 0
    error_count = 0
    
    # Process each file
    for nrrd_file in sorted(nrrd_files):
        print(f"Processing: {nrrd_file.name}")
        
        try:
            # Read NRRD file
            data, header = nrrd.read(str(nrrd_file))
            
            print(f"  Shape: {data.shape}, dtype: {data.dtype}")
            print(f"  Data range: [{np.min(data)}, {np.max(data)}]")
            
            # Ensure data is 3D (or handle 2D/4D appropriately)
            if data.ndim < 3:
                print(f"  Warning: Data has {data.ndim} dimensions. Expected 3D. Skipping.")
                error_count += 1
                continue
            elif data.ndim > 3:
                print(f"  Warning: Data has {data.ndim} dimensions. Taking first volume.")
                data = data[0]  # Take first volume if 4D+
            
            # Extract spacing information from NRRD header
            spacing = None
            space_directions = None
            space_origin = None
            
            # Try to get spacing from 'space directions'
            if 'space directions' in header:
                space_directions = header['space directions']
                # Calculate spacing as the magnitude of each direction vector
                if isinstance(space_directions, np.ndarray):
                    spacing = np.sqrt(np.sum(space_directions**2, axis=1))
                elif isinstance(space_directions, list):
                    spacing = [np.sqrt(sum(d**2 for d in dir_vec)) for dir_vec in space_directions]
                    spacing = np.array(spacing)
            
            # Fallback to 'spacings' if available
            if spacing is None and 'spacings' in header:
                spacing = np.array(header['spacings'])
            
            # Default spacing if not found
            if spacing is None:
                print("  Warning: No spacing information found. Using default spacing [1.0, 1.0, 1.0]")
                spacing = np.array([1.0, 1.0, 1.0])
            
            # Ensure spacing is 3D
            if len(spacing) != 3:
                if len(spacing) > 3:
                    spacing = spacing[:3]
                else:
                    spacing = np.pad(spacing, (0, 3 - len(spacing)), constant_values=1.0)
            
            # Get space origin if available
            if 'space origin' in header:
                space_origin = np.array(header['space origin'])
                if len(space_origin) > 3:
                    space_origin = space_origin[:3]
            
            # Create affine matrix from spacing and origin
            # NRRD uses LPS (left-posterior-superior) space by default
            # NIfTI uses RAS (right-anterior-superior) space
            # We'll preserve the spacing but may need to adjust orientation
            
            affine = np.eye(4, dtype=np.float64)
            
            # Set spacing on diagonal
            affine[0, 0] = spacing[0] if len(spacing) > 0 else 1.0
            affine[1, 1] = spacing[1] if len(spacing) > 1 else 1.0
            affine[2, 2] = spacing[2] if len(spacing) > 2 else 1.0
            
            # Set origin if available
            if space_origin is not None:
                affine[:3, 3] = space_origin
            
            # If we have full space directions, use them for better orientation preservation
            if space_directions is not None and isinstance(space_directions, np.ndarray):
                if space_directions.shape == (3, 3):
                    # Use the direction vectors directly
                    affine[:3, :3] = space_directions
            
            # Check if this is a label file (for nnUNet compatibility)
            # Labels should be integer type with consecutive values starting from 0
            # Check folder name or filename for "label" or "mask"
            file_path_str = str(nrrd_file).lower()
            is_label_file = (
                "label" in file_path_str or 
                "mask" in file_path_str or
                "labelsTr" in file_path_str or
                "labelsTs" in file_path_str
            )
            
            # Ensure data is contiguous and in correct dtype
            if is_label_file:
                # For label files: ensure integer type and consecutive labels
                # Convert to uint8 for nnUNet compatibility
                data_int = data.astype(np.int32)
                unique_labels = np.unique(data_int)
                
                # Check if labels are consecutive starting from 0
                if len(unique_labels) > 0:
                    min_label = int(np.min(unique_labels))
                    max_label = int(np.max(unique_labels))
                    
                    # If labels don't start from 0 or aren't consecutive, remap them
                    if min_label != 0 or len(unique_labels) != (max_label - min_label + 1):
                        print(f"  Warning: Labels not consecutive starting from 0. Remapping...")
                        print(f"    Original labels: {unique_labels[:10]}..." if len(unique_labels) > 10 else f"    Original labels: {unique_labels}")
                        
                        # Create mapping: old_label -> new_label (consecutive from 0)
                        label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
                        data_remapped = np.zeros_like(data_int)
                        for old_label, new_label in label_map.items():
                            data_remapped[data_int == old_label] = new_label
                        data_int = data_remapped
                        
                        unique_labels = np.unique(data_int)
                        print(f"    Remapped labels: {unique_labels[:10]}..." if len(unique_labels) > 10 else f"    Remapped labels: {unique_labels}")
                    
                    # Ensure max label fits in uint8 (0-255)
                    if max_label > 255:
                        print(f"  Warning: Max label {max_label} exceeds uint8 range. Clipping to 255.")
                        data_int = np.clip(data_int, 0, 255)
                
                data = np.ascontiguousarray(data_int.astype(np.uint8))
                print(f"  Label file: {len(unique_labels)} unique labels, dtype: {data.dtype}")
            else:
                # For image files: preserve original dtype if it's a standard medical imaging dtype
                if data.dtype not in [np.uint8, np.uint16, np.int16, np.float32, np.float64]:
                    # Convert to float32 if dtype is unusual
                    print(f"  Converting dtype from {data.dtype} to float32")
                    data = data.astype(np.float32)
                else:
                    data = np.ascontiguousarray(data)
            
            # Create output filename
            # nnUNet naming convention:
            # - Images: case_0000_0000.nii.gz (with _0000 channel suffix)
            # - Labels: case_0000.nii.gz (NO channel suffix)
            base_name = nrrd_file.stem
            
            # Try to extract case number from filename patterns like "Sample_001_0000" or "case_0000"
            case_match = re.search(r'(?:Sample_|case_)(\d+)', base_name, re.IGNORECASE)
            if case_match:
                case_num = int(case_match.group(1))
                # Format as case_0000, case_0001, etc. (4-digit zero-padded)
                if is_label_file:
                    # Labels: case_0000.nii.gz (NO _0000 suffix)
                    output_base_name = f"case_{case_num:04d}"
                else:
                    # Images: case_0000_0000.nii.gz (WITH _0000 suffix for channel/modality)
                    output_base_name = f"case_{case_num:04d}_0000"
            else:
                # If no pattern match, try to preserve original name but clean it up
                if is_label_file:
                    # Remove _0000 suffix for labels
                    output_base_name = base_name.replace("Sample_", "case_").replace("_0000", "")
                else:
                    # Keep _0000 suffix for images
                    output_base_name = base_name.replace("Sample_", "case_")
            
            output_file = output_folder / f"{output_base_name}.nii.gz"
            
            # If recursive, preserve subdirectory structure
            if recursive and nrrd_file.parent != input_path:
                relative_path = nrrd_file.relative_to(input_path)
                output_subdir = output_folder / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_file = output_subdir / f"{base_name}.nii.gz"
            
            # Create NIfTI image
            nii_img = nib.Nifti1Image(data, affine)
            
            # Preserve additional metadata if possible
            # Note: NIfTI has limited header space compared to NRRD
            if 'type' in header:
                # Try to preserve data type info in description
                nii_img.header['descrip'] = f"Converted from NRRD, original type: {header['type']}"
            
            # Save NIfTI file
            nib.save(nii_img, str(output_file))
            
            print(f"  ✓ Saved -> {output_file.name}")
            print(f"    Spacing: {spacing}")
            converted_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {nrrd_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
        
        print()
    
    # Print summary
    print("=" * 70)
    print(f"Conversion complete!")
    print(f"  Successfully converted: {converted_count} file(s)")
    if error_count > 0:
        print(f"  Errors: {error_count} file(s)")
    print(f"  Output directory: {output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    # ============================================================================
    # SET YOUR PATHS HERE (or use command-line arguments)
    # ============================================================================
    # For labels:
    DEFAULT_INPUT_FOLDER_LABELS = "/Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/labelsTr"
    # For training images:
    DEFAULT_INPUT_FOLDER_IMAGES = "/Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/imagesTr"
    
    # Change this to switch between labels and images:
    DEFAULT_INPUT_FOLDER = DEFAULT_INPUT_FOLDER_LABELS  # Change to DEFAULT_INPUT_FOLDER_IMAGES for images
    
    parser = argparse.ArgumentParser(
        description="Convert NRRD files to NII.GZ format, preserving all features and 3D structure"
    )
    parser.add_argument(
        "input_folder",
        type=str,
        nargs="?",  # Make it optional
        default=DEFAULT_INPUT_FOLDER,
        help=f"Directory containing NRRD files to convert (default: {DEFAULT_INPUT_FOLDER})"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for NII.GZ files (default: input_folder/nii_output)"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search for NRRD files recursively in subdirectories"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NRRD to NII.GZ Converter")
    print("=" * 70)
    print(f"Input folder: {args.input_folder}")
    print()
    
    convert_nrrd_to_nii(
        input_folder=args.input_folder,
        output_folder=args.output,
        recursive=args.recursive
    )

