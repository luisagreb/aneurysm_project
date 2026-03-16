"""
OIR → NRRD Conversion Script

Converts Olympus .oir microscopy files to per-channel NRRD files
compatible with the existing nnU-Net preparation pipeline.

For each .oir file, outputs:
    channel_00.nrrd  →  Nucleus   (channel index 0)
    channel_01.nrrd  →  Actin     (channel index 1)
    channel_02.nrrd  →  Mito      (channel index 2)

Output structure mirrors the existing imagesTs_nrrd format:
    data/processed/new_data_nrrd/
    └── 02Asc-0017 +col Cell 1/
        ├── channel_00.nrrd
        ├── channel_01.nrrd
        └── channel_02.nrrd

Usage (from project root):
    python3 src/preparation/oir_to_nrrd.py

Requirements:
    pip install aicsimageio aicspylibczi nrrd
    (aicsimageio reads .oir via the Olympus plugin)
"""

import os
import re
import numpy as np
import nrrd
from pathlib import Path

try:
    from aicsimageio import AICSImage
except ImportError:
    raise ImportError(
        "aicsimageio is required. Install with:\n"
        "  pip install aicsimageio aicspylibczi"
    )

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
INPUT_DIR  = Path("data/raw/new_data_oir")          # Folder with new .oir files
OUTPUT_DIR = Path("data/processed/new_data_nrrd")    # Output NRRD root

# Channel mapping — verify this matches your microscope setup!
# Open one .oir in ImageJ/FIJI to confirm channel order.
CHANNEL_NAMES = {
    0: "Nucleus",
    1: "Actin",
    2: "Mitochondria",
}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def read_oir(oir_path: Path):
    """
    Read an .oir file and return:
        - data array: shape (C, Z, Y, X)
        - voxel spacing: (Z_um, Y_um, X_um) in micrometres
    """
    img = AICSImage(str(oir_path))
    
    # Get data as (C, Z, Y, X) — squeeze out T and S dims if present
    data = img.get_image_data("CZYX", T=0, S=0)  # shape: (C, Z, Y, X)
    
    # Physical pixel sizes in micrometres
    ps = img.physical_pixel_sizes          # PhysicalPixelSizes(Z, Y, X)
    spacing_z = ps.Z if ps.Z else 1.0
    spacing_y = ps.Y if ps.Y else 1.0
    spacing_x = ps.X if ps.X else 1.0
    
    spacing = (float(spacing_z), float(spacing_y), float(spacing_x))
    
    return data, spacing


def save_channel_nrrd(volume: np.ndarray, spacing_zyx: tuple, output_path: Path):
    """
    Save a single 3D volume (Z, Y, X) as NRRD with physical spacing.
    """
    sz, sy, sx = spacing_zyx
    
    # NRRD header with space directions (diagonal matrix = axis-aligned voxels)
    header = {
        "space": "left-posterior-superior",
        "space directions": np.array([
            [sz, 0,  0 ],
            [0,  sy, 0 ],
            [0,  0,  sx],
        ]),
        "space origin": np.array([0.0, 0.0, 0.0]),
        "kinds": ["domain", "domain", "domain"],
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(output_path), volume.astype(np.float32), header=header)


def make_cell_folder_name(oir_path: Path) -> str:
    """
    Strip .oir extension to get the cell folder name.
    E.g. '02Asc-0017 +col Cell 1.oir' → '02Asc-0017 +col Cell 1'
    """
    return oir_path.stem


# ─────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────

def convert_all(input_dir: Path, output_dir: Path):
    oir_files = sorted(input_dir.rglob("*.oir"))
    
    if not oir_files:
        print(f"No .oir files found in {input_dir}")
        return
    
    print("=" * 70)
    print(f"OIR → NRRD Conversion")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(oir_files)} .oir files")
    print("=" * 70)
    
    success = 0
    errors  = 0
    
    for idx, oir_path in enumerate(oir_files, 1):
        # Build output folder preserving the subject/condition subfolder structure
        # e.g. new_data_oir/02Asc-0017/+Col/Cell1.oir
        #  →  new_data_nrrd/02Asc-0017/+Col/Cell1/channel_00.nrrd
        rel = oir_path.relative_to(input_dir)          # e.g. 02Asc-0017/+Col/Cell1.oir
        cell_folder = output_dir / rel.parent / rel.stem  # drop .oir, keep hierarchy
        
        print(f"\n[{idx}/{len(oir_files)}] {rel}")
        
        try:
            data, spacing = read_oir(oir_path)
            n_channels = data.shape[0]
            
            print(f"  Shape (C,Z,Y,X): {data.shape}")
            print(f"  Spacing (Z,Y,X) µm: {spacing}")
            print(f"  Channels: {n_channels}")
            
            for ch_idx in range(min(n_channels, 3)):
                volume   = data[ch_idx]             # (Z, Y, X)
                out_file = cell_folder / f"channel_{ch_idx:02d}.nrrd"
                save_channel_nrrd(volume, spacing, out_file)
                label    = CHANNEL_NAMES.get(ch_idx, f"ch{ch_idx}")
                print(f"  ✓ Saved {label}: {out_file.name}  "
                      f"(shape={volume.shape}, dtype={volume.dtype})")
            
            success += 1
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback; traceback.print_exc()
            errors += 1
    
    print("\n" + "=" * 70)
    print(f"DONE — {success} succeeded, {errors} failed")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 70)
    print("\nNext step:")
    print("  python3 src/preparation/prepare_test_data_for_nnunet.py \\")
    print(f"    --source {output_dir} \\")
    print(f"    --output data/nnUNet/nnUNet_raw/Dataset001_Actin/imagesTs_new")


if __name__ == "__main__":
    convert_all(INPUT_DIR, OUTPUT_DIR)
