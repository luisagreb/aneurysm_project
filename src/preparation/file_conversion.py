from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.readers import BioformatsReader

# ─────────────────────────────────────────────────────────
# Paths  (relative to project root)
# ─────────────────────────────────────────────────────────
INPUT_ROOT  = Path("data/raw/new_data_oir")           # New BAV .oir files
OUTPUT_ROOT = Path("data/processed/new_data_nifti")   # Per-channel NIfTI output

# Channel mapping — verified from original data acquisition
CHANNEL_LABELS = {0: "Nucleus", 1: "Actin", 2: "Mitochondria"}

# ─────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────

def convert_oir_to_nifti_channels(input_file: Path, output_dir: Path) -> dict:
    """
    Convert one .oir file into per-channel .nii.gz files, preserving all
    physical pixel metadata (spacing in X, Y, Z) from the original file.

    Channel mapping:
      channel_00.nii.gz  →  Nucleus
      channel_01.nii.gz  →  Actin
      channel_02.nii.gz  →  Mitochondria

    Args:
        input_file:  Path to the .oir file
        output_dir:  Folder where per-channel NIfTI files are saved

    Returns:
        dict with conversion metadata (spacing, shape, success)
    """
    print(f"  Converting: {input_file.name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Read with BioFormats (preserves all Olympus metadata) ──
    img = AICSImage(str(input_file), reader=BioformatsReader)
    data = img.get_image_data("CZYX")       # shape: (C, Z, Y, X)
    data = data.astype(np.float32)

    # ── Physical voxel sizes from .oir metadata (in µm) ──
    ps = img.physical_pixel_sizes
    if ps and ps.X and ps.Y:
        zsz = float(ps.Z) if ps.Z else float(ps.Y)
        sx  = float(ps.X)
        sy  = float(ps.Y)
        sz  = zsz
    else:
        sx = sy = sz = 1.0
        print("  ⚠  No physical pixel sizes found — using 1 µm isotropic")

    print(f"  Shape (C,Z,Y,X): {data.shape}")
    print(f"  Spacing (µm)  X={sx:.4f}  Y={sy:.4f}  Z={sz:.4f}")

    # ── Transpose from (Z,Y,X) → (X,Y,Z) to match existing pipeline ──
    # prepare_test_data_for_nnunet.py transposes volumes and uses unit spacing.
    # The nnU-Net models were trained on data in this (X,Y,Z) + isotropic format.
    # affine = identity (pixdim=1,1,1) to match existing imagesTs files.
    affine = np.eye(4, dtype=np.float32)

    # ── Save per-channel NIfTI ──
    n_channels = min(data.shape[0], 3)   # cap at 3
    for c in range(n_channels):
        label   = CHANNEL_LABELS.get(c, f"ch{c}")
        vol_zyx = data[c]                          # shape (Z, Y, X)
        vol_xyz = np.transpose(vol_zyx, (2, 1, 0)) # → (X, Y, Z)
        ch_out  = output_dir / f"channel_{c:02d}.nii.gz"
        nib.save(nib.Nifti1Image(vol_xyz, affine), str(ch_out))
        print(f"  ✓ Saved {label}: {ch_out.name}  shape={vol_xyz.shape}")

    return {
        "source":  str(input_file),
        "output":  str(output_dir),
        "shape":   data.shape,
        "spacing_x_um": sx,
        "spacing_y_um": sy,
        "spacing_z_um": sz,
        "n_channels":   n_channels,
        "success":      True,
    }


def iter_oir_files(root: Path):
    """Recursively find .oir / .oi files."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".oir", ".oi"}:
            yield p


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("OIR → NIfTI Channel Conversion")
    print(f"Input:  {INPUT_ROOT.resolve()}")
    print(f"Output: {OUTPUT_ROOT.resolve()}")
    print("=" * 65)

    files = list(iter_oir_files(INPUT_ROOT))
    if not files:
        print("No .oir/.oi files found — check INPUT_ROOT path.")
        return

    print(f"Found {len(files)} .oir files\n")

    records = []
    errors  = 0

    for i, oir_path in enumerate(files, 1):
        # Preserve folder hierarchy:  subject/condition/cell_name/
        rel        = oir_path.relative_to(INPUT_ROOT)     # e.g. 02Asc-0017/+Col/File.oir
        cell_dir   = OUTPUT_ROOT / rel.parent / rel.stem  # drop .oir extension

        print(f"\n[{i}/{len(files)}] {rel}")
        try:
            rec = convert_oir_to_nifti_channels(oir_path, cell_dir)
            records.append(rec)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback; traceback.print_exc()
            records.append({"source": str(oir_path), "success": False, "error": str(e)})
            errors += 1

    # Save conversion log
    log_path = OUTPUT_ROOT / "conversion_log.csv"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(log_path, index=False)

    print("\n" + "=" * 65)
    print(f"DONE — {len(files) - errors} succeeded, {errors} failed")
    print(f"Conversion log: {log_path}")
    print("=" * 65)
    print("\nNext step:")
    print("  Run prepare_test_data_for_nnunet.py pointing --source to:")
    print(f"  {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()