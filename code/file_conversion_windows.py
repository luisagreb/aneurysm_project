from pathlib import Path
import numpy as np
import nibabel as nib
from aicsimageio import AICSImage
from aicsimageio.readers import BioformatsReader

# NAS input
INPUT_ROOT = Path(r"Y:\Luisa\Marie's Data\SMCs_Zstacks\SMCs_Zstacks")

# Local output (IMPORTANT: use raw string or forward slashes)
OUTPUT_ROOT = Path(r"C:\Users\Luisa\Documents\aneurysm_project\Data")


def convert_oir_to_nifti_channels(input_file: Path) -> None:
    """
    Convert one .oir file into:
      - multi-channel .nii.gz
      - per-channel .nii.gz (ch1 nucleus, ch2 actin, ch3 mitochondria)
    and save to a mirrored folder under OUTPUT_ROOT.
    """
    print(f"📂 Converting: {input_file}")

    # Build mirrored output directory:
    # relative path of this file w.r.t INPUT_ROOT
    relative = input_file.relative_to(INPUT_ROOT)
    out_dir = OUTPUT_ROOT / relative.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_file.stem

    # Load with BioFormats
    img = AICSImage(str(input_file), reader=BioformatsReader)

    # Load in CZYX order
    data = img.get_image_data("CZYX").astype(np.float32)

    # Voxel sizes → affine
    ps = img.physical_pixel_sizes
    if ps and ps.X and ps.Y:
        zsz = ps.Z if ps.Z else ps.Y
        sx, sy, sz = float(ps.X) / 1000, float(ps.Y) / 1000, float(zsz) / 1000
        affine = np.diag([sx, sy, sz, 1.0])
    else:
        affine = np.eye(4, dtype=np.float32)

    # ---- SAVE MULTICHANNEL ----
    mc_out = out_dir / f"{base_name}_allchannels.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(mc_out))
    print(f"   ✔ Saved: {mc_out}")

    # ---- SAVE PER CHANNEL ----
    for c in range(data.shape[0]):
        ch_data = data[c]
        ch_out = out_dir / f"{base_name}_ch{c+1}.nii.gz"
        nib.save(nib.Nifti1Image(ch_data, affine), str(ch_out))
        print(f"   ✔ Saved channel {c+1}: {ch_out}")


def iter_oir_files(root: Path):
    """Recursively find .oir / .oi files."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".oir", ".oi"}:
            yield p


def main():
    print(f"🔎 Scanning directory: {INPUT_ROOT}")
    files = list(iter_oir_files(INPUT_ROOT))

    if not files:
        print("❌ No .oir/.oi files found.")
        return

    print(f"📦 Found {len(files)} files to process.")

    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]")
        try:
            convert_oir_to_nifti_channels(f)
        except Exception as e:
            print(f"💥 Error with {f}: {e}")


if __name__ == "__main__":
    main()