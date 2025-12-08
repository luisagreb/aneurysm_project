import SimpleITK as sitk
from pathlib import Path

# INPUT folder: your nnU-Net predictions
input_dir = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/nnunet_preds_val")

# OUTPUT folder: NRRD for ParaView
output_dir = input_dir / "nrrd"
output_dir.mkdir(exist_ok=True)

nii_files = sorted(input_dir.glob("*.nii.gz"))

print(f"Found {len(nii_files)} NIfTI files")

for nii_path in nii_files:
    output_path = output_dir / nii_path.with_suffix("").with_suffix(".nrrd").name
    
    print(f"Converting: {nii_path.name} -> {output_path.name}")
    
    img = sitk.ReadImage(str(nii_path))
    sitk.WriteImage(img, str(output_path))

print("\n‹ Conversion finished successfully!")
print("NRRD files saved to:")
print(output_dir)