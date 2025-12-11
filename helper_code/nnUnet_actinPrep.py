from pathlib import Path
import shutil

project_root = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project")

raw_dir    = project_root / "data" / "raw_ch2_actin_all"
nnunet_dir = project_root / "data" / "ch2_nnunet"

raw_files = sorted(raw_dir.glob("*.nii.gz"))
nnunet_dir.mkdir(parents=True, exist_ok=True)

for idx, src in enumerate(raw_files):
    case_name = f"case_{idx:04d}_0000.nii.gz"
    dst = nnunet_dir / case_name
    shutil.copy2(src, dst)

print(" Built Actin nnUNet inputs:", len(raw_files))