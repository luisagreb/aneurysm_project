
import nibabel as nib
import numpy as np
import glob
from pathlib import Path

def check(folder):
    print(f"Checking {folder}...")
    files = sorted(glob.glob(f"{folder}/*.nii.gz"))[:5]
    for f in files:
        img = nib.load(f)
        data = img.get_fdata()
        unique = np.unique(data)
        print(f"{Path(f).name}: {unique}, type={data.dtype}")

check("/home/luisa/aneurysm_project/data/nnUnet_actin/Dataset001_Actin/labelsTr")
check("/home/luisa/aneurysm_project/data/nnUnet_mitochondria/Dataset002_Mito/labelsTr")
