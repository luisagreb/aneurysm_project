
import nibabel as nib
import numpy as np
from pathlib import Path
import glob

# Path to inference results
results_dir = Path("/home/luisa/aneurysm_project/experiments/V2/inference_results/Dataset001_Actin")
files = sorted(list(results_dir.glob("*.nii.gz")))[:20]

print(f"Checking {len(files)} files in {results_dir}...\n")
print(f"{'Filename':<30} | {'Shape':<20} | {'Unique Vals':<15} | {'Non-Zero Vx':<10}")
print("-" * 85)

for f in files:
    try:
        img = nib.load(str(f))
        data = img.get_fdata()
        unique = np.unique(data)
        non_zero = np.count_nonzero(data)
        
        # Format for readability
        shape_str = str(data.shape)
        unique_str = str(unique) if len(unique) < 5 else f"{unique[:2]}...{unique[-1]}"
        
        print(f"{f.name:<30} | {shape_str:<20} | {unique_str:<15} | {non_zero:<10}")
    except Exception as e:
        print(f"{f.name:<30} | ERROR: {e}")
