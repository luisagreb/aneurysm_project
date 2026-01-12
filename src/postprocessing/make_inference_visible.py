
import nibabel as nib
import numpy as np
from pathlib import Path
import os

# Configuration
INPUT_DIR = Path("/home/luisa/aneurysm_project/experiments/V2/inference_results/Dataset003_Nucleus")
OUTPUT_DIR = INPUT_DIR / "visible"

def make_visible():
    print(f"Processing files in {INPUT_DIR}...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    files = sorted(list(INPUT_DIR.glob("*.nii.gz")))
    count = 0
    
    for f in files:
        if "VISIBLE" in f.name: continue # Skip the test file we made
        
        try:
            # Load
            img = nib.load(str(f))
            data = img.get_fdata()
            
            # Multiply by 255 (0 -> 0, 1 -> 255)
            # This makes "1" appear as bright white instead of black
            data_visible = data * 255
            
            # Save
            # Use uint8 to save space (0-255 fits perfectly)
            new_img = nib.Nifti1Image(data_visible.astype(np.uint8), img.affine, img.header)
            
            output_path = OUTPUT_DIR / f.name
            nib.save(new_img, str(output_path))
            
            count += 1
            if count % 20 == 0:
                print(f"  Processed {count} files...")
                
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    print(f"\nDone! Created {count} visible files in:")
    print(f"{OUTPUT_DIR}")

if __name__ == "__main__":
    make_visible()
