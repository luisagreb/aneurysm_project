
import nrrd
from pathlib import Path
import numpy as np

DATA_ROOT = Path("/home/luisa/aneurysm_project/data")
RAW_DIR = DATA_ROOT / "nrrd_files"

TARGET_SHAPE = {11, 1023, 961} # Set for permutation invariance check

def scan():
    print("Scanning for raw files with shape (11, 1023, 961)...")
    matches = []
    
    for d in RAW_DIR.iterdir():
        if not d.is_dir(): continue
        raw_path = d / "channel_01.nrrd"
        if not raw_path.exists(): continue
        
        try:
            h = nrrd.read_header(str(raw_path))
            sizes = h['sizes']
            if set(sizes) == TARGET_SHAPE:
                print(f"MATCH FOUND: {d.name} -> {sizes}")
                matches.append(d.name)
        except:
            pass
            
    if not matches:
        print("No matches found.")
    else:
        print(f"Found {len(matches)} potential raw files.")

if __name__ == "__main__":
    scan()
