
import os
from pathlib import Path
import unicodedata

DATA_ROOT = Path("/home/luisa/aneurysm_project/data")
RAW_DIR = DATA_ROOT / "nrrd_files"

def normalize_name(name):
    name = unicodedata.normalize('NFC', name)
    return name.lower().strip().replace(" ", "")

def check_orphans(label_dir_name):
    label_dir = DATA_ROOT / label_dir_name
    raw_folders = {normalize_name(d.name): d.name for d in RAW_DIR.iterdir() if d.is_dir()}
    
    print(f"\n--- Checking {label_dir_name} ---")
    orphans = []
    
    for f in label_dir.glob("*.nrrd"):
        norm_stem = normalize_name(f.stem)
        if norm_stem not in raw_folders:
            print(f"Orphan Label: {f.name}")
            orphans.append(f.name)
        else:
            # print(f"Matched: {f.name}")
            pass
            
    print(f"Total Orphans: {len(orphans)} / {len(list(label_dir.glob('*.nrrd')))}")

def main():
    check_orphans("manually_segmented_actin")
    check_orphans("manually_segmented_mitochondria")

if __name__ == "__main__":
    main()
