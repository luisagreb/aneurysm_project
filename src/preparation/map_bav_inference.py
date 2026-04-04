#!/usr/bin/env python3
"""
map_bav_inference.py
====================
Maps the new BAV inference results to their original real names
within the data/organized_data/ folder. 
This will overwrite any old segmentations for these cells with the new BAV results.
"""

import os
import csv
import shutil
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_BASE     = PROJECT_ROOT / "data" / "organized_data"

def sanitize(name: str) -> str:
    return re.sub(r'[():]', '', name).strip()

def copy_file(src: Path, dst: Path):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    return True

def main():
    mapping_csv = PROJECT_ROOT / "outputs" / "new_BAV_inference_mapping.csv"
    if not mapping_csv.exists():
        print(f"Mapping csv not found: {mapping_csv}")
        return

    # The actual inference outputs were saved here
    inf_root = PROJECT_ROOT / "experiments" / "V2" / "inference_results" / "new_BAV"

    with open(mapping_csv, 'r') as f:
        rows = list(csv.DictReader(f))

    print(f"Mapping {len(rows)} BAV inference results to original names in {OUT_BASE} ...")

    ok = missing = 0
    for row in rows:
        # e.g. "01Asc-0231/+Col/01Asc-231 +coll Cell 1"
        cell_full = row.get("cell", "").strip()
        if not cell_full:
            continue
            
        parts = cell_full.split('/')
        if len(parts) != 3:
            print(f"Skipping malformed cell name: {cell_full}")
            continue
            
        subject, condition, cell_name = parts
        
        # Clean condition name
        if "+" in condition:
            cond_folder = "+Collagen"
        else:
            cond_folder = "-Collagen"
            
        cell_safe = sanitize(cell_name)
        dst_root  = OUT_BASE / sanitize(subject) / cond_folder / cell_safe / "segmentation"
        
        # nnU-Net prediction names don't have the _0000 suffix like the input files do
        actin_case = row.get("Actin_case", "").strip().replace("_0000.nii.gz", ".nii.gz")
        mito_case  = row.get("Mito_case", "").strip().replace("_0000.nii.gz", ".nii.gz")
        nuc_case   = row.get("Nucleus_case", "").strip().replace("_0000.nii.gz", ".nii.gz")
        
        seg_pairs = [
            ("actin_seg.nii.gz",        inf_root / "Actin"   / actin_case),
            ("mitochondria_seg.nii.gz", inf_root / "Mito"    / mito_case),
            ("nucleus_seg.nii.gz",      inf_root / "Nucleus" / nuc_case),
        ]
        
        for dst_name, src in seg_pairs:
            if src.exists():
                copy_file(src, dst_root / dst_name)
            else:
                missing += 1
                print(f"Missing source file: {src}")
                
        ok += 1

    print(f"Done mapping {ok} cells! ({missing} segmentation files not found)")

if __name__ == "__main__":
    main()
