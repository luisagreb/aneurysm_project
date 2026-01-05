
import os
from pathlib import Path
import unicodedata

DATA_ROOT = Path("/home/luisa/aneurysm_project/data")
RAW_DIR = DATA_ROOT / "nrrd_files"

DATASETS = [
    {"name": "Actin", "label_dir": "manually_segmented_actin", "raw_channel": "channel_01.nrrd"},
    {"name": "Mito", "label_dir": "manually_segmented_mitochondria", "raw_channel": "channel_02.nrrd"}
]

def normalize_name(name):
    # Standardize: lowercase, strip, no spaces
    return unicodedata.normalize('NFC', name).lower().strip().replace(" ", "")

def analyze_matching(ds):
    print(f"\n=== MATCHING REPORT: {ds['name']} ===")
    label_dir = DATA_ROOT / ds['label_dir']
    
    # 1. Load Resources
    raw_cells = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    label_files = sorted(list(label_dir.glob("*.nrrd")))
    
    used_raw = set()
    used_lbl = set()
    
    exact_matches = []
    fuzzy_matches = []
    
    # --- PASS 1: Normalize Strict ---
    label_map = {normalize_name(f.stem.replace(".seg", "")): f for f in label_files}
    
    for cell in raw_cells:
        raw_path = cell / ds['raw_channel']
        if not raw_path.exists(): continue
        
        norm_cell = normalize_name(cell.name)
        if norm_cell in label_map:
            exact_matches.append(f"{cell.name}  <==>  {label_map[norm_cell].name}")
            used_raw.add(raw_path)
            used_lbl.add(label_map[norm_cell])

    # --- PASS 2: Fuzzy (The "Difficult" ones) ---
    unused_raw = [d for d in raw_cells if (d/ds['raw_channel']) not in used_raw and (d/ds['raw_channel']).exists()]
    unused_lbl = [f for f in label_files if f not in used_lbl]
    
    for lf in unused_lbl:
        norm_lbl = normalize_name(lf.stem.replace(".seg", ""))
        
        match_found = None
        for cell in unused_raw:
            raw_path = cell / ds['raw_channel']
            if raw_path in used_raw: continue
            
            norm_raw = normalize_name(cell.name)
            
            # Logic: strip +/- and check substring
            n_lbl_cl = norm_lbl.replace("+", "").replace("-", "")
            n_raw_cl = norm_raw.replace("+", "").replace("-", "")
            
            if n_lbl_cl in n_raw_cl or n_raw_cl in n_lbl_cl:
                match_found = cell
                break
        
        if match_found:
            fuzzy_matches.append(f"Label: '{lf.name}'  Mapped to Cell: '{match_found.name}'")
            used_raw.add(match_found / ds['raw_channel'])
            used_lbl.add(lf)
            
    # Report
    print(f"Total Labels: {len(label_files)}")
    print(f"Exact Matches: {len(exact_matches)}")
    print(f"Fuzzy/Difficult Matches: {len(fuzzy_matches)}")
    if fuzzy_matches:
        print("--- List of Difficult Matches ---")
        for m in fuzzy_matches:
            print(m)
            
    print(f"Unmatched Labels: {len(unused_lbl) - len(fuzzy_matches)}")
    remaining = [f.name for f in unused_lbl if f not in used_lbl]
    if remaining:
        print("--- Unmatched Labels ---")
        for r in remaining:
            print(r)

def main():
    for ds in DATASETS:
        analyze_matching(ds)

if __name__ == "__main__":
    main()
