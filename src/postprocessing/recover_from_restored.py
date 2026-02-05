#!/usr/bin/env python3
"""
Recover Missing Features
------------------------
Runs analyze_structures.py on the 'restored_names' directory to generate features
for all subjects, including those missing from the main pipeline (like patients 51-57).
"""

import os
import subprocess
import pandas as pd
from pathlib import Path

# Configuration
BASE_DIR = Path("/home/luisa/aneurysm_project/experiments/V2/restored_names")
OUTPUT_DIR = Path("/home/luisa/aneurysm_project/outputs/recovery")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURES = ['Actin', 'Mito', 'Nucleus']
SUBFOLDERS = ['Inference_Raw', 'Training_Labels']  # Check both inference and training

def run_analysis(structure, input_folder, output_csv):
    """Run analyze_structures.py on a specific folder."""
    if not input_folder.exists():
        print(f"Skipping {input_folder} (does not exist)")
        return None
        
    print(f"\nProcessing {structure} in {input_folder}...")
    
    cmd = [
        "python", "/home/luisa/aneurysm_project/src/postprocessing/analyze_structures.py",
        "--input_dir", str(input_folder),
        "--output_csv", str(output_csv),
        "--structure", structure.lower()
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return output_csv
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_folder}: {e}")
        return None

def main():
    all_dfs = []
    
    for struct in STRUCTURES:
        for sub in SUBFOLDERS:
            input_path = BASE_DIR / struct / sub
            output_csv = OUTPUT_DIR / f"{struct.lower()}_{sub.lower()}_features.csv"
            
            if input_path.exists():
                # checks if files exist
                files = list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.tif"))
                if not files:
                    print(f"No files in {input_path}")
                    continue

                res = run_analysis(struct, input_path, output_csv)
                
                if res and res.exists():
                    df = pd.read_csv(res)
                    # Add metadata columns to distinguish
                    # Rename columns to match Advanced_Features_Raw format (Structure_Feature)
                    new_cols = []
                    for c in df.columns:
                        if c in ['Filename', 'Voxel_Source', 'Voxel_X', 'Voxel_Y', 'Voxel_Z']:
                            new_cols.append(c)
                        else:
                            new_cols.append(f"{struct}_{c}")
                    df.columns = new_cols
                    
                    df['Dataset_Source'] = sub # Training or Inference
                    all_dfs.append(df)

    if all_dfs:
        print("\nCombining all recovered features...")
        # We concatenate. Note: This creates a LONG format (rows for Actin, rows for Mito).
        # This matches the format of Advanced_Features_Raw.csv
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        final_csv = "/home/luisa/aneurysm_project/outputs/Advanced_Features_Restored.csv"
        full_df.to_csv(final_csv, index=False)
        print(f"SUCCESS: Saved comprehensive feature list to {final_csv}")
        print(f"Total Rows: {len(full_df)}")
        
        # Check for missing patients
        print("\nChecking for missing patients (51, 54, 55, 56, 57)...")
        targets = [51, 54, 55, 56, 57]
        for t in targets:
            count = full_df['Filename'].apply(lambda x: str(t) in str(x)).sum()
            print(f"  Patient {t}: Found {count} rows")
            
    else:
        print("No features recovered.")

if __name__ == "__main__":
    main()
