#!/usr/bin/env python3
"""
Create final Advanced_Features_Raw.csv with:
1. Per-cell voxel sizes read from original NRRD files
2. Units included in all feature column names
"""

import pandas as pd
import numpy as np
import nibabel as nib
import nrrd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'postprocessing'))
from analyze_structures import analyze_actin, analyze_mito, analyze_nucleus


def get_voxel_size_from_nrrd(cell_name, raw_nrrd_dir):
    """Get exact voxel size from original NRRD file for a cell."""
    # Clean up cell name (remove _segmentation suffix)
    cell_name_clean = cell_name.replace('_segmentation', '').replace('.nii.gz', '')
    
    # Try to find matching folder
    nrrd_folder = raw_nrrd_dir / cell_name_clean
    
    if nrrd_folder.exists():
        # Find a non-mask nrrd file
        nrrd_files = [f for f in nrrd_folder.glob('channel_*.nrrd') if 'mask' not in f.name]
        if nrrd_files:
            try:
                header = nrrd.read_header(str(nrrd_files[0]))
                if 'space directions' in header:
                    dirs = header['space directions']
                    spacing = [float(np.linalg.norm(d)) if hasattr(d, '__len__') else float(d) for d in dirs]
                    return spacing
                elif 'spacings' in header:
                    return [float(s) for s in header['spacings']]
            except Exception as e:
                print(f"  Warning: Could not read voxel from {nrrd_folder.name}: {e}")
    
    # Fallback - return None to indicate we couldn't find it
    return None


def process_structure(structure_name, input_dirs, analyze_func, raw_nrrd_dir):
    """Process all files for a structure from multiple directories."""
    results = []
    found_voxel = 0
    fallback_voxel = 0
    
    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"  Skipping {input_dir} (not found)")
            continue
            
        files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nii.gz')])
        print(f"  {input_dir.name}: {len(files)} files")
        
        for filename in files:
            filepath = input_dir / filename
            
            # Extract cell name
            cell_name = filename.replace('_segmentation.nii.gz', '').replace('.nii.gz', '')
            
            # Get exact voxel size from original NRRD
            voxel_size = get_voxel_size_from_nrrd(filename, raw_nrrd_dir)
            
            if voxel_size is None:
                # Skip file if voxel size not found - prevents data contamination
                print(f"    WARNING: Skipping {filename} - voxel size not found in NRRD")
                continue
            else:
                found_voxel += 1
            
            # Load mask
            nii = nib.load(filepath)
            mask = nii.get_fdata()
            mask = (mask > 0).astype(np.uint8)
            
            # Skip empty masks
            if np.sum(mask) < 10:
                continue
            
            # Analyze with correct voxel size
            metrics = analyze_func(mask, voxel_size)
            
            metrics['CellName'] = cell_name
            metrics['DataSource'] = input_dir.name
            metrics['Voxel_Z_µm'] = voxel_size[0]
            metrics['Voxel_Y_µm'] = voxel_size[1]
            metrics['Voxel_X_µm'] = voxel_size[2]
            
            results.append(metrics)
    
    print(f"  Voxel sizes: {found_voxel} from NRRD, {fallback_voxel} fallback")
    return pd.DataFrame(results)


# Define column name to unit mapping
ACTIN_UNITS = {
    'Volume': 'Volume_µm³',
    'Skeleton_Length': 'Skeleton_Length_µm',
    'Convex_Hull_Volume': 'Convex_Hull_Volume_µm³',
    'Solidity': 'Solidity_ratio',
    'Extent': 'Extent_ratio',
    'Fractional_Anisotropy': 'Fractional_Anisotropy_ratio',
    'Major_Axis': 'Major_Axis_µm',
    'Intermediate_Axis': 'Intermediate_Axis_µm',
    'Minor_Axis': 'Minor_Axis_µm',
}

MITO_UNITS = {
    'Volume': 'Volume_µm³',
    'Surface_Area': 'Surface_Area_µm²',
    'Sphericity': 'Sphericity_ratio',
    'Fragment_Count': 'Fragment_Count_n',
    'Mean_Fragment_Sphericity': 'Mean_Fragment_Sphericity_ratio',
    'Std_Fragment_Sphericity': 'Std_Fragment_Sphericity_ratio',
    'Min_Fragment_Sphericity': 'Min_Fragment_Sphericity_ratio',
    'Max_Fragment_Sphericity': 'Max_Fragment_Sphericity_ratio',
    'Mean_Fragment_Volume': 'Mean_Fragment_Volume_µm³',
    'Junction_Count': 'Junction_Count_n',
    'Branch_Count': 'Branch_Count_n',
    'Mean_Branch_Length': 'Mean_Branch_Length_µm',
    'Total_Network_Length': 'Total_Network_Length_µm',
    'Mean_Tortuosity': 'Mean_Tortuosity_ratio',
    'Cyclomatic_Number': 'Cyclomatic_Number_n',
}

NUCLEUS_UNITS = {
    'Volume': 'Volume_µm³',
    'Sphericity': 'Sphericity_ratio',
    'Elongation': 'Elongation_ratio',
    'Flatness': 'Flatness_ratio',
    'Solidity': 'Solidity_ratio',
}


def add_units_to_columns(df, unit_map, prefix):
    """Rename columns to include units."""
    rename_map = {}
    for col in df.columns:
        if col in ['CellName', 'Filename', 'DataSource', 'Voxel_Z_µm', 'Voxel_Y_µm', 'Voxel_X_µm']:
            continue
        if col in unit_map:
            rename_map[col] = f"{prefix}_{unit_map[col]}"
        else:
            rename_map[col] = f"{prefix}_{col}"
    return df.rename(columns=rename_map)


def main():
    base_dir = Path('/home/luisa/aneurysm_project')
    restored_dir = base_dir / 'experiments/V2/restored_names'
    raw_nrrd_dir = base_dir / 'data/raw/nrrd_files'
    
    print("=== Feature Extraction with Per-Cell Voxel Sizes ===\n")
    
    # Process Actin
    print("Processing Actin...")
    actin_df = process_structure('Actin', [
        restored_dir / 'Actin/Inference_Raw',
        restored_dir / 'Actin/Training_Labels',
    ], analyze_actin, raw_nrrd_dir)
    print(f"  Total: {len(actin_df)} cells\n")
    
    # Process Mito
    print("Processing Mito...")
    mito_df = process_structure('Mito', [
        restored_dir / 'Mito/Inference_Raw',
        restored_dir / 'Mito/Training_Labels',
    ], analyze_mito, raw_nrrd_dir)
    print(f"  Total: {len(mito_df)} cells\n")
    
    # Process Nucleus
    print("Processing Nucleus...")
    nucleus_df = process_structure('Nucleus', [
        restored_dir / 'Nucleus/Inference_Raw',
        restored_dir / 'Nucleus/Training_Labels',
    ], analyze_nucleus, raw_nrrd_dir)
    print(f"  Total: {len(nucleus_df)} cells\n")
    
    # Add units to column names
    actin_with_units = add_units_to_columns(actin_df, ACTIN_UNITS, 'Actin')
    mito_with_units = add_units_to_columns(mito_df, MITO_UNITS, 'Mito')
    nucleus_with_units = add_units_to_columns(nucleus_df, NUCLEUS_UNITS, 'Nucleus')
    
    # Save individual CSVs with units
    actin_with_units.to_csv(base_dir / 'outputs/actin_features_final.csv', index=False)
    mito_with_units.to_csv(base_dir / 'outputs/mito_features_final.csv', index=False)
    nucleus_with_units.to_csv(base_dir / 'outputs/nucleus_features_final.csv', index=False)
    print("Saved individual CSVs to outputs/")
    
    # Merge on CellName
    print("\nMerging features...")
    actin_cols = ['CellName', 'DataSource', 'Voxel_Z_µm', 'Voxel_Y_µm', 'Voxel_X_µm'] + \
                 [c for c in actin_with_units.columns if c.startswith('Actin_')]
    mito_cols = ['CellName'] + [c for c in mito_with_units.columns if c.startswith('Mito_')]
    nucleus_cols = ['CellName'] + [c for c in nucleus_with_units.columns if c.startswith('Nucleus_')]
    
    combined = actin_with_units[actin_cols].merge(
        mito_with_units[mito_cols], on='CellName', how='outer'
    )
    combined = combined.merge(
        nucleus_with_units[nucleus_cols], on='CellName', how='outer'
    )
    
    # Save final CSV
    output_path = base_dir / 'outputs/Advanced_Features_Raw.csv'
    combined.to_csv(output_path, index=False)
    
    print(f"\n=== FINAL OUTPUT ===")
    print(f"File: {output_path}")
    print(f"Total cells: {len(combined)}")
    print(f"Total columns: {len(combined.columns)}")
    print(f"Cells with all three structures: {combined.dropna().shape[0]}")
    print(f"\nColumn names:\n{combined.columns.tolist()}")


if __name__ == '__main__':
    main()
