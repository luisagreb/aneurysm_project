"""
Feature Extraction Pipeline — Main Dataset (200 cells)
-------------------------------------------------------
Runs analyze_structures.py on all restored-name segmentation files for
Actin, Mito, and Nucleus, then merges the three per-structure CSVs into
one wide-format master CSV (one row per cell).

Segmentation sources:
  experiments/V2/restored_names/<Structure>/Inference_Raw/
  experiments/V2/restored_names/<Structure>/Training_Labels/

Voxel sizes are looked up per-cell from the raw NRRD files.

Output:
  outputs/features_actin.csv
  outputs/features_mito.csv
  outputs/features_nucleus.csv
  outputs/Advanced_Features_Raw_200.csv   ← wide-format master

Usage:
  python src/postprocessing/run_feature_extraction.py
  python src/postprocessing/run_feature_extraction.py --structures actin mito
  python src/postprocessing/run_feature_extraction.py --dry_run
"""

import argparse
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path('/home/luisa/aneurysm_project')
SEG_ROOT     = BASE_DIR / 'experiments/V2/restored_names'
NRRD_DIR     = BASE_DIR / 'data/raw/nrrd_files'
OUTPUT_DIR   = BASE_DIR / 'outputs'
ANALYZE_SCRIPT = BASE_DIR / 'src/postprocessing/analyze_structures.py'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURES = {
    'actin':   {'folder': 'Actin',   'csv': OUTPUT_DIR / 'features_actin.csv'},
    'mito':    {'folder': 'Mito',    'csv': OUTPUT_DIR / 'features_mito.csv'},
    'nucleus': {'folder': 'Nucleus', 'csv': OUTPUT_DIR / 'features_nucleus.csv'},
}
SUBDIRS = ['Inference_Raw', 'Training_Labels']


def collect_seg_files(struct_folder):
    """Return list of (input_dir, files) pairs covering both Inference_Raw and Training_Labels."""
    dirs = []
    for sub in SUBDIRS:
        d = SEG_ROOT / struct_folder / sub
        if d.exists():
            files = list(d.glob('*.nii.gz'))
            if files:
                dirs.append((d, len(files)))
    return dirs


def run_analysis(structure, input_dir, output_csv, dry_run=False):
    """Call analyze_structures.py as a subprocess."""
    cmd = [
        sys.executable, str(ANALYZE_SCRIPT),
        '--input_dir',    str(input_dir),
        '--output_csv',   str(output_csv),
        '--structure',    structure,
        '--raw_nrrd_dir', str(NRRD_DIR),
    ]
    print(f"  {'[DRY RUN] ' if dry_run else ''}Running: {' '.join(cmd)}")
    if not dry_run:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0
    return True


def combine_structure_csvs(struct, inf_csv, tr_csv, out_csv):
    """Concatenate inference + training CSVs for one structure, prefix columns."""
    dfs = []
    for csv_path, source in [(inf_csv, 'Inference_Raw'), (tr_csv, 'Training_Labels')]:
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path)
            df['DataSource'] = source
            dfs.append(df)

    if not dfs:
        print(f"  WARNING: no data found for {struct}")
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # Prefix all feature columns with structure name
    skip = {'Filename', 'Voxel_Source', 'Voxel_Z', 'Voxel_Y', 'Voxel_X', 'DataSource'}
    combined.columns = [
        f"{struct.capitalize()}_{c}" if c not in skip else c
        for c in combined.columns
    ]
    # Rename Filename → CellName and strip segmentation suffix
    combined.rename(columns={'Filename': 'CellName'}, inplace=True)
    combined['CellName'] = combined['CellName'].str.replace(
        '_segmentation.nii.gz', '', regex=False).str.replace('.nii.gz', '', regex=False)

    combined.to_csv(out_csv, index=False)
    print(f"  Saved {len(combined)} rows → {out_csv}")
    return combined


def merge_to_wide(actin_df, mito_df, nucleus_df):
    """
    Merge the three structure DataFrames on CellName into one wide-format CSV.
    Each row = one cell with Actin_*, Mito_*, Nucleus_* columns.
    """
    # Use actin as the base (all cells should have actin)
    merged = actin_df.copy()

    voxel_cols = ['Voxel_Source', 'Voxel_Z', 'Voxel_Y', 'Voxel_X']

    for df, prefix in [(mito_df, 'Mito'), (nucleus_df, 'Nucleus')]:
        if df is None:
            continue
        # Drop duplicate voxel/source columns from the right side before merge
        drop_cols = [c for c in voxel_cols + ['DataSource'] if c in df.columns]
        df_merge = df.drop(columns=drop_cols, errors='ignore')
        merged = merged.merge(df_merge, on='CellName', how='outer')

    print(f"\nWide-format merge: {len(merged)} cells, {len(merged.columns)} columns")
    return merged


def main():
    parser = argparse.ArgumentParser(description='Run full feature extraction pipeline.')
    parser.add_argument('--structures', nargs='+', default=['actin', 'mito', 'nucleus'],
                        choices=['actin', 'mito', 'nucleus'],
                        help='Which structures to process (default: all three)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without running them')
    args = parser.parse_args()

    print("=" * 60)
    print("Feature Extraction Pipeline — 200-cell Main Dataset")
    print("=" * 60)

    # ── Step 1: Run analyze_structures.py per structure per subdir ────────────
    per_struct_csvs = {}   # struct → (inf_csv, tr_csv)

    for struct in args.structures:
        cfg = STRUCTURES[struct]
        print(f"\n>>> {struct.upper()}")
        dirs = collect_seg_files(cfg['folder'])
        if not dirs:
            print(f"  No segmentation files found for {struct}, skipping.")
            continue

        inf_csv = OUTPUT_DIR / f'features_{struct}_inference.csv'
        tr_csv  = OUTPUT_DIR / f'features_{struct}_training.csv'

        for input_dir, n_files in dirs:
            is_training = 'Training' in input_dir.name
            out_csv = tr_csv if is_training else inf_csv
            print(f"  {input_dir.name}: {n_files} files → {out_csv.name}")
            success = run_analysis(struct, input_dir, out_csv, dry_run=args.dry_run)
            if not success:
                print(f"  ERROR: analyze_structures.py failed for {input_dir}")

        per_struct_csvs[struct] = (inf_csv, tr_csv)

    if args.dry_run:
        print("\n[DRY RUN] Stopping before merge.")
        return

    # ── Step 2: Combine inference + training per structure ────────────────────
    print("\n>>> Combining inference + training CSVs per structure")
    struct_dfs = {}
    for struct, (inf_csv, tr_csv) in per_struct_csvs.items():
        cfg = STRUCTURES[struct]
        df = combine_structure_csvs(struct, inf_csv, tr_csv, cfg['csv'])
        struct_dfs[struct] = df

    # ── Step 3: Merge into wide-format master CSV ─────────────────────────────
    print("\n>>> Merging into wide-format master CSV")
    actin_df   = struct_dfs.get('actin')
    mito_df    = struct_dfs.get('mito')
    nucleus_df = struct_dfs.get('nucleus')

    if actin_df is None:
        print("ERROR: Actin data missing, cannot build master CSV.")
        return

    master = merge_to_wide(actin_df, mito_df, nucleus_df)

    master_path = OUTPUT_DIR / 'Advanced_Features_Raw_200.csv'
    master.to_csv(master_path, index=False)
    print(f"\nMaster CSV saved → {master_path}")
    print(f"  {len(master)} cells  |  {len(master.columns)} columns")

    # Quick sanity check
    print("\nColumn groups:")
    for prefix in ['Actin', 'Mito', 'Nucleus']:
        cols = [c for c in master.columns if c.startswith(f'{prefix}_')]
        nulls = master[cols].isnull().any(axis=1).sum()
        print(f"  {prefix}: {len(cols)} features  |  {nulls} cells with any NaN")

    print("\nDone.")


if __name__ == '__main__':
    main()
