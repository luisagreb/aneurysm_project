"""
Feature Extraction Pipeline — 3-Group Dataset (TAV-NA / TAV-ATAA / BAV-ATAA)
=============================================================================
Crawls data/organized_data_PostBAVAddition/, extracts morphological features
for every cell using the existing analyze_actin / analyze_mito / analyze_nucleus
functions, and writes a master CSV with group labels.

Voxel size strategy:
  - NRRD raw files  → read from space-directions header (accurate)
  - NIfTI raw files → header contains 1.0 (metadata lost); use estimated
                      DEFAULT_VOXEL = [0.37, 0.207, 0.207] µm (Z, Y, X)
                      and flag Voxel_Source as 'Estimated'

Output:
  outputs/Advanced_Features_3groups.csv

Usage:
  cd /home/luisa/aneurysm_project
  python src/postprocessing/extract_features_3groups.py
  python src/postprocessing/extract_features_3groups.py --dry_run
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import nrrd
from pathlib import Path
from scipy.ndimage import binary_closing

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path('/home/luisa/aneurysm_project')
DATA_ROOT = BASE_DIR / 'data/organized_data_PostBAVAddition'
OUT_CSV   = BASE_DIR / 'outputs/Advanced_Features_3groups.csv'

# ── Group assignments (from specimen spreadsheet) ─────────────────────────────
# Keys are lowercase specimen IDs as they appear in the folder names
GROUP_MAP = {
    # TAV-NA — Tricuspid Aortic Valve, Non-Aneurysmal
    '01asc-0180': ('TAV-NA', 62, 'Male'),
    '01asc-0222': ('TAV-NA', 66, 'Male'),
    '01asc-230':  ('TAV-NA', 63, 'Female'),
    '01c-0083':   ('TAV-NA', None, None),
    '01c-0096':   ('TAV-NA', 53, 'Male'),
    '01c-0097':   ('TAV-NA', 57, 'Female'),
    '01c-0113':   ('TAV-NA', 64, 'Female'),
    '01c-0117':   ('TAV-NA', 46, 'Female'),
    '01c-202':    ('TAV-NA', 78, 'Female'),
    '01asc-0231': ('TAV-NA', 60, 'Male'),
    '01asc-0232': ('TAV-NA', 49, 'Male'),
    # TAV-ATAA — Tricuspid Aortic Valve with Ascending TAA
    '03asc-024':  ('TAV-ATAA', 62, 'Male'),
    '03asc-0043': ('TAV-ATAA', 70, 'Male'),
    '03rt-0045':  ('TAV-ATAA', 75, 'Male'),
    '03asc-0046': ('TAV-ATAA', 75, 'Female'),
    '03asc-0047': ('TAV-ATAA', 68, 'Male'),
    '03asc-0051': ('TAV-ATAA', 65, 'Female'),
    '03asc-0054': ('TAV-ATAA', 75, 'Female'),
    '03asc-0055': ('TAV-ATAA', 66, 'Female'),
    '03asc-0056': ('TAV-ATAA', 44, 'Female'),
    '03asc-0057': ('TAV-ATAA', 73, 'Female'),
    '02asc-0047': ('TAV-ATAA', 40, 'Male'),
    # BAV-ATAA — Bicuspid Aortic Valve with Ascending TAA
    '02asc-0017': ('BAV-ATAA', 73, 'Male'),
    '02asc-0021': ('BAV-ATAA', 61, 'Male'),
    '02asc-0029': ('BAV-ATAA', 64, 'Female'),
    '02asc-0049': ('BAV-ATAA', 58, 'Female'),
    '02asc-0077': ('BAV-ATAA', None, 'Male'),
}

# Fallback voxel size for specimens whose raw files have lost metadata (nii.gz, 1.0)
# Z=0.37µm is consistent across all known specimens; X/Y=0.207µm is the most common value
DEFAULT_VOXEL_ZYX = [0.37, 0.207, 0.207]  # µm, order: Z, Y, X

# ── Import analysis functions ─────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR / 'src/postprocessing'))
from analyze_structures import analyze_actin, analyze_mito, analyze_nucleus


def get_voxel_size(raw_dir: Path):
    """
    Read voxel size from raw files in a cell directory.
    Returns (voxel_size_xyz, source_label) where voxel_size is [vx, vy, vz]
    (NIfTI axis order, as expected by analyze_* functions).
    """
    # Try NRRD first
    nrrd_files = list(raw_dir.glob('*.nrrd'))
    if nrrd_files:
        try:
            header = nrrd.read_header(str(nrrd_files[0]))
            if 'space directions' in header:
                dirs = header['space directions']
                spacing = [float(np.linalg.norm(d)) for d in dirs]  # [vz, vy, vx]
                # Reverse to NIfTI order [vx, vy, vz]
                vox_xyz = [spacing[2], spacing[1], spacing[0]]
                return vox_xyz, 'NRRD'
        except Exception as e:
            print(f"    Warning: NRRD read failed ({e}), falling back to default")

    # nii.gz raw — header is 1.0 (metadata lost), use default
    nii_files = list(raw_dir.glob('*.nii.gz'))
    if nii_files:
        try:
            nii = nib.load(str(nii_files[0]))
            zooms = list(nii.header.get_zooms()[:3])
            # If all 1.0, metadata lost
            if all(abs(z - 1.0) < 0.01 for z in zooms):
                vz, vy, vx = DEFAULT_VOXEL_ZYX
                # NIfTI axis order is X,Y,Z
                return [vx, vy, vz], 'Estimated'
            else:
                # Header has real values (unlikely but handle it)
                return list(zooms), 'NIfTI_Header'
        except Exception as e:
            print(f"    Warning: NIfTI read failed ({e}), using default voxel size")

    # Last resort
    vz, vy, vx = DEFAULT_VOXEL_ZYX
    return [vx, vy, vz], 'Default'


def load_seg(seg_path: Path):
    """Load a segmentation NIfTI and return binary mask."""
    nii = nib.load(str(seg_path))
    mask = (nii.get_fdata() > 0).astype(np.uint8)
    mask = binary_closing(mask, structure=np.ones((3, 3, 3))).astype(np.uint8)
    return mask


def process_cell(cell_dir: Path, specimen_id: str, collagen_status: str,
                 dry_run: bool = False):
    """
    Process one cell directory.
    Returns a dict of features, or None on failure.
    """
    seg_dir = cell_dir / 'segmentation'
    raw_dir = cell_dir / 'raw'

    if not seg_dir.exists():
        print(f"  SKIP (no segmentation dir): {cell_dir.name}")
        return None

    actin_seg  = seg_dir / 'actin_seg.nii.gz'
    mito_seg   = seg_dir / 'mitochondria_seg.nii.gz'
    nucleus_seg = seg_dir / 'nucleus_seg.nii.gz'

    if not all(p.exists() for p in [actin_seg, mito_seg, nucleus_seg]):
        missing = [p.name for p in [actin_seg, mito_seg, nucleus_seg] if not p.exists()]
        print(f"  SKIP (missing segs {missing}): {cell_dir.name}")
        return None

    if dry_run:
        print(f"  [DRY] Would process: {cell_dir.name}")
        return {'CellName': cell_dir.name, 'DRY_RUN': True}

    # Voxel size
    vox_xyz, vox_source = get_voxel_size(raw_dir)
    vx, vy, vz = vox_xyz  # NIfTI order: X, Y, Z

    row = {
        'CellName':       cell_dir.name,
        'Specimen':       specimen_id,
        'Collagen_Status': collagen_status,
        'Voxel_Source':   vox_source,
        'Voxel_Z':        vz,
        'Voxel_Y':        vy,
        'Voxel_X':        vx,
    }

    try:
        actin_mask = load_seg(actin_seg)
        actin_metrics = analyze_actin(actin_mask, vox_xyz)
        for k, v in actin_metrics.items():
            if k not in ('Voxel_Z', 'Voxel_Y', 'Voxel_X'):
                row[f'Actin_{k}'] = v
    except Exception as e:
        print(f"  WARNING actin failed for {cell_dir.name}: {e}")

    try:
        mito_mask = load_seg(mito_seg)
        mito_metrics = analyze_mito(mito_mask, vox_xyz)
        for k, v in mito_metrics.items():
            if k not in ('Voxel_Z', 'Voxel_Y', 'Voxel_X'):
                row[f'Mito_{k}'] = v
    except Exception as e:
        print(f"  WARNING mito failed for {cell_dir.name}: {e}")

    try:
        nuc_mask = load_seg(nucleus_seg)
        nuc_metrics = analyze_nucleus(nuc_mask, vox_xyz)
        for k, v in nuc_metrics.items():
            if k not in ('Voxel_Z', 'Voxel_Y', 'Voxel_X'):
                row[f'Nucleus_{k}'] = v
    except Exception as e:
        print(f"  WARNING nucleus failed for {cell_dir.name}: {e}")

    return row


def main():
    parser = argparse.ArgumentParser(description='Extract features for 3-group analysis.')
    parser.add_argument('--dry_run', action='store_true',
                        help='List cells without running extraction')
    parser.add_argument('--specimen', type=str, default=None,
                        help='Process only this specimen (e.g. 02Asc-0017)')
    args = parser.parse_args()

    print("=" * 65)
    print("Feature Extraction — 3-Group Dataset")
    print(f"Data root: {DATA_ROOT}")
    print("=" * 65)

    all_rows = []
    n_total = 0
    n_ok = 0
    n_skip = 0

    # Iterate over all specimen folders
    spec_dirs = sorted(DATA_ROOT.iterdir())

    for spec_dir in spec_dirs:
        if not spec_dir.is_dir():
            continue

        spec_id = spec_dir.name
        spec_key = spec_id.lower()

        # Filter to specific specimen if requested
        if args.specimen and spec_id.lower() != args.specimen.lower():
            continue

        if spec_key not in GROUP_MAP:
            print(f"\n[SKIP] {spec_id} — not in GROUP_MAP (unknown group)")
            continue

        group, age, sex = GROUP_MAP[spec_key]
        print(f"\n>>> {spec_id}  [{group}]  age={age}  sex={sex}")

        # Iterate +Collagen and -Collagen
        for cond_dir in sorted(spec_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            cond_name = cond_dir.name
            if '+' in cond_name or 'Collagen' in cond_name.lower():
                collagen = 'Collagen'
            elif '-' in cond_name:
                collagen = 'NoCollagen'
            else:
                collagen = cond_name

            # Iterate cells
            for cell_dir in sorted(cond_dir.iterdir()):
                if not cell_dir.is_dir():
                    continue
                n_total += 1
                row = process_cell(cell_dir, spec_id, collagen,
                                   dry_run=args.dry_run)
                if row is not None and not row.get('DRY_RUN'):
                    row['Group'] = group
                    row['Age']   = age
                    row['Sex']   = sex
                    all_rows.append(row)
                    n_ok += 1
                    print(f"  OK  [{collagen:12s}] {cell_dir.name}  vox={row.get('Voxel_Source','?')}")
                elif row and row.get('DRY_RUN'):
                    n_ok += 1
                else:
                    n_skip += 1

    print(f"\n{'='*65}")
    print(f"Processed: {n_ok}/{n_total}  |  Skipped: {n_skip}")

    if args.dry_run:
        print("[DRY RUN] No CSV written.")
        return

    if not all_rows:
        print("ERROR: No cells processed.")
        return

    master = pd.DataFrame(all_rows)

    # Reorder: metadata columns first
    meta_cols = ['CellName', 'Specimen', 'Group', 'Age', 'Sex',
                 'Collagen_Status', 'Voxel_Source', 'Voxel_Z', 'Voxel_Y', 'Voxel_X']
    feat_cols = [c for c in master.columns if c not in meta_cols]
    master = master[meta_cols + feat_cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_CSV, index=False)

    print(f"\nSaved → {OUT_CSV}")
    print(f"  {len(master)} cells  |  {len(master.columns)} columns")
    print()
    print("Group summary:")
    print(master.groupby(['Group', 'Collagen_Status']).size().unstack(fill_value=0))
    print()
    print("Voxel source summary:")
    print(master['Voxel_Source'].value_counts())


if __name__ == '__main__':
    main()
