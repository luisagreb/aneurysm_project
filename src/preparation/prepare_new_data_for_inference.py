"""
Prepare New Data for nnU-Net Inference

Copies per-channel NIfTI files from data/processed/new_data_nifti/
into the correct nnU-Net imagesTs directories, using the existing
naming convention (Actin_Test_XXXX_0000.nii.gz etc.) and starting
numbering after the existing cases.

A master mapping CSV is saved to outputs/new_data_inference_mapping.csv
so original filenames can always be recovered from case IDs.

Channel assignment:
    channel_00.nii.gz  →  Dataset003_Nucleus  (Nucleus_Test_XXXX)
    channel_01.nii.gz  →  Dataset001_Actin    (Actin_Test_XXXX)
    channel_02.nii.gz  →  Dataset002_Mito     (Mito_Test_XXXX)

Usage (from project root):
    python3 src/preparation/prepare_new_data_for_inference.py
"""

import shutil
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
NIFTI_ROOT = Path("data/processed/new_data_nifti")   # Output of file_conversion.py
NNUNET_RAW = Path("data/nnUNet/nnUNet_raw")

DATASETS = {
    "Actin":   {"dataset": "Dataset001_Actin",   "channel": "channel_01.nii.gz", "prefix": "Actin_Test"},
    "Mito":    {"dataset": "Dataset002_Mito",    "channel": "channel_02.nii.gz", "prefix": "Mito_Test"},
    "Nucleus": {"dataset": "Dataset003_Nucleus", "channel": "channel_00.nii.gz", "prefix": "Nucleus_Test"},
}

MAPPING_OUT = Path("outputs/new_data_inference_mapping.csv")


def get_next_case_number(images_ts_dir: Path, prefix: str) -> int:
    """Find the highest existing case number and return next available."""
    if not images_ts_dir.exists():
        return 1
    existing = sorted(images_ts_dir.glob(f"{prefix}_*.nii.gz"))
    if not existing:
        return 1
    # .nii.gz has double extension: stem of "Actin_Test_0153_0000.nii.gz" is "Actin_Test_0153_0000.nii"
    last_stem = existing[-1].name.replace(".nii.gz", "").replace("_0000", "")  # e.g. Actin_Test_0153
    last_num = int(last_stem.split("_")[-1])
    return last_num + 1


def find_cells(nifti_root: Path):
    """
    Find all cell folders (those containing channel_00.nii.gz).
    Returns sorted list of cell folder Paths.
    """
    return sorted([
        p.parent
        for p in nifti_root.rglob("channel_00.nii.gz")
    ])


def main():
    print("=" * 70)
    print("Prepare New Data for nnU-Net Inference")
    print(f"Source: {NIFTI_ROOT.resolve()}")
    print("=" * 70)

    cells = find_cells(NIFTI_ROOT)
    if not cells:
        print("No cell folders found — run file_conversion.py first.")
        return

    print(f"\nFound {len(cells)} cell folders\n")

    # Determine starting case numbers for each dataset
    case_starts = {}
    for name, cfg in DATASETS.items():
        ts_dir = NNUNET_RAW / cfg["dataset"] / "imagesTs"
        ts_dir.mkdir(parents=True, exist_ok=True)
        case_starts[name] = get_next_case_number(ts_dir, cfg["prefix"])
        print(f"  {name:8s} → starts at case {case_starts[name]:04d}  ({ts_dir})")

    print()

    # Process each cell
    records = []
    counters = {name: case_starts[name] for name in DATASETS}

    for cell_dir in cells:
        # Human-readable cell name from folder path
        rel          = cell_dir.relative_to(NIFTI_ROOT)   # e.g. 02Asc-0017/+Col/02Asc-0017 +col Cell 1
        subject      = rel.parts[0]                        # e.g. 02Asc-0017
        condition    = rel.parts[1] if len(rel.parts) > 1 else ""   # e.g. +Col
        cell_name    = rel.parts[-1]                       # e.g. 02Asc-0017 +col Cell 1

        print(f"  {subject} | {condition} | {cell_name}")

        row = {
            "cell_name":  cell_name,
            "subject":    subject,
            "condition":  condition,
            "folder":     str(cell_dir),
        }

        for name, cfg in DATASETS.items():
            src = cell_dir / cfg["channel"]
            if not src.exists():
                print(f"    ⚠  Missing {cfg['channel']} — skipping {name}")
                row[f"{name}_case"] = None
                row[f"{name}_output"] = None
                continue

            case_num  = counters[name]
            ts_dir    = NNUNET_RAW / cfg["dataset"] / "imagesTs"
            dest_name = f"{cfg['prefix']}_{case_num:04d}_0000.nii.gz"
            dest      = ts_dir / dest_name

            shutil.copy2(str(src), str(dest))
            print(f"    ✓ {name:8s} → {dest_name}")

            row[f"{name}_case"]   = f"{cfg['prefix']}_{case_num:04d}"
            row[f"{name}_output"] = str(dest)

            counters[name] += 1

        records.append(row)

    # Save master mapping CSV
    MAPPING_OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(MAPPING_OUT, index=False)

    print("\n" + "=" * 70)
    print(f"DONE — {len(cells)} cells prepared for inference")
    print(f"Mapping saved: {MAPPING_OUT.resolve()}")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run nnU-Net inference for each dataset:")
    print("     source src/setup_env.sh")
    print("     python3 src/preparation/run_new_data_inference.sh")
    print("\nCase number ranges:")
    for name, cfg in DATASETS.items():
        ts_dir = NNUNET_RAW / cfg["dataset"] / "imagesTs"
        start  = case_starts[name]
        end    = counters[name] - 1
        print(f"  {name:8s}: {cfg['prefix']}_{start:04d} → {cfg['prefix']}_{end:04d}")


if __name__ == "__main__":
    main()
