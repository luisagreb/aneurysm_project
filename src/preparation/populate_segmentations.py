#!/usr/bin/env python3
"""
populate_segmentations.py
=========================
Populates organized_data/<subject>/<condition>/<cell>/segmentation/ with
nnUNet segmentation files for all TAV subjects (training labels + inference results).

Sources:
  Training labels   : data/nnUNet/nnUNet_raw/Dataset00X/labelsTr/
  Inference results : experiments/V2/inference_results/Dataset00X/
  Manual segs       : data/raw/manually_segmented_actin/
                      data/raw/manually_segmented_mitochondria/
  Mapping CSVs      : data/nnUNet/nnUNet_raw/{Actin,Mito}_mapping.csv
                      data/nnUNet/nnUNet_raw/Dataset00X/test_mapping.csv
                      data/nnUNet/nnUNet_raw/Dataset003_Nucleus/Nucleus_mapping.csv
"""

import csv
import shutil
import numpy as np
import nrrd
import nibabel as nib
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ORGANIZED = PROJECT / "data" / "organized_data"

# ── Source directories ──────────────────────────────────────────────────────
LABELS_TR = {
    "actin":  PROJECT / "data/nnUNet/nnUNet_raw/Dataset001_Actin/labelsTr",
    "mito":   PROJECT / "data/nnUNet/nnUNet_raw/Dataset002_Mito/labelsTr",
    "nucleus":PROJECT / "data/nnUNet/nnUNet_raw/Dataset003_Nucleus/labelsTr",
}
INFERENCE = {
    "actin":  PROJECT / "experiments/V2/inference_results/Dataset001_Actin",
    "mito":   PROJECT / "experiments/V2/inference_results/Dataset002_Mito",
    "nucleus":PROJECT / "experiments/V2/inference_results/Dataset003_Nucleus",
}

# ── Mapping CSVs ────────────────────────────────────────────────────────────
TRAIN_MAPS = {
    "actin":  PROJECT / "data/nnUNet/nnUNet_raw/Actin_mapping.csv",
    "mito":   PROJECT / "data/nnUNet/nnUNet_raw/Mito_mapping.csv",
    "nucleus":PROJECT / "data/nnUNet/nnUNet_raw/Dataset003_Nucleus/Nucleus_mapping.csv",
}
TEST_MAPS = {
    "actin":  PROJECT / "data/nnUNet/nnUNet_raw/Dataset001_Actin/test_mapping.csv",
    "mito":   PROJECT / "data/nnUNet/nnUNet_raw/Dataset002_Mito/test_mapping.csv",
    "nucleus":PROJECT / "data/nnUNet/nnUNet_raw/Dataset003_Nucleus/test_mapping.csv",
}

MANUAL_DIRS = {
    "actin": PROJECT / "data/raw/manually_segmented_actin",
    "mito":  PROJECT / "data/raw/manually_segmented_mitochondria",
}

SEG_NAMES = {
    "actin":   "actin_seg.nii.gz",
    "mito":    "mitochondria_seg.nii.gz",
    "nucleus": "nucleus_seg.nii.gz",
}


def build_cell_index():
    """Build a dict: normalised_cell_name -> segmentation/ Path."""
    index = {}
    for subj in ORGANIZED.iterdir():
        for cond in subj.iterdir():
            if not cond.is_dir():
                continue
            for cell in cond.iterdir():
                if not cell.is_dir():
                    continue
                key = cell.name.strip().lower()
                seg_dir = cell / "segmentation"
                seg_dir.mkdir(exist_ok=True)
                index[key] = seg_dir
    return index


def copy_seg(src: Path, dst_dir: Path, dst_name: str, dry_run=False):
    if not src.exists():
        return False
    dst = dst_dir / dst_name
    if dst.exists():
        return None  # already present
    if not dry_run:
        shutil.copy2(src, dst)
    return True


def process_training(cell_index, structure, dry_run=False):
    """Copy training labels (manual annotations) into organized_data."""
    mapping_file = TRAIN_MAPS[structure]
    labels_dir   = LABELS_TR[structure]
    seg_name     = SEG_NAMES[structure]

    # Detect column name for cell name (differs between Actin/Mito vs Nucleus)
    with open(mapping_file) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        cell_col = "OriginalCellName" if "OriginalCellName" in fieldnames else "OriginalName"

        copied = skipped = missing_cell = missing_src = 0
        for row in reader:
            case_id   = row["CaseID"].strip()
            cell_name = row[cell_col].strip()
            key       = cell_name.lower()

            seg_dir = cell_index.get(key)
            if seg_dir is None:
                print(f"  [WARN] Cell not found in organized_data: {cell_name!r}")
                missing_cell += 1
                continue

            src = labels_dir / f"{case_id}.nii.gz"
            result = copy_seg(src, seg_dir, seg_name, dry_run)
            if result is True:
                copied += 1
            elif result is None:
                skipped += 1
            else:
                print(f"  [WARN] Source not found: {src}")
                missing_src += 1

    return copied, skipped, missing_cell, missing_src


def process_inference(cell_index, structure, dry_run=False):
    """Copy nnUNet inference results into organized_data."""
    mapping_file = TEST_MAPS[structure]
    inf_dir      = INFERENCE[structure]
    seg_name     = SEG_NAMES[structure]

    with open(mapping_file) as f:
        reader = csv.DictReader(f)
        cell_col = "OriginalName"

        copied = skipped = missing_cell = missing_src = 0
        for row in reader:
            test_id   = row["TestID"].strip()
            cell_name = row[cell_col].strip()
            key       = cell_name.lower()

            seg_dir = cell_index.get(key)
            if seg_dir is None:
                print(f"  [WARN] Cell not found in organized_data: {cell_name!r}")
                missing_cell += 1
                continue

            src = inf_dir / f"{test_id}.nii.gz"
            result = copy_seg(src, seg_dir, seg_name, dry_run)
            if result is True:
                copied += 1
            elif result is None:
                skipped += 1
            else:
                print(f"  [WARN] Source not found: {src}")
                missing_src += 1

    return copied, skipped, missing_cell, missing_src


def nrrd_to_nifti(src: Path, dst: Path, dry_run=False):
    """Convert a binary .nrrd mask to .nii.gz (rescales 0-255 → 0-1)."""
    if dry_run:
        return True
    data, header = nrrd.read(str(src))
    data = (data > 0).astype(np.uint8)
    # Build an identity affine (physical spacing is preserved in the header but
    # not critical for downstream morphometric tools which read voxel spacing separately)
    affine = np.eye(4)
    spacings = header.get("spacings", header.get("space directions", None))
    if spacings is not None:
        try:
            sp = np.array(spacings, dtype=float).flatten()
            if len(sp) >= 3:
                affine[0, 0] = sp[0]
                affine[1, 1] = sp[1] if len(sp) > 1 else sp[0]
                affine[2, 2] = sp[2] if len(sp) > 2 else sp[0]
        except Exception:
            pass
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(dst))
    return True


def process_manual(cell_index, structure, dry_run=False):
    """Convert and copy manually segmented .nrrd files not already in labelsTr."""
    if structure not in MANUAL_DIRS:
        return 0, 0, 0, 0

    manual_dir = MANUAL_DIRS[structure]
    seg_name   = SEG_NAMES[structure]

    # Build set of cell names already covered by training labels
    mapping_file = TRAIN_MAPS[structure]
    cell_col = "OriginalCellName" if structure != "nucleus" else "OriginalName"
    covered = set()
    with open(mapping_file) as f:
        for row in csv.DictReader(f):
            covered.add(row[cell_col].strip().lower())

    copied = skipped = missing_cell = already_covered = 0
    for nrrd_file in sorted(manual_dir.glob("*.nrrd")):
        # Strip trailing .seg from stem if present (e.g. "cell1.seg.nrrd")
        stem = nrrd_file.stem
        if stem.endswith(".seg"):
            stem = stem[:-4]
        key = stem.strip().lower()

        if key in covered:
            already_covered += 1
            continue

        seg_dir = cell_index.get(key)
        if seg_dir is None:
            print(f"  [WARN] No matching cell in organized_data: {stem!r}")
            missing_cell += 1
            continue

        dst = seg_dir / seg_name
        if dst.exists():
            skipped += 1
            continue

        if not dry_run:
            nrrd_to_nifti(nrrd_file, dst)
        copied += 1

    return copied, skipped, already_covered, missing_cell


def main(dry_run=False):
    print(f"Building cell index from {ORGANIZED} ...")
    cell_index = build_cell_index()
    print(f"  Found {len(cell_index)} cells.\n")

    total_copied = total_skipped = total_warn = 0

    for structure in ("actin", "mito", "nucleus"):
        print(f"=== {structure.upper()} ===")

        c, s, mc, ms = process_training(cell_index, structure, dry_run)
        print(f"  Training  : {c} copied, {s} already present, {mc} cells not found, {ms} src missing")
        total_copied  += c
        total_skipped += s
        total_warn    += mc + ms

        c, s, mc, ms = process_inference(cell_index, structure, dry_run)
        print(f"  Inference : {c} copied, {s} already present, {mc} cells not found, {ms} src missing")
        total_copied  += c
        total_skipped += s
        total_warn    += mc + ms

        c, s, cov, mc = process_manual(cell_index, structure, dry_run)
        print(f"  Manual    : {c} converted+copied, {s} already present, {cov} already in training, {mc} cells not found")
        total_copied  += c
        total_skipped += s
        total_warn    += mc

    print(f"\nDone. Total copied: {total_copied} | Already present: {total_skipped} | Warnings: {total_warn}")


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    if dry:
        print("DRY RUN — no files will be copied.\n")
    main(dry_run=dry)
