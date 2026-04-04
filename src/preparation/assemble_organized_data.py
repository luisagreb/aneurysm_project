#!/usr/bin/env python3
"""
assemble_organized_data.py
==========================
Creates a single organized folder with REAL COPIES of all data, ready
for download.

    data/organized_data/
      {subject}/
        +Collagen/  or  -Collagen/
          {cell_name}/
            raw/
              actin.nii.gz
              mitochondria.nii.gz
              nucleus.nii.gz
            segmentation/
              actin_seg.nii.gz
              mitochondria_seg.nii.gz
              nucleus_seg.nii.gz

Data sources:
  1. NEW DATA  — raw: data/processed/new_data_nifti/
               — seg: data/nnUNet/inference_results/new_data/
               — map: outputs/new_data_inference_mapping.csv
  2. ORIGINAL  — raw: data/raw/nrrd_files/ (channel_00/01/02.nrrd)
               — seg: found via test_mapping.csv in nnUNet dataset dirs
"""

import os
import csv
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_BASE     = PROJECT_ROOT / "data" / "organized_data"

# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def copy_file(src: Path, dst: Path):
    """Copy src to dst. Skip if src doesn't exist."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    shutil.copy2(str(src), str(dst))
    return True


def sanitize(name: str) -> str:
    return re.sub(r'[():]', '', name).strip()

# ──────────────────────────────────────────────────────────────────────────────
# Part 1 — New data (uses new_data_inference_mapping.csv)
# ──────────────────────────────────────────────────────────────────────────────

def process_new_data():
    mapping_csv = PROJECT_ROOT / "outputs" / "new_data_inference_mapping.csv"
    if not mapping_csv.exists():
        print("  new_data_inference_mapping.csv not found — skipping.")
        return

    inf_root = PROJECT_ROOT / "data" / "nnUNet" / "inference_results" / "new_data"

    with open(mapping_csv) as f:
        rows = list(csv.DictReader(f))

    print(f"\n{'='*65}")
    print(f"Processing NEW DATA  ({len(rows)} cells)")
    print(f"{'='*65}")

    ok = missing_seg = 0
    for row in rows:
        cell_name  = row["cell_name"].strip()
        subject    = row["subject"].strip()
        condition  = row["condition"].strip()         # +Col or -Col
        raw_folder = PROJECT_ROOT / row["folder"].strip()

        # Clean condition name
        if "+" in condition:
            cond_folder = "+Collagen"
        else:
            cond_folder = "-Collagen"

        cell_safe = sanitize(cell_name)
        dst_root  = OUT_BASE / sanitize(subject) / cond_folder / cell_safe

        # ── RAW ──
        ch_map = {
            "nucleus.nii.gz":      "channel_00.nii.gz",
            "actin.nii.gz":        "channel_01.nii.gz",
            "mitochondria.nii.gz": "channel_02.nii.gz",
        }
        for dst_name, src_name in ch_map.items():
            copy_file(raw_folder / src_name, dst_root / "raw" / dst_name)

        # ── SEGMENTATION ──
        actin_case = row.get("Actin_case", "").strip()
        mito_case  = row.get("Mito_case",  "").strip()
        nuc_case   = row.get("Nucleus_case","").strip()

        seg_pairs = [
            ("actin_seg.nii.gz",        inf_root / "Actin"   / f"{actin_case}.nii.gz"),
            ("mitochondria_seg.nii.gz", inf_root / "Mito"    / f"{mito_case}.nii.gz"),
            ("nucleus_seg.nii.gz",      inf_root / "Nucleus" / f"{nuc_case}.nii.gz"),
        ]
        for dst_name, src in seg_pairs:
            if not copy_file(src, dst_root / "segmentation" / dst_name):
                missing_seg += 1

        ok += 1
        if ok % 20 == 0:
            print(f"  ... {ok}/{len(rows)} cells copied")

    print(f"  Done — {ok} cells copied ({missing_seg} segmentation files not found)")

# ──────────────────────────────────────────────────────────────────────────────
# Part 2 — Original data
# ──────────────────────────────────────────────────────────────────────────────

def parse_cell_meta(cell_name: str):
    subject_m = re.match(r'(\d+[A-Za-z]+-\d+)', cell_name)
    subject = subject_m.group(1).upper() if subject_m else "Unknown"

    if "+col" in cell_name.lower() or "+coll" in cell_name.lower():
        condition = "+Collagen"
    elif "-col" in cell_name.lower() or "nocol" in cell_name.lower():
        condition = "-Collagen"
    else:
        condition = "Unknown"

    return subject, condition


def process_original_data():
    raw_root = PROJECT_ROOT / "data" / "raw" / "nrrd_files"
    if not raw_root.exists():
        print(f"\n  Original raw_root not found: {raw_root} — skipping.")
        return

    # Load test_mapping.csv from each dataset to find segmentation results
    def load_test_mapping(dataset_dir: Path) -> dict:
        tm = dataset_dir / "test_mapping.csv"
        if not tm.exists():
            return {}
        mapping = {}
        with open(tm) as f:
            for row in csv.DictReader(f):
                case_id = (row.get("CaseID") or row.get("case_id") or "").strip()
                cell    = (row.get("OriginalCellName") or row.get("cell_name") or "").strip()
                if case_id and cell:
                    mapping[cell] = case_id
        return mapping

    dataset_actin = PROJECT_ROOT / "data" / "nnUNet" / "nnUNet_raw" / "Dataset001_Actin"
    dataset_mito  = PROJECT_ROOT / "data" / "nnUNet" / "nnUNet_raw" / "Dataset002_Mito"
    dataset_nuc   = PROJECT_ROOT / "data" / "nnUNet" / "nnUNet_raw" / "Dataset003_Nucleus"

    actin_test_map = load_test_mapping(dataset_actin)
    mito_test_map  = load_test_mapping(dataset_mito)
    nuc_test_map   = load_test_mapping(dataset_nuc)

    inf_root = PROJECT_ROOT / "data" / "nnUNet" / "inference_results"

    def find_seg_file(case_id: str) -> Path | None:
        """Search all inference_results subdirs for case_id.nii.gz"""
        for p in inf_root.rglob(f"{case_id}.nii.gz"):
            return p
        return None

    cell_folders = [p for p in sorted(raw_root.iterdir())
                    if p.is_dir() and p.name not in ("manually_segmented", "nnUnet_files")]

    print(f"\n{'='*65}")
    print(f"Processing ORIGINAL DATA  ({len(cell_folders)} cell folders)")
    print(f"{'='*65}")

    ok = skip = missing_seg = 0
    for cell_folder in cell_folders:
        cell_name = cell_folder.name
        subject, condition = parse_cell_meta(cell_name)
        cell_safe = sanitize(cell_name)
        dst_root  = OUT_BASE / sanitize(subject) / condition / cell_safe

        # ── RAW ──
        found_any = False
        for dst_name, src_name in [
            ("nucleus.nrrd",      "channel_00.nrrd"),
            ("actin.nrrd",        "channel_01.nrrd"),
            ("mitochondria.nrrd", "channel_02.nrrd"),
        ]:
            src = cell_folder / src_name
            if not src.exists():
                # try .nii.gz
                alt = src_name.replace(".nrrd", ".nii.gz")
                src = cell_folder / alt
                dst_name = dst_name.replace(".nrrd", ".nii.gz")
            if copy_file(src, dst_root / "raw" / dst_name):
                found_any = True

        if not found_any:
            skip += 1
            continue

        # ── SEGMENTATION ──
        for seg_name, test_map in [
            ("actin_seg",        actin_test_map),
            ("mitochondria_seg", mito_test_map),
            ("nucleus_seg",      nuc_test_map),
        ]:
            case_id = test_map.get(cell_name)
            if case_id:
                seg_path = find_seg_file(case_id)
                if seg_path:
                    copy_file(seg_path, dst_root / "segmentation" / f"{seg_name}.nii.gz")
                else:
                    missing_seg += 1
            else:
                missing_seg += 1

        ok += 1
        if ok % 50 == 0:
            print(f"  ... {ok}/{len(cell_folders)} cells copied")

    print(f"  Done — {ok} cells copied, {skip} skipped ({missing_seg} seg files not found)")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Assembling organized_data folder (REAL COPIES)")
    print(f"Output: {OUT_BASE}")
    print("=" * 65)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    process_new_data()
    process_original_data()

    print(f"\n{'='*65}")
    print("DONE — folder ready for download at:")
    print(f"  {OUT_BASE}")
    print("=" * 65)

    # Summary
    subjects = sorted([d.name for d in OUT_BASE.iterdir() if d.is_dir()])
    print(f"\nSubjects ({len(subjects)}): {', '.join(subjects)}")
    total_cells = sum(1 for _ in OUT_BASE.rglob("raw") if _.is_dir())
    print(f"Total cells: {total_cells}")

    # Disk usage
    import subprocess
    result = subprocess.run(["du", "-sh", str(OUT_BASE)], capture_output=True, text=True)
    print(f"Total size: {result.stdout.strip().split()[0]}")


if __name__ == "__main__":
    main()
