from pathlib import Path
import shutil
import re

# ---------------- PATHS ----------------
SRC_DIR = Path("/Volumes/LuisaHD/cell/aneurysm_project/data/segmented_training")
DST_DIR = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/labels_actin")
DST_DIR.mkdir(parents=True, exist_ok=True)

print("Source dir :", SRC_DIR)
print("Output dir :", DST_DIR)

# ------------ SUBJECT GROUPS (from your Excel) ------------
NON_ANEURYSMAL = {
    "01Asc-180",
    "01Asc-0222",
    "01Asc-0230",
    "01C-0083",
    "01C-0096",
    "01C-0097",
    "01C-0113",
    "01C-0117",
    "01C-0202",
}

ANEURYSMAL = {
    "03Asc-24",
    "03Asc-43",
    "03Rt-45",
    "03Asc-46",
    "03Asc-47",
    "03Asc-51",
    "03Asc-54",
    "03Asc-55",
    "03Asc-56",
    "03Asc-57",
}

def get_status(subject: str) -> str:
    """Return 'aneurysm' / 'nonAneurysm' from subject name."""
    s = subject.strip()
    if s in NON_ANEURYSMAL:
        return "nonAneurysm"
    if s in ANEURYSMAL:
        return "aneurysm"
    print(f"⚠️  Unknown status for subject '{subject}', defaulting to nonAneurysm")
    return "nonAneurysm"

def get_coll_flag(coll_folder: str) -> str:
    """Return 'coll' or 'noColl' from folder name like '+ coll' / 'no coll'."""
    c = coll_folder.lower().replace(" ", "")
    if "nocoll" in c:
        return "noColl"
    if "coll" in c:
        return "coll"
    print(f"⚠️  Could not interpret coll/noColl from folder '{coll_folder}', using 'coll'")
    return "coll"

# pattern to grab cell number from folder name (…cell1, …cell10, etc)
CELL_RE = re.compile(r"cell(\d+)", re.IGNORECASE)

# ------------- FIND ALL ACTIN SEGMENTATIONS -------------
# We take *all* .nrrd files and keep only those whose name contains
# both "segmentation" and "actin" in ANY order:
all_nrrd = list(SRC_DIR.rglob("*.nrrd"))
seg_files = [
    f for f in all_nrrd
    if "segmentation" in f.name.lower() and "actin" in f.name.lower()
]

print(f"\nFound {len(seg_files)} actin segmentation .nrrd files\n")

exported = 0
skipped = 0

for seg in seg_files:
    try:
        # segmented_training/<subject>/<coll_folder>/<cell_folder>/Segmentation*.nrrd
        cell_folder = seg.parent.name           # e.g. "01ASC-0180 +coll ... cell1"
        coll_folder = seg.parents[1].name       # "+ coll" or "no coll"
        subject_folder = seg.parents[2].name    # "01Asc-180", "03Asc-24", ...
    except IndexError:
        print(f"❌ Unexpected folder depth for: {seg}")
        skipped += 1
        continue

    subject = subject_folder
    coll_flag = get_coll_flag(coll_folder)

    m = CELL_RE.search(cell_folder)
    if not m:
        print(f"❌ Could not find 'cellX' in folder name: {cell_folder}")
        skipped += 1
        continue
    cell_idx = int(m.group(1))

    status = get_status(subject)

    # keep original extension (.seg.nrrd / .actin.nrrd / whatever) but standardized name
    dst_ext = seg.suffix  # ".nrrd"
    new_name = f"{subject}_{status}_{coll_flag}_actin_cell{cell_idx}{dst_ext}"
    dst = DST_DIR / new_name

    shutil.copy2(seg, dst)
    print(f"✅ {seg.relative_to(SRC_DIR)}  →  {new_name}")
    exported += 1

print("\n==============================")
print(f"✅ Exported : {exported}")
print(f"❌ Skipped  : {skipped}")
print("==============================")