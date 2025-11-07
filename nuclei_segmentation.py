#!/usr/bin/env python3
"""
Batch nucleus segmentation from DAPI (channel 1) NIfTI z-stacks.

Inputs
------
- Root data folder with subject/cell subfolders that contain channel files.
  We look for ch1 patterns like: "*_ch1.nii.gz" or "*_ch1.nii", or
  a single "*allchannels.nii.gz" and we take channel index 0.

Outputs
-------
For each cell, in a mirrored folder under OUTPUT_ROOT:
- nucleus_mask.nii.gz      (uint8, {0,1})
- nucleus_mask.tif         (uint8 stack)
- nucleus_mask.nrrd        (optional, requires 'pynrrd')
- nucleus_surface.stl      (optional mesh; requires 'trimesh' or 'numpy-stl')

Viewers
-------
- 3D Slicer: open the .nii.gz or .nrrd, enable Volume Rendering or Segment Editor
- Fiji: open .tif stack; or .nii.gz with NIfTI plugin
- ParaView: open .nii.gz (ITK readers) or .stl mesh

Dependencies
------------
pip install numpy nibabel scikit-image tifffile
# optional:
pip install pynrrd trimesh  # (or: pip install numpy-stl)
"""

from __future__ import annotations
import os
from pathlib import Path
import re
import numpy as np
import nibabel as nib
from tifffile import imsave
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, remove_small_holes, ball
from skimage.measure import label, marching_cubes
from skimage.segmentation import clear_border

try:
    import nrrd  # type: ignore
    HAS_NRRD = True
except Exception:
    HAS_NRRD = False

try:
    import trimesh  # type: ignore
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False


# ---------- CONFIG ----------

# Change this to your real root (you gave an example earlier):
DATA_ROOT = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/segmented_training")

# Where to write results (mirrors the input tree):
OUTPUT_ROOT = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/converted")

# Patterns we consider as channel-1 files
CH1_PATTERNS = [r"_ch1\.nii(\.gz)?$", r"_ch01\.nii(\.gz)?$", r"ch1\.nii(\.gz)?$"]

# If only an "allchannels" file exists, use this index for DAPI (0-based)
ALLCHANNELS_NAME = "allchannels"
ALLCHANNELS_CH1_INDEX = 0

# Segmentation params (tweak here if needed)
SMALL_OBJECT_VOXELS = 500    # remove specks smaller than this (3D voxels)
SMALL_HOLE_VOXELS  = 2000    # fill holes smaller than this (3D voxels)
GAUSS_SIGMA        = 1.0     # light denoise; set 0 to skip
KEEP_LARGEST       = True    # nuclei per cell: keep largest CC
CLEAR_BORDER       = True    # drop components touching volume border


# ---------- HELPERS ----------

def find_ch1_file(folder: Path) -> tuple[Path, int] | None:
    """
    Return (path, channel_index) to load.
    If ch1 exists directly, channel_index is -1 (use the file as-is).
    Else if allchannels exists, return it with channel_index = ALLCHANNELS_CH1_INDEX.
    """
    # direct channel-1
    for f in folder.glob("*.nii*"):
        name = f.name.lower()
        if any(re.search(p, name) for p in CH1_PATTERNS):
            return (f, -1)

    # allchannels
    for f in folder.glob("*.nii*"):
        name = f.name.lower()
        if ALLCHANNELS_NAME in name:
            return (f, ALLCHANNELS_CH1_INDEX)

    return None


def load_nifti(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)  # float32 for filtering
    return arr, img


def slice_or_volume_threshold(volume: np.ndarray) -> np.ndarray:
    """
    Robust Otsu: per-slice along Z to accommodate intensity drift.
    Return a boolean mask.
    volume: (Z, Y, X) or (Y, X, Z)? We’ll normalize to (Z, Y, X) below.
    """
    if volume.ndim != 3:
        raise ValueError("Expected 3D volume for segmentation.")
    Z, Y, X = volume.shape
    mask = np.zeros_like(volume, dtype=bool)
    for z in range(Z):
        plane = volume[z]
        # Handle empty planes
        if np.allclose(plane, 0):
            continue
        t = threshold_otsu(plane)
        mask[z] = plane > t
    return mask


def postprocess_3d(mask: np.ndarray) -> np.ndarray:
    """3D morphological cleanup."""
    if CLEAR_BORDER:
        mask = clear_border(mask)

    if SMALL_OBJECT_VOXELS > 0:
        mask = remove_small_objects(mask, SMALL_OBJECT_VOXELS)

    if SMALL_HOLE_VOXELS > 0:
        mask = remove_small_holes(mask, SMALL_HOLE_VOXELS)

    if KEEP_LARGEST:
        lab = label(mask, connectivity=1)
        if lab.max() > 0:
            # keep largest non-zero label
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            k = counts.argmax()
            mask = (lab == k)

    return mask


def ensure_zyx(data: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Best effort to reorder axes to (Z, Y, X).
    For NIfTI, nibabel usually gives (X, Y, Z) — we'll detect the smallest dim as Z if unclear.
    We return (data_zyx, note).
    """
    note = ""
    if data.ndim == 3:
        # Heuristic: Z is often the smallest dimension for microscopy stacks
        order = np.argsort(data.shape)  # ascending
        # Put the smallest last as Z? We'll try common cases explicitly:
        # Common NIfTI orientation from microscopy: (Z, Y, X) or (X, Y, Z).
        # If last axis is much smaller -> assume Z last already.
        if data.shape[2] < min(data.shape[0], data.shape[1]):
            # probably (Y, X, Z) -> move last to first: (Z, Y, X)
            data = np.moveaxis(data, 2, 0)
            note = "Reordered axes to (Z,Y,X) from (*,*,Z)."
        elif data.shape[0] < min(data.shape[1], data.shape[2]):
            # probably (Z, Y, X) already
            note = "Assumed (Z,Y,X)."
        else:
            # probably (X, Y, Z) -> swap to (Z, Y, X)
            data = np.moveaxis(data, 2, 0)  # (X,Y,Z) -> (X,Y) is unchanged; Z goes to front -> (Z,X,Y)
            data = np.swapaxes(data, 1, 2)  # (Z,X,Y) -> (Z,Y,X)
            note = "Reordered axes to (Z,Y,X) from (X,Y,Z)."
        return data, note

    elif data.ndim == 4:
        # assume last dim is channels -> handled before calling this
        raise ValueError("Call ensure_zyx on 3D arrays only (channels must be extracted first).")
    else:
        raise ValueError(f"Unexpected ndim={data.ndim}")


def save_mask_variants(mask_zyx: np.ndarray, ref_img: nib.Nifti1Image, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine voxel spacing from reference header if present
    zooms = ref_img.header.get_zooms()
    # nibabel returns up to 4 dims; we care about the first 3
    spacing = tuple(float(z) for z in zooms[:3]) if len(zooms) >= 3 else (1.0, 1.0, 1.0)

    # NIfTI (Z,Y,X) -> we’ll store as (Y,X,Z) data order inside NIfTI by swapping axes back to RAS-like
    # But the safest is to keep affine and use same orientation as ref
    # Build a NIfTI with the same affine as reference
    mask_uint8 = (mask_zyx.astype(np.uint8))
    mask_nifti = nib.Nifti1Image(mask_uint8, affine=ref_img.affine, header=ref_img.header.copy())
    mask_nifti.set_data_dtype(np.uint8)
    nib.save(mask_nifti, str(out_dir / "nucleus_mask.nii.gz"))

    # TIFF stack (Fiji friendly) — expect (Z, Y, X)
    imsave(str(out_dir / "nucleus_mask.tif"), mask_zyx.astype(np.uint8), imagej=True)

    # Optional NRRD (3D Slicer native)
    if HAS_NRRD:
        hdr = {
            'space': 'left-posterior-superior',
            'kinds': ['domain', 'domain', 'domain'],
            'space directions': np.diag(spacing + (0.0,))[:3, :3],
            'space origin': (0.0, 0.0, 0.0),
        }
        nrrd.write(str(out_dir / "nucleus_mask.nrrd"), mask_zyx.astype(np.uint8), header=hdr)


def maybe_export_surface(mask_zyx: np.ndarray, ref_img: nib.Nifti1Image, out_dir: Path, level: float = 0.5):
    if not HAS_TRIMESH:
        return
    out_path = out_dir / "nucleus_surface.stl"
    # marching cubes expects (Z,Y,X) with spacing to scale vertices
    zooms = ref_img.header.get_zooms()
    spacing = tuple(float(z) for z in zooms[:3]) if len(zooms) >= 3 else (1.0, 1.0, 1.0)
    verts, faces, normals, _ = marching_cubes(mask_zyx, level=level, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    mesh.export(out_path)


# ---------- MAIN BATCH ----------

def process_cell_folder(cell_dir: Path):
    found = find_ch1_file(cell_dir)
    if not found:
        return False

    in_path, ch_idx = found
    arr, ref = load_nifti(in_path)

    # select channel if needed
    if arr.ndim == 4 and ch_idx >= 0:
        # assume last dim is channels
        arr = arr[..., ch_idx]

    # reorder to (Z,Y,X)
    vol, _note = ensure_zyx(arr)

    # denoise (light)
    if GAUSS_SIGMA and GAUSS_SIGMA > 0:
        vol = gaussian(vol, sigma=GAUSS_SIGMA, preserve_range=True)

    # threshold per Z-slice, then 3D cleanup
    mask = slice_or_volume_threshold(vol)
    mask = postprocess_3d(mask)

    # where to save (mirror tree from DATA_ROOT into OUTPUT_ROOT)
    rel = cell_dir.relative_to(DATA_ROOT)
    out_dir = OUTPUT_ROOT / rel
    save_mask_variants(mask, ref, out_dir)

    # optional mesh
    maybe_export_surface(mask, ref, out_dir)
    return True


def main():
    n_total = 0
    n_done = 0
    for root, dirs, files in os.walk(DATA_ROOT):
        root_p = Path(root)
        # “cell folder” heuristic: it contains any .nii*
        if any(f.lower().endswith((".nii", ".nii.gz")) for f in files):
            n_total += 1
            ok = process_cell_folder(root_p)
            if ok:
                print(f"[OK] {root_p}")
                n_done += 1
            else:
                print(f"[SKIP] No ch1/allchannels in {root_p}")

    print(f"\nDone: {n_done}/{n_total} cell folders processed.")
    print(f"Outputs in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()