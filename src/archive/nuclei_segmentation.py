# debug_nuclei_one.py
from pathlib import Path
import numpy as np
import nibabel as nib
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu, threshold_yen, threshold_triangle, gaussian
from skimage.filters import threshold_sauvola as sauvola
from skimage.morphology import remove_small_objects, remove_small_holes, ball, binary_opening, binary_closing
from skimage.measure import label
from skimage.util import img_as_ubyte
import imageio.v2 as iio
import nrrd

# --- path parameters ---
p = Path("/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/segmented_training/01Asc-180/+coll/01ASC-0180 +coll 60x DMSO48h-Zstack cell1/01ASC-0180 +coll 60x DMSO48h-Zstack cell1_ch1.nii.gz") # First cell
ALLCHANNELS = False         # set True if p is the *allchannels* file
CH_INDEX = 0                # if ALLCHANNELS=True, which channel index is DAPI?

# --- load ---
img = nib.load(str(p))
arr = img.get_fdata().astype(np.float32)

print("Loaded shape:", arr.shape, "dtype:", arr.dtype, "min/max:", float(arr.min()), float(arr.max()))
if arr.ndim == 4:
    if not ALLCHANNELS:
        raise RuntimeError("This file has multiple channels; set ALLCHANNELS=True and CH_INDEX correctly.")
    arr = arr[..., CH_INDEX]
    print("Selected channel", CH_INDEX, "->", arr.shape)

# --- put to (Z,Y,X) if needed (heuristic) ---
# For many microscopy NIfTIs, (Z,Y,X) or (X,Y,Z) are common. Make sure Z is the first axis.
if arr.ndim != 3:
    raise RuntimeError("Expected 3D after channel selection.")
# If last axis is much smaller than the others, assume data=(Y,X,Z) -> move Z to front
if arr.shape[2] < min(arr.shape[0], arr.shape[1]):
    arr = np.moveaxis(arr, 2, 0)
    print("Reordered: moved last axis to Z ->", arr.shape)
# Else if looks like (X,Y,Z), swap to (Z,Y,X)
elif arr.shape[0] > arr.shape[2] and arr.shape[1] > arr.shape[2]:
    arr = np.moveaxis(arr, 2, 0)   # (X,Y,Z)->(Z,X,Y)
    arr = np.swapaxes(arr, 1, 2)   # -> (Z,Y,X)
    print("Reordered from (X,Y,Z) to (Z,Y,X) ->", arr.shape)
else:
    print("Assuming already (Z,Y,X) ->", arr.shape)

# --- robust intensity normalization ---
lo, hi = np.percentile(arr, (1, 99.9))
if hi <= lo:  # pathological cases
    hi = arr.max() if arr.max() > 0 else 1.0
    lo = arr.min()
arr_n = rescale_intensity(arr, in_range=(lo, hi), out_range=(0.0, 1.0))
print(f"Rescale in_range=({lo:.4g}, {hi:.4g}) -> out [0,1]")

# light denoise
arr_s = gaussian(arr_n, sigma=1.0, preserve_range=True)

# --- threshold (robust combo) ---
# 1) global fallbacks
try:
    t_otsu = threshold_otsu(arr_s)
except Exception:
    t_otsu = 0.0
t_yen  = threshold_yen(arr_s)
t_tri  = threshold_triangle(arr_s)

# 2) per-slice Sauvola (adaptive) and combine
Z = arr_s.shape[0]
mask_slices = np.zeros_like(arr_s, dtype=bool)
for z in range(Z):
    sl = arr_s[z]
    # window ~ 31 px is a good start; adjust if your XY pixel size is very fine
    thr = sauvola(sl, window_size=31, k=0.2)
    mask_slices[z] = sl > thr

# 3) combine with a conservative global threshold (pick the mildest of the three)
t_global = min(t for t in [t_otsu, t_yen, t_tri] if np.isfinite(t))
mask_global = arr_s > t_global * 0.9  # slightly relaxed

mask = mask_slices | mask_global

# --- 3D clean (be gentle first) ---
mask = binary_opening(mask, ball(1))
mask = binary_closing(mask, ball(1))
mask = remove_small_holes(mask, area_threshold=500)
mask = remove_small_objects(mask, min_size=800)

# Do NOT drop border or keep_largest yet; we want to *see* what's there first.
lab = label(mask)
print("Mask voxels:", int(mask.sum()), "components:", lab.max())

# --- quick visual QC PNGs (mid-Z) ---
mid = Z // 2
overlay = np.clip(arr_n[mid]*0.7 + mask[mid].astype(np.float32)*0.3, 0, 1)
iio.imwrite("qc_mid_raw.png", img_as_ubyte(arr_n[mid]))
iio.imwrite("qc_mid_mask.png", (mask[mid]*255).astype(np.uint8))
iio.imwrite("qc_mid_overlay.png", img_as_ubyte(overlay))

# --- save NIfTI mask to compare in Slicer ---
mask_uint8 = mask.astype(np.uint8)
nib.save(nib.Nifti1Image(mask_uint8, img.affine, img.header), "nucleus_mask_debug.nii.gz")
print("Wrote: nucleus_mask_debug.nii.gz + qc_mid_*.png")

# --- Save as NRRD ---
zooms = img.header.get_zooms()
if len(zooms) >= 3:
    # NIfTI may have (Z,Y,X) spacing if single-channel
    zsp, ysp, xsp = float(zooms[-3]), float(zooms[-2]), float(zooms[-1])
else:
    zsp = ysp = xsp = 1.0

nrrd_header = {
    "type": "uint8",
    "dimension": 3,
    "space": "left-posterior-superior",
    "space directions": [(xsp, 0.0, 0.0), (0.0, ysp, 0.0), (0.0, 0.0, zsp)],
    "encoding": "gzip",
}

nrrd.write("nucleus_mask_debug.nrrd", mask_uint8, nrrd_header)
print("Wrote: nucleus_mask_debug.nrrd")