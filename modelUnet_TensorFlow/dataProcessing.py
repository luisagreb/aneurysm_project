import os
import glob
import sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


def get_data_paths(base_dir, raw_dir, label_dir, raw_ext, label_ext):
    """
    Pair raw images and label masks based on filename.

    RAW example:
        01ASC-0180_nonAneurysm_coll_nucleus_cell1.nii

    LABEL example:
        01ASC-0180_nonAneurysm_coll_nucleusLabel_cell1.nrrd

    So we:
      - strip `raw_ext` ('.nii')
      - replace 'nucleus_' by 'nucleusLabel_'
    """

    raw_pattern = os.path.join(base_dir, raw_dir, f"*{raw_ext}")
    all_raw_paths = sorted(glob.glob(raw_pattern))

    paired_raw = []
    paired_label = []

    for raw_path in all_raw_paths:
        raw_name = os.path.basename(raw_path)

        if not raw_name.endswith(raw_ext):
            continue

        base_name = raw_name[:-len(raw_ext)]  # e.g. 01ASC-..._nucleus_cell1
        label_base = base_name.replace("nucleus_", "nucleusLabel_")
        label_name = f"{label_base}{label_ext}"
        label_path = os.path.join(base_dir, label_dir, label_name)

        if os.path.exists(label_path):
            paired_raw.append(raw_path)
            paired_label.append(label_path)
        else:
            print(
                f"[get_data_paths] WARNING: no label for RAW '{raw_name}'\n"
                f"    expected: {label_path}",
                file=sys.stderr,
                flush=True,
            )

    return paired_raw, paired_label


def load_and_preprocess_volume(file_path, target_shape, is_mask=False):
    """
    Load a 3D volume with SimpleITK, resize to target_shape (D,H,W),
    and normalize / binarize.

    Returns array of shape (D, H, W, 1) with dtype float32.
    """
    try:
        itk_img = sitk.ReadImage(file_path)
        vol = sitk.GetArrayFromImage(itk_img).astype(np.float32)
        # SimpleITK gives (Z, Y, X); we want (D, H, W) but names don’t matter
        # as long as RAW and MASK are treated the same.
    except Exception as e:
        print(f"[load_and_preprocess_volume] ERROR reading {file_path}: {e}",
              file=sys.stderr)
        return np.zeros(target_shape + (1,), dtype=np.float32)

    current_shape = np.array(vol.shape, dtype=np.float32)  # (D,H,W)
    target_shape = np.array(target_shape, dtype=np.float32)

    zoom_factors = (target_shape / current_shape)
    order = 0 if is_mask else 1  # nearest for masks, linear for images

    try:
        vol_resized = zoom(vol, zoom_factors, order=order)
    except Exception as e:
        print(f"[load_and_preprocess_volume] ERROR resizing {file_path}: {e}",
              file=sys.stderr)
        return np.zeros(target_shape.astype(int).tolist() + [1], dtype=np.float32)

    if is_mask:
        vol_resized = (vol_resized > 0.5).astype(np.float32)
    else:
        vmin = float(vol_resized.min())
        vmax = float(vol_resized.max())
        if vmax > vmin:
            vol_resized = (vol_resized - vmin) / (vmax - vmin)
        else:
            vol_resized = np.zeros_like(vol_resized, dtype=np.float32)

    vol_resized = vol_resized.astype(np.float32)

    # (D,H,W,1)
    return np.expand_dims(vol_resized, axis=-1)


def _augment_pair(raw_vol, label_vol):
    """
    Simple paired augmentation: random flips along axes.
    Both raw_vol and label_vol are (D, H, W, 1).
    """
    # flip depth
    if np.random.rand() < 0.5:
        raw_vol = raw_vol[::-1, :, :, :]
        label_vol = label_vol[::-1, :, :, :]
    # flip height
    if np.random.rand() < 0.5:
        raw_vol = raw_vol[:, ::-1, :, :]
        label_vol = label_vol[:, ::-1, :, :]
    # flip width
    if np.random.rand() < 0.5:
        raw_vol = raw_vol[:, :, ::-1, :]
        label_vol = label_vol[:, :, ::-1, :]
    return raw_vol, label_vol


def prepare_data_generator(raw_paths,
                           label_paths,
                           target_shape,
                           batch_size,
                           num_classes,
                           shuffle=True,
                           augment=False):
    """
    Keras-style generator yielding (X, Y) batches.

    X: (B, D, H, W, 1)
    Y: (B, D, H, W, 1) for binary segmentation.
    """

    assert len(raw_paths) == len(label_paths), "raw / label mismatch"
    data_size = len(raw_paths)
    indices = np.arange(data_size)

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, data_size, batch_size):
            end = min(start + batch_size, data_size)
            batch_idx = indices[start:end]

            bsz = len(batch_idx)
            batch_x = np.zeros((bsz,) + tuple(target_shape) + (1,),
                               dtype=np.float32)
            batch_y = np.zeros((bsz,) + tuple(target_shape) + (1,),
                               dtype=np.float32)

            for i, idx in enumerate(batch_idx):
                raw_vol = load_and_preprocess_volume(
                    raw_paths[idx], target_shape, is_mask=False
                )
                lbl_vol = load_and_preprocess_volume(
                    label_paths[idx], target_shape, is_mask=True
                )

                if augment:
                    raw_vol, lbl_vol = _augment_pair(raw_vol, lbl_vol)

                batch_x[i] = raw_vol
                batch_y[i] = lbl_vol  # binary → single channel

            yield batch_x, batch_y