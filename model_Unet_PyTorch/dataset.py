# dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import nrrd
from scipy.ndimage import zoom


def _load_volume_nii(path):
    img = nib.load(path)
    vol = img.get_fdata().astype(np.float32)
    return vol  # (D,H,W) or (Z,Y,X), we’ll treat it as (D,H,W)


def _load_label_nrrd(path):
    data, _ = nrrd.read(path)
    return data.astype(np.uint8)  # (D,H,W)


def _resize_3d(vol, target_shape):
    """
    Resize a 3D volume to target_shape using nearest neighbour (for label)
    or linear interpolation (for image).
    """
    dz = target_shape[0] / vol.shape[0]
    dy = target_shape[1] / vol.shape[1]
    dx = target_shape[2] / vol.shape[2]
    return zoom(vol, (dz, dy, dx), order=1)  # order=1: linear


class NucleusDataset(Dataset):
    def __init__(self,
                 base_dir,
                 raw_dir="raw_ch1_nucleus",
                 label_dir="label_nucleus",
                 raw_ext=".nii",
                 label_ext=".nrrd",
                 target_shape=(64, 64, 64)):
        super().__init__()
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, raw_dir)
        self.label_dir = os.path.join(base_dir, label_dir)
        self.raw_ext = raw_ext
        self.label_ext = label_ext
        self.target_shape = target_shape

        # discover raw files
        pattern = os.path.join(self.raw_dir, f"*{self.raw_ext}")
        self.raw_paths = sorted(glob.glob(pattern))

        self.pairs = []
        for raw_path in self.raw_paths:
            fname = os.path.basename(raw_path)
            # Example:
            # RAW:   01ASC-0180_nonAneurysm_coll_nucleus_cell1.nii
            # LABEL: 01ASC-0180_nonAneurysm_coll_nucleusLabel_cell1.nrrd
            label_name = fname.replace("_nucleus_", "_nucleusLabel_").replace(self.raw_ext, self.label_ext)
            label_path = os.path.join(self.label_dir, label_name)
            if os.path.exists(label_path):
                self.pairs.append((raw_path, label_path))
            else:
                print(f"Warning: Label not found for {fname} at {label_path}")

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No paired volumes found in {self.raw_dir} and {self.label_dir}"
            )
        print(f"✅ Found {len(self.pairs)} paired RAW/LABEL volumes.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        raw_path, label_path = self.pairs[idx]
        vol = _load_volume_nii(raw_path)       # float32
        lab = _load_label_nrrd(label_path)     # uint8

        # normalize image [0,1]
        if vol.max() > vol.min():
            vol = (vol - vol.min()) / (vol.max() - vol.min())
        else:
            vol = np.zeros_like(vol, dtype=np.float32)

        # resize to target_shape for both
        if vol.shape != self.target_shape:
            vol = _resize_3d(vol, self.target_shape)
        if lab.shape != self.target_shape:
            lab = _resize_3d(lab, self.target_shape)
            lab = (lab > 0.5).astype(np.uint8)  # keep binary

        # add channel dimension (C=1)
        vol = vol[np.newaxis, ...]   # (1,D,H,W)
        lab = lab[np.newaxis, ...]   # (1,D,H,W)

        vol_t = torch.from_numpy(vol).float()
        lab_t = torch.from_numpy(lab).float()  # for Dice/BCE loss

        return vol_t, lab_t, os.path.basename(raw_path)


def get_train_val_loaders(base_dir,
                          raw_dir="raw_ch1_nucleus",
                          label_dir="label_nucleus",
                          raw_ext=".nii",
                          label_ext=".nrrd",
                          target_shape=(64, 64, 64),
                          batch_size=2,
                          val_split=0.15):

    dataset = NucleusDataset(
        base_dir=base_dir,
        raw_dir=raw_dir,
        label_dir=label_dir,
        raw_ext=raw_ext,
        label_ext=label_ext,
        target_shape=target_shape,
    )

    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, n_train, n_val