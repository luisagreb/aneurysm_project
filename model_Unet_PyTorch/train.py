# train_unet3d.py
import os
import sys
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import UNet3D
from dataset import get_train_val_loaders

# ---- config ----
BASE_DATA_DIRECTORY = r"C:\Users\Luisa\Documents\aneurysm_project\Data"
RAW_CHANNEL_DIR = "raw_ch1_nucleus"
LABEL_MASK_DIR = "label_nucleus"
RAW_EXTENSION = ".nii"    # adjust if .nii.gz
LABEL_EXTENSION = ".nrrd"

IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH = 64, 64, 64
N_CHANNELS = 1
N_CLASSES = 1
BATCH_SIZE = 2           # start small; can increase if memory allows
EPOCHS = 30
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
MODEL_SAVE_PATH = "3d_unet_nucleus_seg_pytorch.pth"
LOG_FILE = "training_log_pytorch.csv"


# ---- Dice metric & loss ----
def dice_score(y_pred, y_true, smooth=1e-7):
    """
    Dice coefficient (for metric).
    y_pred, y_true: (N,1,D,H,W), y_pred is sigmoid output [0,1].
    """
    y_pred_bin = (y_pred > 0.5).float()
    y_pred_f = y_pred_bin.view(y_pred_bin.size(0), -1)
    y_true_f = y_true.view(y_true.size(0), -1)

    intersection = (y_pred_f * y_true_f).sum(dim=1)
    union = y_pred_f.sum(dim=1) + y_true_f.sum(dim=1)

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # y_pred is logits; apply sigmoid inside loss
        y_pred = torch.sigmoid(y_pred)
        y_pred_f = y_pred.view(y_pred.size(0), -1)
        y_true_f = y_true.view(y_true.size(0), -1)

        intersection = (y_pred_f * y_true_f).sum(dim=1)
        union = y_pred_f.sum(dim=1) + y_true_f.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for images, masks, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)    # (N,1,D,H,W)
        masks = masks.to(device)      # (N,1,D,H,W)

        optimizer.zero_grad()
        outputs = model(images)       # logits
        loss = criterion(outputs, masks)
        dice = dice_score(torch.sigmoid(outputs), masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_dice += dice.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)
    return epoch_loss, epoch_dice


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = dice_score(torch.sigmoid(outputs), masks)

            running_loss += loss.item() * images.size(0)
            running_dice += dice.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)
    return epoch_loss, epoch_dice


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    try:
        train_loader, val_loader, n_train, n_val = get_train_val_loaders(
            base_dir=BASE_DATA_DIRECTORY,
            raw_dir=RAW_CHANNEL_DIR,
            label_dir=LABEL_MASK_DIR,
            raw_ext=RAW_EXTENSION,
            label_ext=LABEL_EXTENSION,
            target_shape=(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            val_split=VAL_SPLIT,
        )
    except FileNotFoundError as e:
        print("FATAL:", e)
        sys.exit(1)

    print(f"Training set size: {n_train}")
    print(f"Validation set size: {n_val}")

    model = UNet3D(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    # logging
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "TrainLoss", "TrainDice", "ValLoss", "ValDice", "TimeSec"])

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - start

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"  Val   loss: {val_loss:.4f} | Val   Dice: {val_dice:.4f} | Time: {elapsed:.1f}s")

        # save best
        if val_loss < best_val_loss:
            print(f"  ✅ Val loss improved {best_val_loss:.4f} -> {val_loss:.4f}, saving {MODEL_SAVE_PATH}")
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # append to log
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_dice, val_loss, val_dice, elapsed])

    print("\nTraining complete.")
    print("Best model weights saved to", MODEL_SAVE_PATH)