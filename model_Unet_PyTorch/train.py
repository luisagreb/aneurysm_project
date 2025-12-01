import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm # For progress bars
import csv
import time

# --- Handling imports for module execution ---
# Adjust these imports if model.py and dataset.py are not in the same directory
from model import UNet3D
from dataset import get_train_val_loaders

try:
    from UNet3D import model
    from dataset import (
        get_train_val_loaders,
        _load_and_preprocess_volume,
        prepare_data_generator,
    )
except ImportError:
    from modelUnet.uNet_model import unet_model
    from modelUnet.dataProcessing import (
        get_data_paths,
        load_and_preprocess_volume,
        get_data_paths,
    )

# --- Configuration Variables ---

# >>> WARNING: UPDATE THIS PATH TO YOUR ACTUAL DATA LOCATION! <<<
BASE_DATA_DIRECTORY = "C:\\Users\\Luisa\\Documents\\aneurysm_project\\Data"
RAW_CHANNEL_DIR = "raw_ch1_nucleus"
LABEL_MASK_DIR = "label_nucleus"
RAW_EXTENSION = ".nii.gz"
LABEL_EXTENSION = ".nrrd"

# Model and Training Parameters
IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH = 64, 64, 64 # Target resolution for all volumes
N_CHANNELS = 1
N_CLASSES = 1                                # Binary segmentation
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
MODEL_SAVE_PATH = "3d_unet_nucleus_seg_pytorch.pth"
LOG_FILE = "training_log_pytorch.csv"


# --- Custom Loss and Metrics (PyTorch) ---

def dice_score(y_pred, y_true, smooth=1e-7):
    """
    Dice coefficient metric (for evaluation).
    y_pred and y_true are (N, C, D, H, W) tensors.
    """
    # Threshold the prediction (sigmoid output) to get a binary mask for metric calculation
    y_pred = (y_pred > 0.5).float()
    
    # Flatten the volumes except the batch dimension
    y_pred_f = y_pred.view(y_pred.shape[0], -1)
    y_true_f = y_true.view(y_true.shape[0], -1)
    
    intersection = torch.sum(y_pred_f * y_true_f, dim=1)
    union = torch.sum(y_pred_f, dim=1) + torch.sum(y_true_f, dim=1)
    
    # Calculate Dice score for each sample in the batch
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.mean() # Return mean Dice score across the batch

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation (1 - Dice Score).
    """
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # The input tensors should already be the sigmoid output [0, 1]
        
        # Flatten the volumes except the batch dimension
        y_pred_f = y_pred.view(y_pred.shape[0], -1)
        y_true_f = y_true.view(y_true.shape[0], -1)

        intersection = torch.sum(y_pred_f * y_true_f, dim=1)
        union = torch.sum(y_pred_f, dim=1) + torch.sum(y_true_f, dim=1)

        dice_coefficient = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Mean loss across the batch
        return torch.mean(1.0 - dice_coefficient)


# --- Training and Validation Step Functions ---

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Performs one epoch of training."""
    model.train() # Set model to training mode
    running_loss = 0.0
    running_dice = 0.0
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad() # Zero the parameter gradients

        # Forward pass
        outputs = model(images)
        
        # Calculate loss and metric
        loss = criterion(outputs, masks)
        dice = dice_score(outputs, masks)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running stats (scaled by batch size for correct averaging)
        running_loss += loss.item() * images.size(0)
        running_dice += dice.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)
    return epoch_loss, epoch_dice

def validate_one_epoch(model, dataloader, criterion, device):
    """Performs one epoch of validation."""
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    running_dice = 0.0
    
    with torch.no_grad(): # Disable gradient calculation during validation
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            
            # Calculate loss and metric
            loss = criterion(outputs, masks)
            dice = dice_score(outputs, masks)
            
            running_loss += loss.item() * images.size(0)
            running_dice += dice.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)
    return epoch_loss, epoch_dice


# --- Main Execution ---

if __name__ == "__main__":
    
    # 1. Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Data Loading
    try:
        train_loader, val_loader, train_size, val_size = get_train_val_loaders(
            base_dir=BASE_DATA_DIRECTORY,
            raw_dir=RAW_CHANNEL_DIR,
            label_dir=LABEL_MASK_DIR,
            raw_ext=RAW_EXTENSION,
            label_ext=LABEL_EXTENSION,
            target_shape=(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            val_split=VAL_SPLIT
        )
        print(f"\nTraining set size: {train_size}")
        print(f"Validation set size: {val_size}")
        
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please check the BASE_DATA_DIRECTORY and ensure all necessary libraries are installed (torch, SimpleITK, scipy, numpy, tqdm).")
        sys.exit(1)
        
    # 3. Model, Loss, and Optimizer Setup
    model = UNet3D(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler to reduce learning rate if validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # 4. Logging Setup
    # Create the log file with headers
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train Dice', 'Val Loss', 'Val Dice', 'Time (s)'])

    # 5. Training Loop
    print("\nStarting PyTorch 3D U-Net training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        # Training phase
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        print(f"Time: {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            # Save the model's state dictionary
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # Log results
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_dice:.4f}", f"{val_loss:.4f}", f"{val_dice:.4f}", f"{epoch_time:.2f}"])
            
    print("\nTraining complete.")
    print(f"Best model weights saved to {MODEL_SAVE_PATH}")