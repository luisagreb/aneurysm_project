import os
import sys
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)
import matplotlib.pyplot as plt

# --- Handling imports for module execution ---
try:
    from uNet_model import unet_model
    from dataProcessing import (
        get_data_paths,
        load_and_preprocess_volume,
        prepare_data_generator,
    )
except ImportError:
    from modelUnet.uNet_model import unet_model
    from modelUnet.dataProcessing import (
        get_data_paths,
        load_and_preprocess_volume,
        prepare_data_generator,
    )

# =================== CONFIG ===================

# macOS example:
# BASE_DATA_DIRECTORY = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/Data"

# Windows:
BASE_DATA_DIRECTORY = r"C:\Users\Luisa\Documents\aneurysm_project\Data"

RAW_CHANNEL_DIR = "raw_ch1_nucleus"
LABEL_MASK_DIR = "label_nucleus"
RAW_EXTENSION = ".nii"   # on lab computer only .nii
LABEL_EXTENSION = ".nrrd"

# Model and Training Parameters
IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH = 64, 64, 64  # Target resolution for all volumes
NUM_CLASSES = 1                                 # Binary segmentation (Nucleus vs Background)
BATCH_SIZE = 4
EPOCHS = 5
VAL_SPLIT = 0.15                                # 15% for validation
MODEL_SAVE_PATH = "3d_unet_nucleus_seg.h5"      # Where to save the best model weights
LOG_CSV_PATH = "training_log.csv"

# ==============================================


if __name__ == "__main__":

    # 1. Configuration Output
    print("--- Configuration ---")
    print(f"Base Data Directory: {BASE_DATA_DIRECTORY}")
    print(f"Script will search for RAW images here:   {os.path.join(BASE_DATA_DIRECTORY, RAW_CHANNEL_DIR)}")
    print(f"Script will search for LABEL masks here: {os.path.join(BASE_DATA_DIRECTORY, LABEL_MASK_DIR)}")
    print(f"Target Volume Size: ({IMG_DEPTH}, {IMG_HEIGHT}, {IMG_WIDTH})")
    print("---------------------")

    # 2. Get Data Paths
    try:
        raw_paths, label_paths = get_data_paths(
            base_dir=BASE_DATA_DIRECTORY,
            raw_dir=RAW_CHANNEL_DIR,
            label_dir=LABEL_MASK_DIR,
            raw_ext=RAW_EXTENSION,
            label_ext=LABEL_EXTENSION,
        )

        if not raw_paths:
            print("\nERROR: No paired data files found.")
            print(f"       Raw files were searched under: {os.path.join(BASE_DATA_DIRECTORY, RAW_CHANNEL_DIR)}")
            print(f"       Label masks were searched under: {os.path.join(BASE_DATA_DIRECTORY, LABEL_MASK_DIR)}")
            print("       Please check the naming convention and paths.")
            sys.exit(1)

        print(f"\nFound {len(raw_paths)} paired image/mask volumes.")
        print("--- Verified Data Pairs (First 5) ---")
        for i in range(min(5, len(raw_paths))):
            print(f"RAW   {i+1}: {raw_paths[i]}")
            print(f"LABEL {i+1}: {label_paths[i]}")
        print("-------------------------------------")

    except Exception as e:
        print(f"An error occurred during data path retrieval: {e}")
        sys.exit(1)

    # 3. Split Data for Training and Validation
    indices = np.arange(len(raw_paths))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=VAL_SPLIT,
        random_state=42,
    )

    train_raw_paths = [raw_paths[i] for i in train_indices]
    train_label_paths = [label_paths[i] for i in train_indices]
    val_raw_paths = [raw_paths[i] for i in val_indices]
    val_label_paths = [label_paths[i] for i in val_indices]

    print(f"Training set size:   {len(train_raw_paths)}")
    print(f"Validation set size: {len(val_label_paths)}")

    # 4. Create generators
    train_generator = prepare_data_generator(
        raw_paths=train_raw_paths,
        label_paths=train_label_paths,
        target_shape=(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        shuffle=True,
    )

    validation_generator = None
    use_validation = len(val_raw_paths) > 0
    if use_validation:
        validation_generator = prepare_data_generator(
            raw_paths=val_raw_paths,
            label_paths=val_label_paths,
            target_shape=(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            num_classes=NUM_CLASSES,
            shuffle=False,
        )

    # 5. Build and Compile Model
    model = unet_model(
        input_shape=(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, 1),
        num_classes=NUM_CLASSES,
    )

    # 6. Define Callbacks
    callbacks = []

    # If we do have validation, we can monitor val_loss
    if use_validation:
        callbacks.append(
            ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", verbose=1)
        )
        callbacks.append(
            EarlyStopping(patience=10, monitor="val_loss", verbose=1, restore_best_weights=True)
        )
        callbacks.append(
            ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                monitor="val_loss",
                verbose=1,
                min_lr=1e-6,
            )
        )
    else:
        # Fallback: monitor training loss if no validation set
        print("\n⚠️ No validation set: callbacks will monitor training loss instead.\n")
        callbacks.append(
            ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="loss", verbose=1)
        )
        callbacks.append(
            EarlyStopping(patience=10, monitor="loss", verbose=1, restore_best_weights=True)
        )
        callbacks.append(
            ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                monitor="loss",
                verbose=1,
                min_lr=1e-6,
            )
        )

    callbacks.append(
        CSVLogger(LOG_CSV_PATH, separator=",", append=True)
    )

    # 7. Training the Model – robust step computation
    # Use ceil so we don't drop the "last small batch"
    steps_per_epoch = math.ceil(len(train_raw_paths) / BATCH_SIZE)

    if steps_per_epoch <= 0:
        print("\nERROR: Steps per epoch is zero because the training set is empty.")
        sys.exit(1)

    if use_validation:
        validation_steps = max(1, math.ceil(len(val_raw_paths) / BATCH_SIZE))
    else:
        validation_steps = None

    print("\nStarting model training...")
    if use_validation:
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
        )
    else:
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            callbacks=callbacks,
        )

    print("\nTraining complete.")
    print(f"Best model weights saved to {MODEL_SAVE_PATH}")

    # 8. Plot training curves
    hist = history.history
    print("History keys:", hist.keys())

    # --- Plot Loss ---
    plt.figure()
    plt.plot(hist["loss"], label="Train loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)

    # --- Plot Dice coefficient if present ---
    if "dice_coefficient" in hist:
        plt.figure()
        plt.plot(hist["dice_coefficient"], label="Train Dice")
        if "val_dice_coefficient" in hist:
            plt.plot(hist["val_dice_coefficient"], label="Val Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Dice coefficient")
        plt.title("Training vs Validation Dice")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("dice_coefficient_curve.png", dpi=150)

    # --- (Optional) Plot learning rate ---
    if "lr" in hist:
        plt.figure()
        plt.plot(hist["lr"], label="Learning rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.title("Learning rate schedule")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("lr_curve.png", dpi=150)