import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
import matplotlib.pyplot as plt

# Local imports
try:
    from uNet_model import unet_model
    from dataProcessing import (
        get_data_paths,
        load_and_preprocess_volume,
        prepare_data_generator,
    )
except ImportError:
    # if called from repo root
    from modelUnet_TensorFlow.uNet_model import unet_model
    from modelUnet_TensorFlow.dataProcessing import (
        get_data_paths,
        load_and_preprocess_volume,
        prepare_data_generator,
    )

# ---------------- CONFIG ----------------

# MACOS example:
# BASE_DATA_DIRECTORY = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/Data"
# WINDOWS:
BASE_DATA_DIRECTORY = r"C:\Users\Luisa\Documents\aneurysm_project\Data"

RAW_CHANNEL_DIR = "raw_ch1_nucleus"
LABEL_MASK_DIR = "label_nucleus"
RAW_EXTENSION = ".nii"
LABEL_EXTENSION = ".nrrd"

IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH = 64, 64, 64
INPUT_SHAPE = (IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, 1)
NUM_CLASSES = 1

BATCH_SIZE = 2        # keep small for RAM
EPOCHS = 150          # large, but early stopping will cut
VAL_SPLIT = 0.15
MODEL_SAVE_PATH = "3d_unet_nucleus_seg.h5"
LOG_CSV_PATH = "training_log.csv"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def plot_history(history, out_prefix=""):
    hist = history.history
    print("History keys:", hist.keys())

    # Loss
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
    plt.savefig(out_prefix + "loss_curve.png", dpi=150)

    # Dice
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
        plt.savefig(out_prefix + "dice_curve.png", dpi=150)

    # LR
    if "lr" in hist:
        plt.figure()
        plt.plot(hist["lr"], label="Learning rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.title("Learning rate schedule")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_prefix + "lr_curve.png", dpi=150)


if __name__ == "__main__":

    print("--- Configuration ---")
    print(f"Base Data Directory: {BASE_DATA_DIRECTORY}")
    print(f"Script will search for RAW images here: "
          f"{os.path.join(BASE_DATA_DIRECTORY, RAW_CHANNEL_DIR)}")
    print(f"Script will search for LABEL masks here: "
          f"{os.path.join(BASE_DATA_DIRECTORY, LABEL_MASK_DIR)}")
    print(f"Target Volume Size: ({IMG_DEPTH}, {IMG_HEIGHT}, {IMG_WIDTH})")
    print("---------------------")

    # 1. Pair paths
    try:
        raw_paths, label_paths = get_data_paths(
            base_dir=BASE_DATA_DIRECTORY,
            raw_dir=RAW_CHANNEL_DIR,
            label_dir=LABEL_MASK_DIR,
            raw_ext=RAW_EXTENSION,
            label_ext=LABEL_EXTENSION,
        )
    except Exception as e:
        print(f"[train_model] ERROR during get_data_paths: {e}")
        sys.exit(1)

    if not raw_paths:
        print("\nERROR: No paired data files found.")
        print(f"       Please check {BASE_DATA_DIRECTORY}")
        sys.exit(1)

    print(f"\nFound {len(raw_paths)} paired image/mask volumes.")
    print("--- Verified Data Pairs (First 5) ---")
    for i in range(min(5, len(raw_paths))):
        print(f"RAW   {i+1}: {raw_paths[i]}")
        print(f"LABEL {i+1}: {label_paths[i]}")
    print("-------------------------------------")

    # 2. Train / Val split
    indices = np.arange(len(raw_paths))
    train_idx, val_idx = train_test_split(
        indices, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    train_raw_paths = [raw_paths[i] for i in train_idx]
    train_label_paths = [label_paths[i] for i in train_idx]
    val_raw_paths = [raw_paths[i] for i in val_idx]
    val_label_paths = [label_paths[i] for i in val_idx]

    print(f"Training set size:   {len(train_raw_paths)}")
    print(f"Validation set size: {len(val_label_paths)}")

    # 3. Generators
    target_shape = (IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH)

    train_gen = prepare_data_generator(
        raw_paths=train_raw_paths,
        label_paths=train_label_paths,
        target_shape=target_shape,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        shuffle=True,
        augment=True,
    )

    val_gen = prepare_data_generator(
        raw_paths=val_raw_paths,
        label_paths=val_label_paths,
        target_shape=target_shape,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        shuffle=False,
        augment=False,
    )

    steps_per_epoch = max(1, len(train_raw_paths) // BATCH_SIZE)
    validation_steps = max(1, len(val_raw_paths) // BATCH_SIZE)

    print(f"Steps per epoch:      {steps_per_epoch}")
    print(f"Validation steps:     {validation_steps}")

    # 4. Model
    model = unet_model(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        learning_rate=1e-4,
        class_weights=(0.2, 0.8),
    )
    model.summary()

    # 5. Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        EarlyStopping(
            patience=15,
            monitor="val_loss",
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            monitor="val_loss",
            verbose=1,
            min_lr=1e-6,
        ),
        CSVLogger(LOG_CSV_PATH, separator=",", append=True),
    ]

    print("\nStarting model training...\n")

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    print("\nTraining complete.")
    print(f"Best model weights saved to {MODEL_SAVE_PATH}")

    plot_history(history, out_prefix="")