import os
import sys
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# --- Handling imports for module execution (fixed previously) ---
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
# -------------------------------------------------------------------


# --- Configuration Variables ---

# UPDATED PATH: This is the folder that contains your component subdirectories.
BASE_DATA_DIRECTORY = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/Data"
RAW_CHANNEL_DIR = "raw_ch1_nucleus"
LABEL_MASK_DIR = "label_nucleus"
RAW_EXTENSION = ".nii.gz"
LABEL_EXTENSION = ".nrrd"

# Model and Training Parameters
IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH = 64, 64, 64 # Target resolution for all volumes
NUM_CLASSES = 1                              # Binary segmentation (Nucleus vs Background)
BATCH_SIZE = 4
EPOCHS = 100
VAL_SPLIT = 0.15                             # 15% for validation
MODEL_SAVE_PATH = "3d_unet_nucleus_seg.h5"   # Where to save the best model weights

# --- Set up TensorFlow Environment (omitted for brevity) ---


# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Configuration Output
    print("--- Configuration ---")
    print(f"Base Data Directory: {BASE_DATA_DIRECTORY}")
    print(f"Script will search for RAW images here: {os.path.join(BASE_DATA_DIRECTORY, RAW_CHANNEL_DIR)}")
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
            print(f"\nERROR: No paired data files found.")
            print(f"       Raw files found by glob, but no corresponding label mask found in: {os.path.join(BASE_DATA_DIRECTORY, LABEL_MASK_DIR)}")
            print("Please check the naming convention and paths.")
            sys.exit(1)
            
        print(f"\nFound {len(raw_paths)} paired image/mask volumes.")
        
        # DEBUG: Print the first few paired paths to verify
        print("--- Verified Data Pairs (First 5) ---")
        for i in range(min(5, len(raw_paths))):
             print(f"RAW {i+1}: {raw_paths[i]}")
             print(f"LABEL {i+1}: {label_paths[i]}")
        print("-----------------------------------")


    except Exception as e:
        print(f"An error occurred during data path retrieval: {e}")
        sys.exit(1)

    # 3. Split Data for Training and Validation (omitted for brevity)
    # ... rest of the training script ...
    indices = np.arange(len(raw_paths))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=VAL_SPLIT,
        random_state=42
    )
    
    train_raw_paths = [raw_paths[i] for i in train_indices]
    train_label_paths = [label_paths[i] for i in train_indices]
    val_raw_paths = [raw_paths[i] for i in val_indices]
    val_label_paths = [label_paths[i] for i in val_indices]
    
    print(f"Training set size: {len(train_raw_paths)}")
    print(f"Validation set size: {len(val_label_paths)}")

    # Create generators
    train_generator = prepare_data_generator(
        raw_paths=train_raw_paths,
        label_paths=train_label_paths,
        target_shape=(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        shuffle=True
    )

    validation_generator = prepare_data_generator(
        raw_paths=val_raw_paths,
        label_paths=val_label_paths,
        target_shape=(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        shuffle=False
    )

    # 4. Build and Compile Model (omitted for brevity)
    model = unet_model(
        input_shape=(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, 1),
        num_classes=NUM_CLASSES
    )

    # 5. Define Callbacks (omitted for brevity)
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(patience=10, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss', verbose=1, min_lr=1e-6),
        CSVLogger("training_log.csv", separator=",", append=True)
    ]

    # 6. Training the Model
    steps_per_epoch = len(train_raw_paths) // BATCH_SIZE
    validation_steps = len(val_raw_paths) // BATCH_SIZE
    
    if steps_per_epoch == 0 and len(train_raw_paths) > 0:
        print(f"\nWarning: Not enough training data ({len(train_raw_paths)} files) for batch size {BATCH_SIZE}. Training steps will be zero or very few.")
    
    if steps_per_epoch == 0 and len(train_raw_paths) == 0:
        print("\nERROR: Steps per epoch is zero because the training set is empty.")
        sys.exit(1)


    print("\nStarting model training...")

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    print("\nTraining complete.")
    print(f"Best model weights saved to {MODEL_SAVE_PATH}")