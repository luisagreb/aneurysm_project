#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
PROJECT_DIR="/home/luisa/aneurysm_project"
DATA_DIR="$PROJECT_DIR/data"

# Common Output Directories
export nnUNet_preprocessed="$DATA_DIR/nnUNet_preprocessed"
export nnUNet_results="$DATA_DIR/nnUNet_results"

# Ensure output directories exist
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

# Activate Virtual Environment
source "$PROJECT_DIR/venv/bin/activate"

echo "=========================================================="
echo "Starting nnU-Net Training Pipeline at $(date)"
echo "=========================================================="

# ---------------------------------------------------------
# 1. Train Actin (Dataset 001)
# ---------------------------------------------------------
echo ""
echo ">>> [1/4] Processing Actin (Dataset001)..."
export nnUNet_raw="$DATA_DIR/nnUnet_actin"

echo "Running Plan & Preprocess (Actin)..."
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity

echo ">>> [2/4] Training Actin (Dataset001, Fold 0)..."
nnUNetv2_train 001 3d_fullres 0

# ---------------------------------------------------------
# 2. Train Mitochondria (Dataset 002)
# ---------------------------------------------------------
echo ""
echo ">>> [3/4] Processing Mitochondria (Dataset002)..."
export nnUNet_raw="$DATA_DIR/nnUnet_mitochondria"

echo "Running Plan & Preprocess (Mito)..."
nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity

echo ">>> [4/4] Training Mitochondria (Dataset002, Fold 0)..."
nnUNetv2_train 002 3d_fullres 0

echo ""
echo "=========================================================="
echo "All Training Completed Successfully at $(date)"
echo "=========================================================="
