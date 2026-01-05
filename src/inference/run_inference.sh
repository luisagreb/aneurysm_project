#!/bin/bash
# Run inference for Actin and Mitochondria models

# Set nnUNet environment variables
# Set nnUNet environment variables
export nnUNet_raw="/home/luisa/aneurysm_project/data/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/luisa/aneurysm_project/data/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/luisa/aneurysm_project/data/nnUNet/nnUNet_results"

# Define paths
ACTIN_INPUT="/home/luisa/aneurysm_project/data/processed/actin_segmentation/imagesTs"
ACTIN_OUTPUT="/home/luisa/aneurysm_project/test_results/actin_predictions"

MITO_INPUT="/home/luisa/aneurysm_project/data/processed/nnUnet_mitochondria/Dataset002_Mito/imagesTs"
MITO_OUTPUT="/home/luisa/aneurysm_project/test_results/mito_predictions"

# Create output directories
mkdir -p "$ACTIN_OUTPUT"
mkdir -p "$MITO_OUTPUT"

echo "============================================"
echo "Starting Inference"
echo "============================================"

# 1. Run Actin Prediction
echo "Running Actin Model (Dataset 001)..."
nnUNetv2_predict -i "$ACTIN_INPUT" -o "$ACTIN_OUTPUT" -d 001 -c 3d_fullres -f 0

echo "--------------------------------------------"

# 2. Run Mitochondria Prediction
echo "Running Mitochondria Model (Dataset 002)..."
nnUNetv2_predict -i "$MITO_INPUT" -o "$MITO_OUTPUT" -d 002 -c 3d_fullres -f 0

echo "============================================"
echo "Inference Complete!"
echo "Actin results: $ACTIN_OUTPUT"
echo "Mito results:  $MITO_OUTPUT"
