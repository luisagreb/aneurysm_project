#!/bin/bash
# nnUNet VM Setup Script
# Run this on your VM to set up nnUNet

set -e  # Exit on error

echo "=========================================="
echo "nnUNet VM Setup Script"
echo "=========================================="

# Configuration
DATASET_ID=${1:-001}  # First argument or default to 001
DATASET_NAME="NucleusSeg"
DATASET_PATH=${2:-"/path/to/datasets/nucleus_segmentation"}  # Second argument or default path

echo "Dataset ID: $DATASET_ID"
echo "Dataset Name: $DATASET_NAME"
echo "Dataset Path: $DATASET_PATH"
echo ""

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ Conda found"
    USE_CONDA=true
else
    echo "⚠ Conda not found, using system Python"
    USE_CONDA=false
fi

# Step 1: Create/activate environment
if [ "$USE_CONDA" = true ]; then
    echo ""
    echo "Step 1: Setting up conda environment..."
    if conda env list | grep -q "^nnunet "; then
        echo "  Environment 'nnunet' exists, activating..."
        eval "$(conda shell.bash hook)"
        conda activate nnunet
    else
        echo "  Creating new conda environment 'nnunet'..."
        conda create -n nnunet python=3.9 -y
        eval "$(conda shell.bash hook)"
        conda activate nnunet
    fi
else
    echo ""
    echo "Step 1: Using system Python"
    echo "  Make sure Python 3.9+ is installed"
fi

# Step 2: Install nnUNet
echo ""
echo "Step 2: Installing nnUNet..."
pip install nnunetv2

# Step 3: Set environment variables
echo ""
echo "Step 3: Setting up environment variables..."

# Get absolute paths
NNUNET_RAW="${NNUNET_RAW:-$HOME/nnUNet_raw}"
NNUNET_PREPROCESSED="${NNUNET_PREPROCESSED:-$HOME/nnUNet_preprocessed}"
NNUNET_RESULTS="${NNUNET_RESULTS:-$HOME/nnUNet_results}"

echo "  nnUNet_raw: $NNUNET_RAW"
echo "  nnUNet_preprocessed: $NNUNET_PREPROCESSED"
echo "  nnUNet_results: $NNUNET_RESULTS"

# Create directories
mkdir -p "$NNUNET_RAW"
mkdir -p "$NNUNET_PREPROCESSED"
mkdir -p "$NNUNET_RESULTS"

# Add to bashrc if not already there
BASHRC_FILE="$HOME/.bashrc"
if ! grep -q "export nnUNet_raw" "$BASHRC_FILE" 2>/dev/null; then
    echo "" >> "$BASHRC_FILE"
    echo "# nnUNet environment variables" >> "$BASHRC_FILE"
    echo "export nnUNet_raw=\"$NNUNET_RAW\"" >> "$BASHRC_FILE"
    echo "export nnUNet_preprocessed=\"$NNUNET_PREPROCESSED\"" >> "$BASHRC_FILE"
    echo "export nnUNet_results=\"$NNUNET_RESULTS\"" >> "$BASHRC_FILE"
    echo "  ✓ Added environment variables to ~/.bashrc"
else
    echo "  ⚠ Environment variables already in ~/.bashrc"
fi

# Export for current session
export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"

# Step 4: Copy dataset
echo ""
echo "Step 4: Setting up dataset..."

DATASET_DIR="$NNUNET_RAW/Dataset${DATASET_ID}_${DATASET_NAME}"

if [ -d "$DATASET_PATH" ]; then
    echo "  Copying dataset from $DATASET_PATH to $DATASET_DIR..."
    mkdir -p "$DATASET_DIR"
    cp -r "$DATASET_PATH"/* "$DATASET_DIR/"
    echo "  ✓ Dataset copied"
else
    echo "  ⚠ Dataset path not found: $DATASET_PATH"
    echo "  Please copy your dataset manually to: $DATASET_DIR"
    echo "  Required structure:"
    echo "    $DATASET_DIR/"
    echo "      ├── dataset.json"
    echo "      ├── imagesTr/"
    echo "      └── labelsTr/"
fi

# Step 5: Verify dataset
echo ""
echo "Step 5: Verifying dataset structure..."

if [ -f "$DATASET_DIR/dataset.json" ]; then
    echo "  ✓ dataset.json found"
else
    echo "  ✗ dataset.json missing!"
fi

if [ -d "$DATASET_DIR/imagesTr" ]; then
    IMG_COUNT=$(find "$DATASET_DIR/imagesTr" -name "*.nii.gz" | wc -l)
    echo "  ✓ imagesTr/ found ($IMG_COUNT files)"
else
    echo "  ✗ imagesTr/ missing!"
fi

if [ -d "$DATASET_DIR/labelsTr" ]; then
    LBL_COUNT=$(find "$DATASET_DIR/labelsTr" -name "*.nii.gz" | wc -l)
    echo "  ✓ labelsTr/ found ($LBL_COUNT files)"
else
    echo "  ✗ labelsTr/ missing!"
fi

# Step 6: Preprocess
echo ""
echo "Step 6: Planning and preprocessing..."
echo "  This may take a while..."
echo "  Run this command manually:"
echo ""
echo "  nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity"
echo ""

# Summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Verify dataset:"
echo "   nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity"
echo ""
echo "2. Train model (choose one):"
echo "   # 2D (faster):"
echo "   nnUNetv2_train Dataset${DATASET_ID} 2d 0"
echo ""
echo "   # 3D full resolution (best quality, needs GPU):"
echo "   nnUNetv2_train Dataset${DATASET_ID} 3d_fullres 0"
echo ""
echo "3. For 5-fold cross-validation, train all folds:"
echo "   for fold in 0 1 2 3 4; do"
echo "       nnUNetv2_train Dataset${DATASET_ID} 3d_fullres \$fold"
echo "   done"
echo ""
echo "Environment variables are set for this session."
echo "To use in new sessions, run: source ~/.bashrc"
echo "=========================================="

