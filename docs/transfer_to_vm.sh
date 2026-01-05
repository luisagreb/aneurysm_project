#!/bin/bash
# Script to transfer nnUNet dataset from Mac to VM
# Usage: ./transfer_to_vm.sh [vm_user@vm_ip] [remote_path]

set -e

# Configuration
SOURCE_DIR="/Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation"
VM_USER_HOST=${1:-"user@vm-ip"}  # Change this to your VM credentials
REMOTE_PATH=${2:-"~/datasets/nucleus_segmentation"}  # Change this to your desired path on VM

echo "=========================================="
echo "nnUNet Dataset Transfer Script"
echo "=========================================="
echo ""
echo "Source: $SOURCE_DIR"
echo "Destination: $VM_USER_HOST:$REMOTE_PATH"
echo ""
echo "Make sure you have:"
echo "  1. SSH access to VM"
echo "  2. Sufficient disk space on VM"
echo "  3. Network connection"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Check if source exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "✗ Error: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Check if rsync is available (preferred method)
if command -v rsync &> /dev/null; then
    echo "Using rsync (recommended - faster, resumable)..."
    echo ""
    
    # Create remote directory first
    echo "Creating remote directory..."
    ssh "$VM_USER_HOST" "mkdir -p $(dirname $REMOTE_PATH)"
    
    # Transfer with rsync
    echo "Transferring files..."
    echo "This may take a while depending on file sizes..."
    echo ""
    
    rsync -avz --progress \
        --exclude='*.DS_Store' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        "$SOURCE_DIR/" "$VM_USER_HOST:$REMOTE_PATH/"
    
    echo ""
    echo "✓ Transfer complete!"
    
elif command -v scp &> /dev/null; then
    echo "Using scp (slower, but works)..."
    echo ""
    echo "Transferring files..."
    echo "This may take a while..."
    echo ""
    
    # Create remote directory
    ssh "$VM_USER_HOST" "mkdir -p $REMOTE_PATH"
    
    # Transfer
    scp -r "$SOURCE_DIR"/* "$VM_USER_HOST:$REMOTE_PATH/"
    
    echo ""
    echo "✓ Transfer complete!"
    
else
    echo "✗ Error: Neither rsync nor scp found!"
    echo "Please install one of them:"
    echo "  - rsync: brew install rsync (on Mac)"
    echo "  - scp: Usually pre-installed"
    exit 1
fi

# Verify transfer
echo ""
echo "Verifying transfer..."
FILE_COUNT_LOCAL=$(find "$SOURCE_DIR" -name "*.nii.gz" | wc -l | tr -d ' ')
FILE_COUNT_REMOTE=$(ssh "$VM_USER_HOST" "find $REMOTE_PATH -name '*.nii.gz' 2>/dev/null | wc -l" | tr -d ' ')

echo "  Local files: $FILE_COUNT_LOCAL"
echo "  Remote files: $FILE_COUNT_REMOTE"

if [ "$FILE_COUNT_LOCAL" -eq "$FILE_COUNT_REMOTE" ]; then
    echo "  ✓ File counts match!"
else
    echo "  ⚠ File counts don't match. Please verify manually."
fi

echo ""
echo "=========================================="
echo "Transfer Summary"
echo "=========================================="
echo "Dataset location on VM: $REMOTE_PATH"
echo ""
echo "Next steps on VM:"
echo "1. SSH into VM:"
echo "   ssh $VM_USER_HOST"
echo ""
echo "2. Verify dataset structure:"
echo "   ls -la $REMOTE_PATH"
echo "   ls -la $REMOTE_PATH/imagesTr | head"
echo "   ls -la $REMOTE_PATH/labelsTr | head"
echo ""
echo "3. Copy to nnUNet_raw folder:"
echo "   export DATASET_ID=001"
echo "   mkdir -p \$nnUNet_raw/Dataset\${DATASET_ID}_NucleusSeg"
echo "   cp -r $REMOTE_PATH/* \$nnUNet_raw/Dataset\${DATASET_ID}_NucleusSeg/"
echo ""
echo "4. Run preprocessing:"
echo "   nnUNetv2_plan_and_preprocess -d \$DATASET_ID --verify_dataset_integrity"
echo "=========================================="

