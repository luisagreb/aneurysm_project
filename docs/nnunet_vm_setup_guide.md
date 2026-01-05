# nnUNet VM Setup Guide

## Prerequisites

1. Dataset prepared on your Mac: `/Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation`
2. VM with GPU access (recommended)
3. Python environment on VM

## Step 1: Transfer Dataset to VM

### Option A: Using SCP (Secure Copy)
```bash
# From your Mac, transfer the dataset folder
scp -r /Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation user@vm-ip:/path/to/datasets/

# Or transfer just the necessary files
scp -r /Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/imagesTr user@vm-ip:/path/to/datasets/nucleus_segmentation/
scp -r /Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/labelsTr user@vm-ip:/path/to/datasets/nucleus_segmentation/
scp /Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/dataset.json user@vm-ip:/path/to/datasets/nucleus_segmentation/
```

### Option B: Using rsync (faster, resumable)
```bash
rsync -avz --progress /Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/ user@vm-ip:/path/to/datasets/nucleus_segmentation/
```

### Option C: Mount Network Drive
If VM can access network storage, mount the drive directly.

## Step 2: Install nnUNet on VM

```bash
# SSH into your VM
ssh user@vm-ip

# Create conda environment (recommended)
conda create -n nnunet python=3.9
conda activate nnunet

# Install nnUNet
pip install nnunetv2

# Verify installation
nnUNetv2_plan_and_preprocess -h
```

## Step 3: Set Environment Variables

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
# nnUNet environment variables
export nnUNet_raw="/path/to/datasets/nnUNet_raw"
export nnUNet_preprocessed="/path/to/datasets/nnUNet_preprocessed"
export nnUNet_results="/path/to/datasets/nnUNet_results"

# Create directories
mkdir -p $nnUNet_raw
mkdir -p $nnUNet_preprocessed
mkdir -p $nnUNet_results
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Step 4: Prepare Dataset for nnUNet

nnUNet expects datasets in a specific structure:

```
nnUNet_raw/
└── Dataset001_NucleusSeg/
    ├── dataset.json
    ├── imagesTr/
    │   ├── case_0001_0000.nii.gz
    │   └── ...
    └── labelsTr/
        ├── case_0001.nii.gz
        └── ...
```

### Copy your dataset:

```bash
# On VM
DATASET_ID=001  # Change this if you already have datasets
DATASET_NAME="NucleusSeg"

# Create dataset folder
mkdir -p $nnUNet_raw/Dataset${DATASET_ID}_${DATASET_NAME}

# Copy files
cp -r /path/to/datasets/nucleus_segmentation/* $nnUNet_raw/Dataset${DATASET_ID}_${DATASET_NAME}/

# Verify structure
ls -la $nnUNet_raw/Dataset${DATASET_ID}_${DATASET_NAME}/
```

## Step 5: Preprocess Dataset

```bash
# Activate environment
conda activate nnunet

# Plan and preprocess (this analyzes your dataset)
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

# This will:
# - Verify dataset integrity
# - Create preprocessing plans
# - Preprocess data for training
```

## Step 6: Train nnUNet

### Option A: 2D Training (faster, less memory)
```bash
# Train 2D U-Net
nnUNetv2_train Dataset${DATASET_ID} 2d 0

# This trains fold 0. For 5-fold cross-validation, train folds 0-4:
nnUNetv2_train Dataset${DATASET_ID} 2d 0
nnUNetv2_train Dataset${DATASET_ID} 2d 1
nnUNetv2_train Dataset${DATASET_ID} 2d 2
nnUNetv2_train Dataset${DATASET_ID} 2d 3
nnUNetv2_train Dataset${DATASET_ID} 2d 4
```

### Option B: 3D Full Resolution (best quality, needs GPU)
```bash
# Train 3D full resolution U-Net
nnUNetv2_train Dataset${DATASET_ID} 3d_fullres 0

# For 5-fold cross-validation:
for fold in 0 1 2 3 4; do
    nnUNetv2_train Dataset${DATASET_ID} 3d_fullres $fold
done
```

### Option C: 3D Low Resolution (faster than fullres)
```bash
nnUNetv2_train Dataset${DATASET_ID} 3d_lowres 0
```

## Step 7: Predict on Test Data

```bash
# After training, predict on test images
nnUNetv2_predict -i /path/to/test/images -o /path/to/output -d Dataset${DATASET_ID} -f 0

# Or use ensemble of all folds (better results)
nnUNetv2_predict -i /path/to/test/images -o /path/to/output -d Dataset${DATASET_ID} -f 0 1 2 3 4
```

## Step 8: Monitor Training

Training logs are saved in:
```
$nnUNet_results/Dataset${DATASET_ID}_${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/
```

Check progress:
```bash
tail -f $nnUNet_results/Dataset${DATASET_ID}_${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log.txt
```

## Quick Reference Commands

```bash
# Set dataset ID
export DATASET_ID=001

# Verify dataset
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

# Train single fold
nnUNetv2_train Dataset${DATASET_ID} 3d_fullres 0

# Train all folds (5-fold cross-validation)
for fold in 0 1 2 3 4; do
    nnUNetv2_train Dataset${DATASET_ID} 3d_fullres $fold
done

# Predict
nnUNetv2_predict -i /path/to/images -o /path/to/output -d Dataset${DATASET_ID} -f 0 1 2 3 4
```

## Troubleshooting

1. **Out of memory**: Use 2d or 3d_lowres instead of 3d_fullres
2. **CUDA errors**: Check GPU availability with `nvidia-smi`
3. **Dataset not found**: Verify environment variables and dataset folder structure
4. **Preprocessing fails**: Check that all images and labels have matching shapes

## Notes

- Training time: 2D (~hours), 3D (~days) depending on GPU
- Recommended: Start with 2D to verify everything works
- Best results: 3D full resolution with 5-fold cross-validation
- Dataset ID: nnUNet uses Dataset001, Dataset002, etc. Use next available number

