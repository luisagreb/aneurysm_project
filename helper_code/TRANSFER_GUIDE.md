# Transfer Dataset to VM - Quick Guide

## Method 1: Using rsync (Recommended - Fastest)

### Step 1: Edit the transfer script
```bash
# Edit transfer_to_vm.sh and update:
# - VM_USER_HOST="your-username@vm-ip-address"
# - REMOTE_PATH="~/datasets/nucleus_segmentation"  # or your preferred path
```

### Step 2: Run the script
```bash
cd helper_code
chmod +x transfer_to_vm.sh
./transfer_to_vm.sh
```

### Manual rsync command:
```bash
rsync -avz --progress \
  /Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/ \
  user@vm-ip:~/datasets/nucleus_segmentation/
```

## Method 2: Using SCP (Simple, but slower)

```bash
# Transfer entire folder
scp -r /Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation \
  user@vm-ip:~/datasets/

# Or transfer specific folders
scp -r /Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/imagesTr \
  user@vm-ip:~/datasets/nucleus_segmentation/

scp -r /Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/labelsTr \
  user@vm-ip:~/datasets/nucleus_segmentation/

scp /Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/dataset.json \
  user@vm-ip:~/datasets/nucleus_segmentation/
```

## Method 3: Using SFTP (Interactive)

```bash
# Connect via SFTP
sftp user@vm-ip

# In SFTP prompt:
mkdir -p datasets/nucleus_segmentation
cd datasets/nucleus_segmentation
lcd /Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation
put -r imagesTr
put -r labelsTr
put dataset.json
exit
```

## Method 4: Mount Network Drive (If VM can access)

If your VM can access network storage:
```bash
# On VM, mount the network drive
# Then copy directly
cp -r /mnt/network_drive/nucleus_segmentation ~/datasets/
```

## Method 5: Using Cloud Storage (Google Drive, Dropbox, etc.)

1. Upload dataset to cloud storage from Mac
2. Download on VM
3. Extract and verify

## After Transfer - Verify on VM

```bash
# SSH into VM
ssh user@vm-ip

# Check dataset structure
ls -la ~/datasets/nucleus_segmentation/
ls -la ~/datasets/nucleus_segmentation/imagesTr | head -10
ls -la ~/datasets/nucleus_segmentation/labelsTr | head -10

# Count files
find ~/datasets/nucleus_segmentation/imagesTr -name "*.nii.gz" | wc -l
find ~/datasets/nucleus_segmentation/labelsTr -name "*.nii.gz" | wc -l

# Verify dataset.json
cat ~/datasets/nucleus_segmentation/dataset.json
```

## Quick Transfer Commands

### Copy-paste ready (update credentials):

```bash
# Replace these variables:
VM_USER="your-username"
VM_IP="your-vm-ip-address"
REMOTE_PATH="~/datasets/nucleus_segmentation"

# Then run:
rsync -avz --progress \
  /Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/ \
  ${VM_USER}@${VM_IP}:${REMOTE_PATH}/
```

## Troubleshooting

### SSH Key Setup (to avoid password prompts):
```bash
# Generate SSH key if you don't have one
ssh-keygen -t rsa -b 4096

# Copy to VM
ssh-copy-id user@vm-ip

# Now you can SSH without password
```

### Check disk space on VM:
```bash
ssh user@vm-ip "df -h"
```

### Resume interrupted transfer:
```bash
# rsync automatically resumes, just run again
rsync -avz --progress source/ user@vm-ip:destination/
```

### Check transfer progress:
```bash
# rsync shows progress automatically
# For scp, you can use pv (pipe viewer):
brew install pv  # On Mac
tar czf - source/ | pv | ssh user@vm-ip "cd destination && tar xzf -"
```

## Estimated Transfer Time

- 25 images (~50-100 MB each) = ~1.25-2.5 GB
- 25 labels (~5-20 MB each) = ~125-500 MB
- Total: ~1.5-3 GB

Transfer time depends on:
- Network speed
- Method used (rsync is fastest)
- VM location (local network vs cloud)

Typical times:
- Local network: 1-5 minutes
- Cloud VM: 5-30 minutes

