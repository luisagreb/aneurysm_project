import SimpleITK as sitk
import os
from pathlib import Path

def fix_all_headers(images_dir, labels_dir):
    # Get all label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')]
    
    for lab_name in label_files:
        # Construct the matching image name (e.g., case_0002.nii.gz -> case_0002_0000.nii.gz)
        img_name = lab_name.replace(".nii.gz", "_0000.nii.gz")
        
        img_path = os.path.join(images_dir, img_name)
        lab_path = os.path.join(labels_dir, lab_name)
        
        if os.path.exists(img_path):
            img = sitk.ReadImage(img_path)
            lab = sitk.ReadImage(lab_path)
            
            # Match the label metadata to the image
            lab.SetSpacing(img.GetSpacing())
            lab.SetOrigin(img.GetOrigin())
            lab.SetDirection(img.GetDirection())
            
            # Overwrite the label with the fixed header
            sitk.WriteImage(lab, lab_path)
            print(f"✅ Aligned: {lab_name} to {img_name}")
        else:
            print(f"⚠️ Warning: No matching image found for {lab_name}")

# Set your paths here
images_tr_path = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/nnUNet_raw/DatasetXXX_Actin/imagesTr"
labels_tr_path = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/nnUNet_raw/DatasetXXX_Actin/labelsTr"

fix_all_headers(images_tr_path, labels_tr_path)