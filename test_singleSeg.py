import nibabel as nib
import numpy as np
import nrrd
import torch
# from your_model_library import ThreeDUnet # Placeholder for your model
file_path = "data/segmented_training/01Asc-180/+coll/01ASC-0180 +coll 60x DMSO48h-Zstack cell1/01ASC-0180 +coll 60x DMSO48h-Zstack cell1_ch1.nii.gz" # First cell

# 1. READ THE NII.GZ FILE
def load_nii_gz(file_path):
    """Loads a .nii.gz file and returns the 3D volume data and affine matrix."""
    nii_img = nib.load(file_path)
    data = nii_img.get_fdata()
    # Store the affine matrix to preserve spatial orientation for NRRD output
    return data, nii_img.affine

# 2. PRE-PROCESSING (Crucial for Robustness)
def preprocess(data):
    """Normalize and prepare data for the 3D U-Net model."""
    # Convert data type to float32 (required by most DL models)
    data = data.astype(np.float32)
    # Intensity normalization (e.g., Min-Max or Z-Score)
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    # Add batch and channel dimensions (e.g., [1, 1, D, H, W] for a single 3D image)
    return np.expand_dims(np.expand_dims(data, axis=0), axis=0)

# 3. SEGMENTATION INFERENCE (Core Robust Step)
def segment_with_unet(preprocessed_data, model):
    """Passes the data through the trained 3D U-Net model."""
    # Convert to PyTorch tensor
    input_tensor = torch.from_numpy(preprocessed_data).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        # The model outputs a probability map
        output_prob_map = model(input_tensor).squeeze().cpu().numpy()
        
    # Threshold the probability map (e.g., at 0.5) to get a binary mask
    segmentation_mask = (output_prob_map > 0.5).astype(np.uint8)
    
    return segmentation_mask

# 4. POST-PROCESSING (For Instance Separation)
# (Optional but recommended for robust individual nucleus analysis)
from skimage.measure import label
def postprocess_instance(segmentation_mask):
    """Separates individual nuclei using connected components."""
    # Label distinct regions (each nucleus gets a unique integer ID)
    labeled_mask = label(segmentation_mask, connectivity=3)
    return labeled_mask

# 5. WRITE THE .NRRD FILE
def save_nrrd(data, file_path, affine_matrix):
    """Writes the segmented 3D data to a .nrrd file, preserving spatial information."""
    # Extract the physical spacing from the affine matrix
    # This is a critical step for Slicer compatibility and dimensional accuracy
    spacing = np.diag(affine_matrix)[:3]
    
    header = {
        'space': 'left-posterior-superior', # Common Slicer/ITK convention
        'space directions': [
            (spacing[0], 0, 0), 
            (0, spacing[1], 0), 
            (0, 0, spacing[2])
        ],
        'type': 'uint8', # Segmentation labels are typically integers
        'encoding': 'gzip'
    }
    nrrd.write(file_path, data, header=header)

# MAIN EXECUTION
if __name__ == "__main__":
    nii_file = "01ASC-0180 +coll 60x DMSO48h-Zstack cell1_ch1.nii" # The nucleus channel
    output_nrrd_file = "Nucleus_Segmentation.nrrd"
    
    # 1. Load Data
    raw_data, affine = load_nii_gz(nii_file)
    
    # 2. Pre-process
    preprocessed_data = preprocess(raw_data)
    
    # 3. Load Model (You need to define and load your trained 3D U-Net model here)
    # model = ThreeDUnet()
    # model.load_state_dict(torch.load("best_nucleus_unet.pth"))
    # model.eval()
    
    # 4. Segmentation (Skipping actual model inference for code outline)
    # The output from the U-Net is what you want to save. 
    # For a placeholder, let's assume we have a mask.
    # segmentation_mask = segment_with_unet(preprocessed_data, model)
    
    # Placeholder: Use a mock mask for demonstration
    segmentation_mask = np.zeros(raw_data.shape, dtype=np.uint8) 
    # This should be replaced with the actual model output!
    
    # 5. Post-process (Instance Segmentation)
    final_labeled_segmentation = postprocess_instance(segmentation_mask)
    
    # 6. Save NRRD
    save_nrrd(final_labeled_segmentation, output_nrrd_file, affine)
    print(f"Segmentation saved to {output_nrrd_file}")