
import os
import shutil
import pandas as pd
from pathlib import Path

# Configuration
DATA_ROOT = Path("/home/luisa/aneurysm_project/data/nnUNet/nnUNet_raw")
INFERENCE_ROOT = Path("/home/luisa/aneurysm_project/experiments/V2/inference_results")
EXPORT_ROOT = Path("/home/luisa/aneurysm_project/experiments/V2/restored_names")

DATASETS = [
#    {
#        "name": "Actin",
#        "id": "Dataset001_Actin",
#        "train_map": DATA_ROOT / "Actin_mapping.csv",
#        "train_id_col": "CaseID",
#        "train_name_col": "OriginalCellName",
#        "test_map": DATA_ROOT / "Dataset001_Actin/test_mapping.csv",
#        "test_id_col": "TestID",
#        "test_name_col": "OriginalName"
#    },
#    {
#        "name": "Mito",
#        "id": "Dataset002_Mito",
#        "train_map": DATA_ROOT / "Mito_mapping.csv",
#        "train_id_col": "CaseID",
#        "train_name_col": "OriginalCellName",
#        "test_map": DATA_ROOT / "Dataset002_Mito/test_mapping.csv",
#        "test_id_col": "TestID",
#        "test_name_col": "OriginalName"
#    },
     {
         "name": "Nucleus",
         "id": "Dataset003_Nucleus",
         "train_map": DATA_ROOT / "Dataset003_Nucleus/Nucleus_mapping.csv",
         "train_id_col": "CaseID",
         "train_name_col": "OriginalName", 
         "test_map": DATA_ROOT / "Dataset003_Nucleus/test_mapping.csv",
         "test_id_col": "TestID",
         "test_name_col": "OriginalName"
     }
]

def safe_copy(src, dst):
    if not src.exists():
        return False
    # print(f"Copying {src.name} -> {dst.name}")
    shutil.copy2(src, dst)
    return True

def restore_dataset(ds):
    print(f"\n--- Restoring {ds['name']} ---")
    
    # 1. Training Data (Images & Manual Labels)
    if ds['train_map'].exists():
        print("  Processing Training Data...")
        df = pd.read_csv(ds['train_map'])
        
        out_train_img = EXPORT_ROOT / ds['name'] / "Training_Images"
        out_train_lbl = EXPORT_ROOT / ds['name'] / "Training_Labels"
        out_train_img.mkdir(parents=True, exist_ok=True)
        out_train_lbl.mkdir(parents=True, exist_ok=True)
        
        for _, row in df.iterrows():
            cid = row[ds['train_id_col']]
            orig = row[ds['train_name_col']]
            
            # Image
            src_img = DATA_ROOT / ds['id'] / "imagesTr" / f"{cid}_0000.nii.gz"
            dst_img = out_train_img / f"{orig}.nii.gz"
            safe_copy(src_img, dst_img)
            
            # Label
            src_lbl = DATA_ROOT / ds['id'] / "labelsTr" / f"{cid}.nii.gz"
            dst_lbl = out_train_lbl / f"{orig}.nii.gz"
            safe_copy(src_lbl, dst_lbl)
            
    # 2. Test Data & Inference Results
    if ds['test_map'].exists():
        print("  Processing Test Data & Inference...")
        df = pd.read_csv(ds['test_map'])
        
        out_test_img = EXPORT_ROOT / ds['name'] / "Test_Images"
        out_inf_raw = EXPORT_ROOT / ds['name'] / "Inference_Raw"
        out_inf_vis = EXPORT_ROOT / ds['name'] / "Inference_Visible"
        
        out_test_img.mkdir(parents=True, exist_ok=True)
        out_inf_raw.mkdir(parents=True, exist_ok=True)
        out_inf_vis.mkdir(parents=True, exist_ok=True)
        
        for _, row in df.iterrows():
            tid = row[ds['test_id_col']]
            orig = row[ds['test_name_col']]
            
            # Test Image
            src_img = DATA_ROOT / ds['id'] / "imagesTs" / f"{tid}_0000.nii.gz"
            dst_img = out_test_img / f"{orig}.nii.gz"
            safe_copy(src_img, dst_img)
            
            # Inference Raw
            src_inf = INFERENCE_ROOT / ds['id'] / f"{tid}.nii.gz"
            dst_inf = out_inf_raw / f"{orig}_segmentation.nii.gz"
            safe_copy(src_inf, dst_inf)
            
            # Inference Visible
            src_vis = INFERENCE_ROOT / ds['id'] / "visible" / f"{tid}.nii.gz"
            dst_vis = out_inf_vis / f"{orig}_segmentation_visible.nii.gz"
            safe_copy(src_vis, dst_vis)

    print(f"  Done. Check {EXPORT_ROOT / ds['name']}")

if __name__ == "__main__":
    print(f"Exporting data to: {EXPORT_ROOT}")
    # if EXPORT_ROOT.exists():
    #    shutil.rmtree(EXPORT_ROOT)
    
    for ds in DATASETS:
        restore_dataset(ds)
