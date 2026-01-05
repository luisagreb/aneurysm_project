"""
Quick checklist to verify files are ready for nnUNet.
"""

from pathlib import Path
import json
import re
import numpy as np
import nibabel as nib


def check_nnunet_readiness(dataset_root):
    """
    Check if dataset is ready for nnUNet.
    
    Args:
        dataset_root: Root directory containing imagesTr, labelsTr folders
    """
    root = Path(dataset_root)
    
    print("=" * 70)
    print("nnUNet READINESS CHECKLIST")
    print("=" * 70)
    print(f"Dataset root: {dataset_root}\n")
    
    checks = []
    
    # Check 1: Folder structure
    images_tr = root / "imagesTr"
    images_ts = root / "imagesTs"
    labels_tr = root / "labelsTr"
    
    print("1. Folder Structure:")
    if images_tr.exists():
        print(f"   ✓ imagesTr/ exists")
        checks.append(True)
    else:
        print(f"   ✗ imagesTr/ missing")
        checks.append(False)
    
    if images_ts.exists():
        print(f"   ✓ imagesTs/ exists")
    else:
        print(f"   ⚠ imagesTs/ missing (optional for training)")
    
    if labels_tr.exists():
        print(f"   ✓ labelsTr/ exists")
        checks.append(True)
    else:
        print(f"   ✗ labelsTr/ missing")
        checks.append(False)
    
    # Check 2: Files exist
    print("\n2. Files:")
    if images_tr.exists():
        # Check for both correct pattern and any .nii.gz files
        image_files = list(images_tr.glob("case_*_0000.nii.gz"))
        all_nii_files = list(images_tr.glob("*.nii.gz"))
        
        print(f"   Found {len(image_files)} correctly named files (case_*_0000.nii.gz)")
        print(f"   Found {len(all_nii_files)} total .nii.gz files in imagesTr/")
        
        if len(all_nii_files) > 0 and len(image_files) == 0:
            print(f"   ⚠ Files exist but don't match naming pattern!")
            print(f"     Examples: {[f.name for f in all_nii_files[:5]]}")
            print(f"     → Run fix_nnunet_files.py to rename them")
        elif len(image_files) > 0:
            print(f"     Example: {image_files[0].name}")
        
        checks.append(len(all_nii_files) > 0)  # At least some files exist
    else:
        checks.append(False)
    
    if labels_tr.exists():
        label_files = list(labels_tr.glob("case_*.nii.gz"))
        # Filter out files with _0000 suffix (those are images, not labels)
        label_files = [f for f in label_files if not f.name.endswith("_0000.nii.gz")]
        print(f"   ✓ Found {len(label_files)} label files in labelsTr/")
        if len(label_files) > 0:
            print(f"     Example: {label_files[0].name}")
        checks.append(len(label_files) > 0)
    else:
        checks.append(False)
    
    # Check 3: Matching pairs
    print("\n3. Image-Label Pairs:")
    if images_tr.exists() and labels_tr.exists():
        # Check both correctly named and all files
        image_files_correct = sorted(images_tr.glob("case_*_0000.nii.gz"))
        image_files_all = sorted(images_tr.glob("*.nii.gz"))
        
        # Also check imagesTs for images
        if images_ts.exists():
            test_files_all = sorted(images_ts.glob("*.nii.gz"))
            test_files_correct = sorted(images_ts.glob("case_*_0000.nii.gz"))
            print(f"   imagesTs: {len(test_files_all)} total files ({len(test_files_correct)} correctly named)")
        else:
            test_files_all = []
            test_files_correct = []
        
        # Use correct pattern if available, otherwise use all files
        image_files = image_files_correct if image_files_correct else image_files_all
        
        label_files = sorted(labels_tr.glob("case_*.nii.gz"))
        label_files = [f for f in label_files if not f.name.endswith("_0000.nii.gz")]
        
        # Extract case IDs from images
        image_cases = set()
        for img in image_files:
            if "_0000" in img.stem:
                case_id = img.stem.rsplit("_", 1)[0]  # case_0000_0000 -> case_0000
            else:
                # Try to extract case number from any pattern
                match = re.search(r'(?:case_|Sample_)(\d+)', img.stem, re.IGNORECASE)
                if match:
                    case_id = f"case_{int(match.group(1)):04d}"
                else:
                    case_id = img.stem  # Use stem as fallback
            image_cases.add(case_id)
        
        # Extract case IDs from labels
        label_cases = set()
        for lbl in label_files:
            case_id = lbl.stem  # case_0000
            label_cases.add(case_id)
        
        matching = image_cases & label_cases
        missing_labels = image_cases - label_cases
        missing_images = label_cases - image_cases
        
        print(f"   ✓ Matching pairs: {len(matching)}")
        if len(image_files_all) == 0:
            print(f"   ⚠ No image files found in imagesTr/")
            print(f"     → Check if files need to be moved from imagesTs/ or renamed")
        if missing_labels:
            print(f"   ⚠ Images without labels: {len(missing_labels)}")
            print(f"     Examples: {list(missing_labels)[:5]}")
        if missing_images:
            print(f"   ⚠ Labels without images: {len(missing_images)}")
            print(f"     Examples: {list(missing_images)[:5]}")
            if len(image_files_all) == 0 and len(test_files_all) > 0:
                print(f"     → Images might be in imagesTs/ instead of imagesTr/")
        
        checks.append(len(matching) > 0 or len(image_files_all) > 0)
    else:
        checks.append(False)
    
    # Check 4: dataset.json
    print("\n4. dataset.json:")
    dataset_json = root / "dataset.json"
    if dataset_json.exists():
        print(f"   ✓ dataset.json exists")
        try:
            with open(dataset_json, 'r') as f:
                config = json.load(f)
            print(f"     Name: {config.get('name', 'N/A')}")
            print(f"     Training cases: {config.get('numTraining', 0)}")
            print(f"     Labels: {config.get('labels', {})}")
            checks.append(True)
        except Exception as e:
            print(f"   ✗ Error reading dataset.json: {e}")
            checks.append(False)
    else:
        print(f"   ✗ dataset.json missing")
        print(f"     You need to create this file for nnUNet")
        checks.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_checks = all(checks)
    if all_checks:
        print("✓ All checks passed! Your dataset appears ready for nnUNet.")
        print("\nNext steps:")
        print("1. Run validation: python helper_code/validate_nnunet_files.py <imagesTr> -l <labelsTr>")
        print("2. Start nnUNet training!")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Run fix_nnunet_files.py to rename files correctly")
        print("- Create dataset.json file (see prepare_actin_dataset_for_nnunet.py for example)")
        print("- Ensure imagesTr and labelsTr folders exist")
    
    print("=" * 70)
    
    # Return values for potential JSON creation
    matching_pairs = matching if images_tr.exists() and labels_tr.exists() and 'matching' in locals() else set()
    image_files_found = image_files_all if images_tr.exists() and 'image_files_all' in locals() else []
    label_files_found = label_files if labels_tr.exists() and 'label_files' in locals() else []
    test_files_found = test_files_all if images_ts.exists() and 'test_files_all' in locals() else []
    
    return all_checks, matching_pairs, image_files_found, label_files_found, test_files_found


def create_dataset_json(dataset_root, dataset_name="NucleusSeg", modality_name="nucleus", 
                        label_names=None, create_if_exists=False):
    """
    Create dataset.json file for nnUNet.
    
    Args:
        dataset_root: Root directory containing imagesTr, labelsTr, imagesTs
        dataset_name: Name of the dataset
        modality_name: Name of the imaging modality
        label_names: Dict mapping label IDs to names (e.g., {0: "background", 1: "nucleus"})
        create_if_exists: If True, overwrite existing dataset.json
    """
    root = Path(dataset_root)
    images_tr = root / "imagesTr"
    labels_tr = root / "labelsTr"
    images_ts = root / "imagesTs"
    
    dataset_json_path = root / "dataset.json"
    
    if dataset_json_path.exists() and not create_if_exists:
        print(f"\n⚠ dataset.json already exists at {dataset_json_path}")
        print("   Use --create-json to overwrite it")
        return False
    
    # Find training image-label pairs
    if not images_tr.exists() or not labels_tr.exists():
        print("\n✗ Cannot create dataset.json: imagesTr or labelsTr missing")
        return False
    
    # Get all image and label files
    image_files = sorted(images_tr.glob("case_*_0000.nii.gz"))
    if not image_files:
        # Try any .nii.gz files
        image_files = sorted(images_tr.glob("*.nii.gz"))
    
    label_files = sorted(labels_tr.glob("case_*.nii.gz"))
    label_files = [f for f in label_files if not f.name.endswith("_0000.nii.gz")]
    
    # Extract case IDs and find matching pairs
    train_cases = []
    for img_file in image_files:
        if "_0000" in img_file.stem:
            case_id = img_file.stem.rsplit("_", 1)[0]  # case_0000_0000 -> case_0000
        else:
            match = re.search(r'(?:case_|Sample_)(\d+)', img_file.stem, re.IGNORECASE)
            if match:
                case_id = f"case_{int(match.group(1)):04d}"
            else:
                continue
        
        # Check if corresponding label exists
        label_file = labels_tr / f"{case_id}.nii.gz"
        if label_file.exists():
            train_cases.append(case_id)
    
    # Get test cases
    test_cases = []
    if images_ts.exists():
        test_files = sorted(images_ts.glob("case_*_0000.nii.gz"))
        if not test_files:
            test_files = sorted(images_ts.glob("*.nii.gz"))
        
        for test_file in test_files:
            if "_0000" in test_file.stem:
                case_id = test_file.stem.rsplit("_", 1)[0]
            else:
                match = re.search(r'(?:case_|Sample_)(\d+)', test_file.stem, re.IGNORECASE)
                if match:
                    case_id = f"case_{int(match.group(1)):04d}"
                else:
                    continue
            
            # Only add if not in training set
            if case_id not in train_cases:
                test_cases.append(test_file.name)
    
    # Determine label names by checking a label file
    if label_names is None:
        label_names = {0: "background"}
        if label_files:
            try:
                lbl = nib.load(str(label_files[0]))
                data = np.asarray(lbl.dataobj)
                unique_labels = np.unique(data.astype(np.int32))
                unique_labels = unique_labels[unique_labels >= 0]  # Remove negative values
                
                # Create label names
                for i, label_id in enumerate(sorted(unique_labels)):
                    if label_id == 0:
                        label_names[0] = "background"
                    else:
                        label_names[int(label_id)] = f"class_{int(label_id)}"
            except Exception as e:
                print(f"   ⚠ Could not read labels to determine classes: {e}")
                label_names = {0: "background", 1: "nucleus"}  # Default
    
    # Create dataset.json structure
    dataset_json = {
        "name": dataset_name,
        "description": f"{modality_name} segmentation dataset",
        "tensorImageSize": "3D",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "modality": {
            "0": modality_name
        },
        "labels": label_names,
        "numTraining": len(train_cases),
        "numTest": len(test_cases),
        "training": [],
        "test": [],
        "channel_names": {
            "0": modality_name
        },
        "file_ending": ".nii.gz"
    }
    
    # Fill training entries
    for case_id in sorted(train_cases):
        dataset_json["training"].append({
            "image": f"./imagesTr/{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })
    
    # Fill test entries
    for test_file_name in sorted(test_cases):
        dataset_json["test"].append(f"./imagesTs/{test_file_name}")
    
    # Save dataset.json
    try:
        with open(dataset_json_path, 'w') as f:
            json.dump(dataset_json, f, indent=4)
        print(f"\n✓ Created dataset.json at: {dataset_json_path}")
        print(f"   Training cases: {len(train_cases)}")
        print(f"   Test cases: {len(test_cases)}")
        print(f"   Labels: {label_names}")
        return True
    except Exception as e:
        print(f"\n✗ Error creating dataset.json: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check if dataset is ready for nnUNet and optionally create dataset.json"
    )
    parser.add_argument(
        "dataset_root",
        type=str,
        nargs="?",
        default="/Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation",
        help="Root directory containing imagesTr and labelsTr folders"
    )
    parser.add_argument(
        "--create-json",
        action="store_true",
        help="Create dataset.json file if missing or to overwrite existing one"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="NucleusSeg",
        help="Name for the dataset (default: NucleusSeg)"
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="nucleus",
        help="Name of the imaging modality (default: nucleus)"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Label names in order (e.g., --labels background nucleus). First is background (0), rest are classes."
    )
    
    args = parser.parse_args()
    
    # Run readiness check
    all_checks, matching, image_files_all, label_files, test_files_all = check_nnunet_readiness(args.dataset_root)
    
    # Create dataset.json if requested
    if args.create_json:
        label_names = None
        if args.labels:
            label_names = {i: name for i, name in enumerate(args.labels)}
        
        create_dataset_json(
            args.dataset_root,
            dataset_name=args.dataset_name,
            modality_name=args.modality,
            label_names=label_names,
            create_if_exists=True
        )
    elif not all_checks:
        print("\n💡 Tip: Run with --create-json to automatically create dataset.json file")

