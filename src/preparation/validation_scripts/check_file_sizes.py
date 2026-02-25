"""
Check and compare file sizes of NII.GZ files to understand size differences.
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict


def analyze_file_sizes(folder_path):
    """
    Analyze file sizes and data characteristics of NII.GZ files.
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Directory {folder_path} does not exist")
        return
    
    # Find all NII.GZ files
    nii_files = sorted(folder.glob("*.nii.gz"))
    
    if not nii_files:
        print(f"No .nii.gz files found in {folder_path}")
        return
    
    print("=" * 80)
    print("FILE SIZE ANALYSIS")
    print("=" * 80)
    print(f"Folder: {folder_path}")
    print(f"Found {len(nii_files)} file(s)\n")
    
    file_info = []
    
    for nii_file in nii_files:
        try:
            # Get file size
            file_size = nii_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            # Load file and get data info
            img = nib.load(str(nii_file))
            data = img.get_fdata()
            
            shape = data.shape
            dtype = data.dtype
            data_min = float(np.min(data))
            data_max = float(np.max(data))
            data_mean = float(np.mean(data))
            
            # Calculate theoretical uncompressed size
            bytes_per_voxel = data.dtype.itemsize
            uncompressed_size = np.prod(shape) * bytes_per_voxel
            uncompressed_size_mb = uncompressed_size / (1024 * 1024)
            compression_ratio = uncompressed_size / file_size if file_size > 0 else 0
            
            # Count unique values (for labels)
            unique_count = len(np.unique(data))
            
            file_info.append({
                'name': nii_file.name,
                'size_mb': file_size_mb,
                'shape': shape,
                'dtype': str(dtype),
                'min': data_min,
                'max': data_max,
                'mean': data_mean,
                'unique_values': unique_count,
                'uncompressed_mb': uncompressed_size_mb,
                'compression_ratio': compression_ratio
            })
            
        except Exception as e:
            print(f"Error processing {nii_file.name}: {e}")
            continue
    
    # Print summary table
    print(f"{'Filename':<40} {'Size (MB)':<12} {'Shape':<20} {'Dtype':<10} {'Unique':<8} {'Compression':<12}")
    print("-" * 80)
    
    for info in file_info:
        print(f"{info['name']:<40} {info['size_mb']:>10.2f}  {str(info['shape']):<20} {info['dtype']:<10} "
              f"{info['unique_values']:<8} {info['compression_ratio']:>10.2f}x")
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    sizes = [info['size_mb'] for info in file_info]
    print(f"File sizes:")
    print(f"  Min:  {min(sizes):.2f} MB")
    print(f"  Max:  {max(sizes):.2f} MB")
    print(f"  Mean: {np.mean(sizes):.2f} MB")
    print(f"  Std:  {np.std(sizes):.2f} MB")
    
    # Group by dtype
    dtype_groups = defaultdict(list)
    for info in file_info:
        dtype_groups[info['dtype']].append(info['size_mb'])
    
    if len(dtype_groups) > 1:
        print(f"\nFile sizes by data type:")
        for dtype, sizes_list in dtype_groups.items():
            print(f"  {dtype}: {np.mean(sizes_list):.2f} MB (avg), {len(sizes_list)} files")
    
    # Group by shape
    shape_groups = defaultdict(list)
    for info in file_info:
        shape_groups[str(info['shape'])].append(info['size_mb'])
    
    if len(shape_groups) > 1:
        print(f"\nFile sizes by shape:")
        for shape, sizes_list in sorted(shape_groups.items()):
            print(f"  {shape}: {np.mean(sizes_list):.2f} MB (avg), {len(sizes_list)} files")
    
    # Check for outliers
    if len(sizes) > 1:
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        outliers = [info for info in file_info if abs(info['size_mb'] - mean_size) > 2 * std_size]
        
        if outliers:
            print(f"\n⚠ Outliers (size differs by >2 std from mean):")
            for info in outliers:
                diff = info['size_mb'] - mean_size
                print(f"  {info['name']}: {info['size_mb']:.2f} MB (diff: {diff:+.2f} MB)")
                print(f"    Shape: {info['shape']}, Dtype: {info['dtype']}, Unique values: {info['unique_values']}")
    
    print("\n" + "=" * 80)
    print("EXPLANATION OF SIZE DIFFERENCES:")
    print("=" * 80)
    print("File sizes can differ due to:")
    print("1. Image dimensions (shape): Larger images = larger files")
    print("2. Data type: uint8 (1 byte) < uint16 (2 bytes) < float32 (4 bytes)")
    print("3. Compression efficiency: Gzip compresses repetitive data better")
    print("4. Data content: Labels (few unique values) compress better than images")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze file sizes of NII.GZ files"
    )
    parser.add_argument(
        "folder",
        type=str,
        nargs="?",
        default="/Volumes/LuisaHD/NewData/nnU-Net/nucleus_segmentation/imagesTr",
        help="Folder containing NII.GZ files to analyze"
    )
    
    args = parser.parse_args()
    
    analyze_file_sizes(args.folder)

