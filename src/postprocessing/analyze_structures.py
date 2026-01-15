import os
import sys
import argparse
import numpy as np
import pandas as pd
import tifffile
from skimage import measure, morphology
from scipy.spatial import ConvexHull
from pathlib import Path

from skan import Skeleton, summarize
from skimage.measure import regionprops_table
import pandas as pd

def calculate_sphericity(volume, surface_area):
    """
    Calculate sphericity using the formula:
    Ψ = (π^(1/3) * (6V)^(2/3)) / A
    
    Sphericity ranges from 0 to 1, where 1 is a perfect sphere.
    Values may exceed 1 due to surface area underestimation by marching cubes
    algorithm when meshes are coarse, so we clamp to [0, 1].
    """
    if surface_area == 0:
        return 0.0
    raw_sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area
    # Clamp to valid range [0, 1] since marching cubes can underestimate surface area
    return min(1.0, max(0.0, raw_sphericity))

def analyze_mito(mask, voxel_size):
    """
    Analyze Mitochondria mask.
    Features: Total Volume, Surface Area, Sphericity, Fragment Count.
    """
    # Voxel dimensions
    vx, vy, vz = voxel_size
    voxel_vol = vx * vy * vz
    
    # Total Volume
    volume_pixels = np.sum(mask > 0)
    total_volume = volume_pixels * voxel_vol
    
    # Surface Area (Marching Cubes)
    # Use spacing to account for voxel size
    try:
        verts, faces, normals, values = measure.marching_cubes(mask, spacing=voxel_size)
        surface_area = measure.mesh_surface_area(verts, faces)
    except (ValueError, RuntimeError):
        # Handle cases where mask is empty or too small
        surface_area = 0.0
        
    # Sphericity
    sphericity = calculate_sphericity(total_volume, surface_area)

    # --- Advanced Network Topology (skan) ---
    # Skeletonize first
    try:
        from skan import Skeleton, summarize
        skeleton_image = morphology.skeletonize(mask)
        if np.sum(skeleton_image) > 1: # skan needs at least 2 pixels
             branch_data = summarize(Skeleton(skeleton_image))
             
             # Calculate metrics from branch_data dataframe
             junction_count = branch_data['node-id-src'].nunique() # distinct junctions approximately
             
             skel_obj = Skeleton(skeleton_image)
             degrees = skel_obj.degrees

             
             junction_count = np.sum(skel_obj.degrees > 2)
             branch_count = branch_data.shape[0]
             mean_branch_length = branch_data['branch-distance'].mean()
             total_network_length = branch_data['branch-distance'].sum()
             
             # Tortuosity = branch-distance / euclidean-distance
             # Handle division by zero (loops have 0 euclidean distance)
             with np.errstate(divide='ignore', invalid='ignore'):
                tortuosity = branch_data['branch-distance'] / branch_data['euclidean-distance']
             
             # Replace inf with NaN for mean calculation
             tortuosity = tortuosity.replace([np.inf, -np.inf], np.nan)
             mean_tortuosity = tortuosity.mean()
             
             if pd.isna(mean_tortuosity):
                 mean_tortuosity = 0.0
             
             # Cyclomatic number = E - N + P
             # Edges - Nodes + Connected Components
             # E = branch_count
             # N = number of unique nodes in the graph representation
             # P = Connected components of the skeleton
             
             # skan summary has 'skeleton-id' for connected components
             num_components = branch_data['skeleton-id'].nunique()
             # Nodes in graph:
             unique_nodes = set(branch_data['node-id-src']).union(set(branch_data['node-id-dst']))
             num_nodes = len(unique_nodes)
             num_edges = branch_count
             
             cyclomatic_number = num_edges - num_nodes + num_components
        else:
             junction_count = 0
             branch_count = 0
             mean_branch_length = 0.0
             total_network_length = 0.0
             mean_tortuosity = 1.0 # Line
             cyclomatic_number = 0
    except Exception as e:
        print(f"Skan error: {e}")
        junction_count = 0
        branch_count = 0
        mean_branch_length = 0.0
        total_network_length = 0.0
        mean_tortuosity = 0.0
        cyclomatic_number = 0

    # Fragment Count
    labeled_mask, num_features = measure.label(mask, return_num=True)
    fragment_count = num_features
    
    # --- Per-Fragment Sphericity ---
    # Calculate sphericity for each individual mitochondrial fragment
    fragment_sphericities = []
    fragment_volumes = []
    
    if fragment_count > 0:
        for frag_id in range(1, fragment_count + 1):
            frag_mask = (labeled_mask == frag_id).astype(np.uint8)
            frag_volume_pixels = np.sum(frag_mask)
            
            # Skip very small fragments (< 10 voxels)
            if frag_volume_pixels < 10:
                continue
                
            frag_volume = frag_volume_pixels * voxel_vol
            fragment_volumes.append(frag_volume)
            
            # Calculate surface area for this fragment
            try:
                frag_verts, frag_faces, _, _ = measure.marching_cubes(frag_mask, spacing=voxel_size)
                frag_surface = measure.mesh_surface_area(frag_verts, frag_faces)
                frag_sphericity = calculate_sphericity(frag_volume, frag_surface)
                fragment_sphericities.append(frag_sphericity)
            except (ValueError, RuntimeError):
                # Fragment too small for marching cubes
                pass
    
    # Aggregate fragment statistics
    if fragment_sphericities:
        mean_frag_sphericity = np.mean(fragment_sphericities)
        std_frag_sphericity = np.std(fragment_sphericities)
        min_frag_sphericity = np.min(fragment_sphericities)
        max_frag_sphericity = np.max(fragment_sphericities)
        mean_frag_volume = np.mean(fragment_volumes)
    else:
        mean_frag_sphericity = 0.0
        std_frag_sphericity = 0.0
        min_frag_sphericity = 0.0
        max_frag_sphericity = 0.0
        mean_frag_volume = 0.0
    
    return {
        'Volume': total_volume,
        'Surface_Area': surface_area,
        'Sphericity': sphericity,  # Whole mask sphericity
        'Fragment_Count': fragment_count,
        'Mean_Fragment_Sphericity': mean_frag_sphericity,  
        'Std_Fragment_Sphericity': std_frag_sphericity,    
        'Min_Fragment_Sphericity': min_frag_sphericity,    
        'Max_Fragment_Sphericity': max_frag_sphericity,    
        'Mean_Fragment_Volume': mean_frag_volume,          
        'Junction_Count': junction_count,
        'Branch_Count': branch_count,
        'Mean_Branch_Length': mean_branch_length,
        'Total_Network_Length': total_network_length,
        'Mean_Tortuosity': mean_tortuosity,
        'Cyclomatic_Number': cyclomatic_number
    }

def analyze_actin(mask, voxel_size):
    """
    Analyze Actin mask.
    Features: Total Volume, Skeleton Length.
    """
    # Voxel dimensions
    vx, vy, vz = voxel_size
    voxel_vol = vx * vy * vz
    
    # Total Volume
    volume_pixels = np.sum(mask > 0)
    total_volume = volume_pixels * voxel_vol
    
    # Skeleton Length
    # Skeletonize 3D
    skeleton = morphology.skeletonize(mask) # Count pixels to estimate length 
    skeleton_pixels = np.sum(skeleton > 0)
    
    
    # Skeleton Length
    skeleton = morphology.skeletonize(mask)
    skeleton_pixels = np.sum(skeleton > 0)
    
    # --- Advanced Shape Descriptors ---
    
    # Convex Hull Volume & Solidity
    points = np.argwhere(mask > 0)
    if len(points) >= 4: # Need at least 4 points for 3D hull
        try:
            # Adjust points by voxel size for correct volume
            points_real = points * np.array([vx, vy, vz])
            hull = ConvexHull(points_real)
            hull_volume = hull.volume
            solidity = total_volume / hull_volume if hull_volume > 0 else 0
        except Exception:
            hull_volume = 0.0
            solidity = 0.0
    else:
        hull_volume = 0.0
        solidity = 0.0
        
    # Extent
    # Bounding Box Volume
    if len(points) > 0:
        min_coords = np.min(points, axis=0) * np.array([vx, vy, vz])
        max_coords = np.max(points, axis=0) * np.array([vx, vy, vz])
        dims = (max_coords - min_coords) + np.array([vx, vy, vz])
        bbox_volume = np.prod(dims)
        extent = total_volume / bbox_volume if bbox_volume > 0 else 0
    else:
        bbox_volume = 0.0
        extent = 0.0
        
        
    # Principal Axis Lengths & Fractional Anisotropy
    
    mask_single_label = (mask > 0).astype(int)
    regions = measure.regionprops(mask_single_label, spacing=voxel_size)
    if regions:
        props = regions[0]
        
        try:
            # Get inertia tensor eigenvalues (sorted descending: i1 >= i2 >= i3)
            evals = np.array(props.inertia_tensor_eigvals)  # Convert to numpy array
            i1, i2, i3 = evals[0], evals[1], evals[2]
            
            # Calculate Fractional Anisotropy
            if np.sum(evals**2) > 0:
                mean_eval = np.mean(evals)
                numerator = np.sum((evals - mean_eval)**2)
                denominator = np.sum(evals**2)
                fractional_anisotropy = np.sqrt(3/2) * np.sqrt(numerator / denominator)
            else:
                fractional_anisotropy = 0.0
            
            # Calculate axis lengths from inertia tensor eigenvalues
            # For a solid ellipsoid with semi-axes a >= b >= c:
            # I1 = (1/5) * mass * (b² + c²)  <- largest moment, around shortest axis
            # I2 = (1/5) * mass * (a² + c²)
            # I3 = (1/5) * mass * (a² + b²)  <- smallest moment, around longest axis
            # Solving: a² = (5/2) * (I2 + I3 - I1), etc.
            
            # Note: For voxel shapes, we use a scaling factor
            # Semi-axis lengths (a = major, b = intermediate, c = minor)
            a_sq = 2.5 * (i2 + i3 - i1)  # Major axis squared
            b_sq = 2.5 * (i1 + i3 - i2)  # Intermediate axis squared  
            c_sq = 2.5 * (i1 + i2 - i3)  # Minor axis squared
            
            # Handle numerical issues (negative values from non-ellipsoid shapes)
            major_axis = np.sqrt(max(0, a_sq)) * 2  # Full axis length (not semi-axis)
            intermediate_axis = np.sqrt(max(0, b_sq)) * 2
            minor_axis = np.sqrt(max(0, c_sq)) * 2
            
        except Exception as e:
            import traceback
            print(f"Exception in axis calculation: {e}")
            traceback.print_exc()
            fractional_anisotropy = 0.0
            major_axis = 0.0
            intermediate_axis = 0.0
            minor_axis = 0.0
    else:
        fractional_anisotropy = 0.0
        major_axis = 0.0
        intermediate_axis = 0.0
        minor_axis = 0.0
        

    return {
        'Volume': total_volume,
        'Skeleton_Length_Pixels': skeleton_pixels,
        'Convex_Hull_Volume': hull_volume,
        'Solidity': solidity,
        'Extent': extent,
        'Fractional_Anisotropy': fractional_anisotropy,
        'Major_Axis': major_axis,
        'Intermediate_Axis': intermediate_axis,
        'Minor_Axis': minor_axis
    }

def analyze_nucleus(mask, voxel_size):
    """
    Analyze Nucleus mask.
    Features: Volume, Sphericity, Elongation.
    """
    # Voxel dimensions
    vx, vy, vz = voxel_size
    voxel_vol = vx * vy * vz
    
    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask, spacing=voxel_size)
    
    if not regions:
        return {
            'Volume': 0.0,
            'Sphericity': 0.0,
            'Elongation': 0.0
        }
        
    # Sort by size and take the largest (assumed to be the nucleus)
    regions.sort(key=lambda x: x.area, reverse=True)
    main_nucleus = regions[0]
    
    # Volume (from all pixels or just the main one? Prompt says "Nucleus (Testing Deformation)". 
    # Likely the mask contains just the nucleus.
    volume_pixels = np.sum(mask > 0)
    total_volume = volume_pixels * voxel_vol
    
    # Surface Area for Sphericity
    try:
        verts, faces, normals, values = measure.marching_cubes(mask, spacing=voxel_size)
        surface_area = measure.mesh_surface_area(verts, faces)
    except (ValueError, RuntimeError):
        surface_area = 0.0
    
    sphericity = calculate_sphericity(total_volume, surface_area)
    
    # Elongation: Major Axis / Minor Axis
    if hasattr(main_nucleus, 'major_axis_length') and hasattr(main_nucleus, 'minor_axis_length'):
       major = main_nucleus.major_axis_length
       minor = main_nucleus.minor_axis_length
       
       evals = main_nucleus.inertia_tensor_eigvals
       
       elongation = major / minor if minor > 0 else 0.0
       
       # Flatness? Needs intermediate.
       # If not available, set to 0.
       flatness = 0.0 
       
       try:
           i1, i2, i3 = evals # Descending: i1 >= i2 >= i3
           # Check for non-physicality (negative under sqrt)
           t1 = i2 + i3 - i1
           t2 = i1 + i3 - i2
           t3 = i1 + i2 - i3
           
           if t1 > 0 and t2 > 0 and t3 > 0:
               # r_min correspond to i_max (i1) -> c
               # r_mid corresponds to i_mid (i2) -> b
               # r_max corresponds to i_min (i3) -> a
               
               sorted_sq = np.sort([t1, t2, t3]) # Ascending: t_min, t_mid, t_max
               # a proportional to sqrt(t_max)
               # b proportional to sqrt(t_mid)
               # c proportional to sqrt(t_min)
               
               c_semi = np.sqrt(sorted_sq[0])
               b_semi = np.sqrt(sorted_sq[1])
               a_semi = np.sqrt(sorted_sq[2])
               
               flatness = b_semi / a_semi # Inter / Major
               elongation = a_semi / c_semi # Major / Minor
           
       except Exception:
           pass
       
    else:
       elongation = 0.0
       flatness = 0.0
       
    # Solidity (Volume / Convex Hull Volume)
    try:
        solidity = main_nucleus.solidity
    except AttributeError:
        # Fallback if not calculated by regionprops
        points = main_nucleus.coords
        if len(points) >= 4:
             hull = ConvexHull(points)
             solidity = main_nucleus.area / hull.volume if hull.volume > 0 else 0
        else:
             solidity = 0.0
    
    return {
        'Volume': total_volume,
        'Sphericity': sphericity,
        'Elongation': elongation,
        'Flatness': flatness,
        'Solidity': solidity
    }

import nibabel as nib

def main():
    parser = argparse.ArgumentParser(description='Analyze 3D segmentation masks.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing TIF or NIfTI files')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--structure', type=str, required=True, choices=['mito', 'actin', 'nucleus'], help='Structure to analyze')
    parser.add_argument('--voxel_size', type=float, nargs=3, default=None, 
                        help='Voxel size in Z Y X (µm). If not provided, reads from NIfTI header automatically.')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)
        
    results = []
    
    files = sorted(list(input_dir.glob('*.tif')) + list(input_dir.glob('*.tiff')) + list(input_dir.glob('*.nii.gz')))
    
    print(f"Found {len(files)} files in {input_dir}. structure={args.structure}")
    
    for file_path in files:
        try:
            # Load file and determine voxel size
            if file_path.name.endswith('.nii.gz'):
                nii = nib.load(file_path)
                mask = nii.get_fdata()
                
                # Auto-read voxel size from NIfTI header if not provided
                if args.voxel_size is None:
                    # NIfTI header stores voxel size in pixdim (Z, Y, X order typically)
                    zooms = nii.header.get_zooms()
                    voxel_size = list(zooms[:3])  # Get first 3 dimensions
                    print(f"  Voxel size from header: {voxel_size} µm")
                else:
                    voxel_size = args.voxel_size
            else:
                mask = tifffile.imread(file_path)
                # For TIFF files, must use provided voxel size or default
                if args.voxel_size is None:
                    voxel_size = [1.0, 1.0, 1.0]
                    print(f"  Warning: No voxel size in TIFF, using default {voxel_size}")
                else:
                    voxel_size = args.voxel_size
            
            # Ensure binary (0 and 1)
            mask = (mask > 0).astype(np.uint8)
            
            # Analyze based on structure
            if args.structure == 'mito':
                metrics = analyze_mito(mask, voxel_size)
            elif args.structure == 'actin':
                metrics = analyze_actin(mask, voxel_size)
            elif args.structure == 'nucleus':
                metrics = analyze_nucleus(mask, voxel_size)
            
            # Store voxel size used for reference
            metrics['Voxel_Z'] = voxel_size[0]
            metrics['Voxel_Y'] = voxel_size[1]
            metrics['Voxel_X'] = voxel_size[2]
            metrics['Filename'] = file_path.name
            results.append(metrics)
            print(f"Processed {file_path.name}")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        # Reorder columns to put Filename first, then voxel info
        cols = ['Filename', 'Voxel_Z', 'Voxel_Y', 'Voxel_X'] + [c for c in df.columns if c not in ['Filename', 'Voxel_Z', 'Voxel_Y', 'Voxel_X']]
        df = df[cols]
        df.to_csv(args.output_csv, index=False)
        print(f"Saved results to {args.output_csv}")
    else:
        print("No results to save.")
        
if __name__ == "__main__":
    main()
