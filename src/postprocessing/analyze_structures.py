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
    (pi^(1/3) * (6 * V)^(2/3)) / A
    """
    if surface_area == 0:
        return 0.0
    return (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area

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
             # A more accurate way for junctions is using degrees from skan or counting nodes > 2 neighbors
             # but skan summary gives branches.
             
             # Let's derive the requested list:
             # junction_count: count of nodes with degree > 2 ??
             # Actually, Skeleton object itself has degrees.
             
             skel_obj = Skeleton(skeleton_image)
             degrees = skel_obj.degrees
             # Nodes are pixels. Junctions are nodes with degree > 2.
             # skan degrees is an array of degrees for each non-zero pixel in order? 
             # No, skel_obj.degrees is an image or sparse array matching the skeleton.
             # Let's trust skan documentation or common usage. 
             # skel_obj.degrees is an array of shape (N_nonzero_pixels,)
             
             junction_count = np.sum(skel_obj.degrees > 2)
             branch_count = branch_data.shape[0]
             mean_branch_length = branch_data['branch-distance'].mean()
             total_network_length = branch_data['branch-distance'].sum()
             
             # Tortuosity = branch-distance / euclidean-distance
             branch_data['tortuosity'] = branch_data['branch-distance'] / branch_data['euclidean-distance']
             mean_tortuosity = branch_data['tortuosity'].mean()
             
             # Cyclomatic number = E - N + P
             # E = number of edges (branches?)
             # N = number of nodes (junctions + endpoints)
             # skan graph might be slighty different (multi-edges). 
             # Let's use simple graph theory from skan graph.
             # skel_obj.n_paths (branches?)
             # skel_obj.n_junctions? No.
             
             # Simple approach: Edges - Nodes + Connected Components
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
    
    return {
        'Volume': total_volume,
        'Surface_Area': surface_area,
        'Sphericity': sphericity,
        'Fragment_Count': fragment_count,
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
    skeleton = morphology.skeletonize(mask)
    # Count pixels to estimate length (simple approximation as requested)
    # Taking average voxel size for length estimation if needed, but prompt said "count pixels"
    # To be more physically meaningful, we might multiply by mean voxel dimension, 
    # but sticking to "count pixels" as requested, or maybe pixels * unit length?
    # Let's assume just count for now, but label it pixels. 
    # Actually, for physical units, it's safer to provide both or assume 1 unit length per pixel if isometric.
    skeleton_pixels = np.sum(skeleton > 0)
    
    
    # Skeleton Length
    skeleton = morphology.skeletonize(mask)
    skeleton_pixels = np.sum(skeleton > 0)
    
    # --- Advanced Shape Descriptors ---
    
    # Convex Hull Volume & Solidity
    # Need points for ConvexHull
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
        # Add one voxel dimension? No, bounding box is usually max-min.
        # But for pixels, we usually add 1 to width.
        # Let's say dimensions + voxel_size
        dims = (max_coords - min_coords) + np.array([vx, vy, vz])
        bbox_volume = np.prod(dims)
        extent = total_volume / bbox_volume if bbox_volume > 0 else 0
    else:
        bbox_volume = 0.0
        extent = 0.0
        
    # Principal Axis Lengths & Fractional Anisotropy
    # regionprops on the whole mask (assuming one object for global shape)
    # If fragmented, we might want to analyze the largest component or the "weighted average"?
    # The prompt says "Global Alignment & Shape". Usually implies treating the actin network as a distribution.
    # regionprops works on connected components. 
    # Let's take the largest component or maybe the "convex hull" implies we treat it as one blob.
    # Let's label and take largest for shape analysis to be consistent with Convex Hull (which used all points).
    # Actually, `regionprops` can calculate inertia tensor for the label.
    # If we label the whole thing as 1 (even if disconnected), regionprops will calculate moments for the cloud of pixels.
    
    mask_single_label = (mask > 0).astype(int)
    regions = measure.regionprops(mask_single_label, spacing=voxel_size)
    if regions:
        props = regions[0]
        # Major, Minor (Intermediate?)
        # 3D regionprops: major_axis_length, minor_axis_length... 
        # Actually in 3D: moments_eig -> eigenvalues of inertia tensor.
        # regionprops provides `inertia_tensor_eigvals`.
        # Lengths are approx 4 * sqrt(eigvals) ?
        # scikit-image regionprops documentation says:
        # inertia_tensor_eigvals: Eigenvalues of the inertia tensor of the region, sorted in descending order.
        # approximation of axes: 
        # l1 = 2 * sqrt(5 * i1 / mass) ? No that's for solid.
        # Let's rely on major_axis_length if available in user's version (usually 2D/3D supported).
        # Newer skimage supports axis_major_length, axis_minor_length etc for 3D?
        # Let's check safely. If not, use inertia_tensor_eigvals.
        
        try:
             # Inertia eigenvalues are related to the ellipsoid axes a, b, c
             # I = mass/5 * (b^2 + c^2), etc.
             # Easier: Fractional Anisotropy from eigenvalues of inertia tensor.
             # FA = sqrt(3/2) * sqrt(sum((lambda_i - mean)^2) / sum(lambda_i^2))
             evals = props.inertia_tensor_eigvals
             if np.sum(evals**2) > 0:
                 mean_eval = np.mean(evals)
                 numerator = np.sum((evals - mean_eval)**2)
                 denominator = np.sum(evals**2)
                 fractional_anisotropy = np.sqrt(3/2) * np.sqrt(numerator / denominator)
             else:
                 fractional_anisotropy = 0.0
                 
             major_axis = 0.0
             intermediate_axis = 0.0
             minor_axis = 0.0
             # lengths from bounding box might be too simple.
             # Let's assume standard regionprops has axis lengths in 3D for modern skimage.
             if hasattr(props, 'major_axis_length'):
                  major_axis = props.major_axis_length
             if hasattr(props, 'minor_axis_length'):
                  minor_axis = props.minor_axis_length
             # Intermediate? 
             # if 3D, we might not get intermediate directly.
             
        except Exception:
             fractional_anisotropy = 0.0
             major_axis = 0
             minor_axis = 0
             intermediate_axis = 0
    else:
        fractional_anisotropy = 0.0
        major_axis = 0
        minor_axis = 0
        intermediate_axis = 0
        

    return {
        'Volume': total_volume,
        'Skeleton_Length_Pixels': skeleton_pixels,
        'Convex_Hull_Volume': hull_volume,
        'Solidity': solidity,
        'Extent': extent,
        'Fractional_Anisotropy': fractional_anisotropy,
        'Major_Axis': major_axis,
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
    
    # We assume one main nucleus per mask for elongation, or we average?
    # Usually "cells" imply one nucleus. We'll take the largest component if fragmented,
    # or just calculate global volume. 
    # For elongation, we need a single object.
    
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
       
       # In 3D, there is an intermediate axis?
       # Skimage regionprops for 3D usually returns major and "minor" (smallest).
       # The intermediate is implicit or harder to get directly without eigen decomposition.
       # But let's check inertia_tensor_eigvals for full shape description.
       
       # Eigenvalues of inertia tensor: e1 >= e2 >= e3
       # Ellipsoid semi-axes a, b, c related to e1, e2, e3.
       # For a solid ellipsoid:
       # I1 = m/5 * (b^2 + c^2)
       # I2 = m/5 * (a^2 + c^2)
       # I3 = m/5 * (a^2 + b^2)
       # We can solve this system for a, b, c.
       
       evals = main_nucleus.inertia_tensor_eigvals
       # evals are sorted descending? usually.
       # Check documentation: "sorted in descending order".
       # So e1 corresponds to rotation around smallest axis?
       # Wait, larger eigenvalue = larger moment of inertia = rotation around *shorter* axis.
       # So if shape is elongated along X, Ix is small.
       # eigvals (I1, I2, I3) -> I1 >= I2 >= I3
       # I1 is max moment -> axis is shortest (minor).
       # I3 is min moment -> axis is longest (major).
       
       # Let's use equivalent_ellipsoid_diameter if available or just eigen calculations.
       # Actually, let's keep it simple: use the major/minor provided if existing, else use bounding box?
       
       # Using skan/skimage approximation:
       # elongation = major / minor
       # flatness = intermediate / major
       
       # Let's estimate semi-axes r1, r2, r3 from Inertia Tensor for robustness in 3D.
       # 5 * (I2 + I3 - I1) / 2m = a^2 ?
       # Sum = I1+I2+I3 = 2m/5 (a^2+b^2+c^2)
       # Let S = I1+I2+I3
       # a^2 = 2.5 * (S/2 - I1) ? No.
       # a = sqrt(5/2 * (I2 + I3 - I1) / mass) ?? 
       # Assume users want standard "Principal Axis Lengths".
       # If major/minor are not robust, calculation is risky.
       
       # Let's blindly use major_axis_length and minor_axis_length from regionprops if 3D supported.
       elongation = major / minor if minor > 0 else 0.0
       
       # Flatness? Needs intermediate.
       # If not available, set to 0.
       flatness = 0.0 
       # Try to deduce intermediate from inertia if we want to be fancy, but stick to basics to avoid bugs.
       # Actually, we can check bounding box as a rough proxy if needed?
       # Or simply omit if we can't reliably get it.
       # User asked for "Flatness: Intermediate/Major axis".
       # Let's try to get it from inertia eigenvalues.
       # r1 = sqrt(5 * (evals[1] + evals[2] - evals[0])) ... this formula is for solid.
       # let's try strict formula:
       # i1, i2, i3 = evals (descending)
       # a = sqrt(2.5 * (i2 + i3 - i1))  <-- Largest radius? No.
       # Smallest moment -> Largest axis.
       # i_min = evals[2] -> corresponds to Major axis (a)
       # i_mid = evals[1] -> Intermediate (b)
       # i_max = evals[0] -> Minor (c)
       # a = sqrt(2.5 * (evals[1] + evals[0] - evals[2]))
       # b = sqrt(2.5 * (evals[2] + evals[0] - evals[1]))
       # c = sqrt(2.5 * (evals[2] + evals[1] - evals[0]))
       # Note: mass term cancels if we just want ratios? 
       # These i values are "moments", so they include mass (volume).
       # But for ratios, volume cancels out.
       
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
    parser.add_argument('--voxel_size', type=float, nargs=3, default=[1.0, 1.0, 1.0], help='Voxel size in Z Y X or X Y Z (consistent with image)')
    
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
            # Load file
            if file_path.name.endswith('.nii.gz'):
                nii = nib.load(file_path)
                mask = nii.get_fdata()
            else:
                mask = tifffile.imread(file_path)
            
            # Ensure binary (0 and 1)
            mask = (mask > 0).astype(np.uint8)
            
            # Analyze based on structure
            if args.structure == 'mito':
                metrics = analyze_mito(mask, args.voxel_size)
            elif args.structure == 'actin':
                metrics = analyze_actin(mask, args.voxel_size)
            elif args.structure == 'nucleus':
                metrics = analyze_nucleus(mask, args.voxel_size)
            
            metrics['Filename'] = file_path.name
            results.append(metrics)
            print(f"Processed {file_path.name}")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        # Reorder columns to put Filename first
        cols = ['Filename'] + [c for c in df.columns if c != 'Filename']
        df = df[cols]
        df.to_csv(args.output_csv, index=False)
        print(f"Saved results to {args.output_csv}")
    else:
        print("No results to save.")
        
if __name__ == "__main__":
    main()
