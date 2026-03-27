import os
import sys
import argparse
import numpy as np
import pandas as pd
import tifffile
import nibabel as nib
import nrrd
from skimage import measure, morphology
from scipy.spatial import ConvexHull
from pathlib import Path

from skan import Skeleton, summarize
from skimage.measure import regionprops_table
import pandas as pd


def get_voxel_size_from_nrrd(nifti_path, raw_nrrd_dir):
    """
    Extract voxel size from original NRRD file for a given NIfTI.
    
    Args:
        nifti_path: Path to NIfTI segmentation file (str or Path)
        raw_nrrd_dir: Path to directory containing NRRD files (str or Path)
    
    Returns:
        [vz, vy, vx] in microns, or None if not found
    """
    # Extract cell name from NIfTI filename.
    # Use .name and strip both .nii.gz and .nii extensions (Path.stem only strips .gz).
    fname = Path(nifti_path).name
    for suffix in ('_segmentation.nii.gz', '.nii.gz', '_segmentation.nii', '.nii'):
        if fname.endswith(suffix):
            fname = fname[: -len(suffix)]
            break
    cell_name = fname
    nrrd_folder = Path(raw_nrrd_dir) / cell_name
    
    if nrrd_folder.exists():
        # Find a non-mask NRRD file
        nrrd_files = [f for f in nrrd_folder.glob('channel_*.nrrd') if 'mask' not in f.name]
        if nrrd_files:
            try:
                header = nrrd.read_header(str(nrrd_files[0]))
                if 'space directions' in header:
                    dirs = header['space directions']
                    spacing = [float(np.linalg.norm(d)) if hasattr(d, '__len__') else float(d) for d in dirs]
                    return spacing
                elif 'spacings' in header:
                    return [float(s) for s in header['spacings']]
            except Exception as e:
                print(f"  Warning: Could not read voxel from {nrrd_folder.name}: {e}")
    
    return None

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

def calculate_circularity(mask):
    """
    Calculate circularity from the maximum area slice of a 3D mask.
    
    Circularity = 4π × Area / Perimeter²
    
    This is more robust than 3D sphericity because it doesn't depend on 
    marching cubes mesh resolution. Circularity is always bounded [0, 1]
    where 1 is a perfect circle.
    
    Args:
        mask: 3D binary array (nucleus mask)
    
    Returns:
        float: Circularity value [0, 1]
    """
    if mask.sum() == 0:
        return 0.0
    
    max_area = 0
    max_perimeter = 0
    
    # Check all 3 axes and find the slice with maximum area
    for axis in range(3):
        for i in range(mask.shape[axis]):
            # Extract 2D slice
            if axis == 0:
                slice_2d = mask[i, :, :]
            elif axis == 1:
                slice_2d = mask[:, i, :]
            else:
                slice_2d = mask[:, :, i]
            
            if slice_2d.sum() == 0:
                continue
            
            # Get region properties
            labeled = measure.label(slice_2d)
            regions = measure.regionprops(labeled)
            
            if not regions:
                continue
            
            # Use the largest region in this slice
            main_region = max(regions, key=lambda r: r.area)
            
            area = main_region.area
            perimeter = main_region.perimeter
            
            # Track maximum area slice
            if area > max_area:
                max_area = area
                max_perimeter = perimeter
    
    # Calculate circularity for the maximum area slice
    if max_perimeter > 0:
        circularity = 4 * np.pi * max_area / (max_perimeter ** 2)
        # Clamp to valid range (should be automatic, but just in case)
        return min(1.0, max(0.0, circularity))
    
    return 0.0


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
    # Pad by 1 voxel so the z-boundary caps are closed — without padding,
    # objects touching the z edge produce an open mesh that halves surface area.
    try:
        mask_padded = np.pad(mask, 1, mode='constant')
        verts, faces, normals, values = measure.marching_cubes(mask_padded, spacing=voxel_size)
        surface_area = measure.mesh_surface_area(verts, faces)
    except (ValueError, RuntimeError):
        surface_area = 0.0

    # Sphericity
    sphericity = calculate_sphericity(total_volume, surface_area)

    # --- Advanced Network Topology (skan) ---
    try:
        from skan import Skeleton, summarize
        import networkx as nx
        
        skeleton_image = morphology.skeletonize(mask)
        if np.sum(skeleton_image) > 1: # skan needs at least 2 pixels
             # Pass voxel_size to Skeleton for correct physical spacing
             skel_obj = Skeleton(skeleton_image, spacing=voxel_size)
             branch_data = summarize(skel_obj)
             
             # Branch pruning: filter out noise
             # User feedback suggests current mean (1.26 um) is low, implying noise.
             # Increasing threshold to 1.0 microns (approx 3-5 voxels).
             MIN_BRANCH_LENGTH = 1.0  
             real_branches = branch_data[branch_data['branch-distance'] > MIN_BRANCH_LENGTH]
             
             branch_count = len(real_branches)
             mean_branch_length = real_branches['branch-distance'].mean() if branch_count > 0 else 0.0
             total_network_length = real_branches['branch-distance'].sum()
             
             # Tortuosity
             if branch_count > 0:
                 with np.errstate(divide='ignore', invalid='ignore'):
                    tortuosity = real_branches['branch-distance'] / real_branches['euclidean-distance']
                 
                 tortuosity = tortuosity.replace([np.inf, -np.inf], np.nan)
                 mean_tortuosity = tortuosity.mean()
                 if pd.isna(mean_tortuosity): mean_tortuosity = 0.0
             else:
                 mean_tortuosity = 0.0
             
             # CALCULATE TOPOLOGY FROM PRUNED GRAPH
             if branch_count > 0:
                 # Build new graph from filtered branches to correctly count components/nodes/edges
                 # Use MultiGraph to preserve parallel edges (cycles of length 2)
                 G = nx.from_pandas_edgelist(
                     real_branches, 
                     source='node-id-src', 
                     target='node-id-dst', 
                     create_using=nx.MultiGraph()
                 )
                 
                 # Recalculate metrics on the cleaned graph
                 num_edges = G.number_of_edges() # Should match branch_count
                 num_nodes = G.number_of_nodes()
                 num_components = nx.number_connected_components(G)
                 
                 # Cyclomatic number: E - N + C (Guaranteed >= 0)
                 cyclomatic_number = max(0, num_edges - num_nodes + num_components)
                 
                 # Junction count: Nodes with degree > 2
                 junction_count = sum(1 for n, d in G.degree() if d > 2)
                 
             else:
                 cyclomatic_number = 0
                 junction_count = 0
                 
        else:
             junction_count = 0
             branch_count = 0
             mean_branch_length = 0.0
             total_network_length = 0.0
             mean_tortuosity = 1.0
             cyclomatic_number = 0
             
    except Exception as e:
        print(f"Skan error: {e}")
        junction_count = 0
        branch_count = 0
        mean_branch_length = 0.0
        total_network_length = 0.0
        mean_tortuosity = 0.0
        cyclomatic_number = 0

    # Fragment Count — only count fragments >= 10 voxels (consistent with
    # Mean_Fragment_Volume which also skips fragments < 10 voxels)
    labeled_mask, num_features = measure.label(mask, return_num=True)

    # --- Per-Fragment Sphericity ---
    fragment_sphericities = []
    fragment_volumes = []
    fragment_count = 0   # counted below after applying the same size filter

    if num_features > 0:
        regions = measure.regionprops(labeled_mask)
        for region in regions:
            if region.area < 10: # Skip tiny voxel fragments (noise)
                continue
            fragment_count += 1
                
            frag_volume = region.area * voxel_vol
            fragment_volumes.append(frag_volume)
            
            # Sphericity needs mesh, which is expensive for 1000s of frags
            # We can use an approximation for tiny ones or skip.
            # Only do full mesh for decent size fragments?
            # Existing code did masking.
            # Optimized approach:
            if region.area > 20: 
               try:
                   # Crop to region bbox to save memory/speed
                   min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
                   frag_crop = mask[min_z:max_z, min_y:max_y, min_x:max_x]
                   # Only keep this label
                   frag_mask = (labeled_mask[min_z:max_z, min_y:max_y, min_x:max_x] == region.label).astype(np.uint8)
                   
                   # Add padding for marching cubes
                   frag_mask = np.pad(frag_mask, 1, mode='constant')
                   
                   verts, faces, _, _ = measure.marching_cubes(frag_mask, spacing=voxel_size)
                   frag_surface = measure.mesh_surface_area(verts, faces)
                   frag_sphericity = calculate_sphericity(frag_volume, frag_surface)
                   fragment_sphericities.append(frag_sphericity)
               except:
                   pass
    
    if fragment_sphericities:
        mean_frag_sphericity = np.mean(fragment_sphericities)
        std_frag_sphericity = np.std(fragment_sphericities)
        min_frag_sphericity = np.min(fragment_sphericities)
        max_frag_sphericity = np.max(fragment_sphericities)
    else:
        mean_frag_sphericity = 0.0
        std_frag_sphericity = 0.0
        min_frag_sphericity = 0.0
        max_frag_sphericity = 0.0
        
    mean_frag_volume = np.mean(fragment_volumes) if fragment_volumes else 0.0
    
    return {
        'Volume': total_volume,
        'Surface_Area': surface_area,
        'Sphericity': sphericity,
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
    
    # Skeleton Length - Use skan for accurate 3D physical length
    skeleton = morphology.skeletonize(mask)
    
    if np.sum(skeleton) > 1:
        # CRITICAL FIX: Use skan with spacing for diagonal correction
        from skan import Skeleton, summarize
        skel_actin = Skeleton(skeleton, spacing=voxel_size)
        actin_branch_data = summarize(skel_actin)
        
        # PRUNING: Filter noise branches for consistency with Mito checking
        MIN_BRANCH_LENGTH = 1.0
        actin_clean = actin_branch_data[actin_branch_data['branch-distance'] > MIN_BRANCH_LENGTH]
        
        skeleton_length = actin_clean['branch-distance'].sum()
    else:
        skeleton_length = 0.0
    
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
    # CRITICAL FIX: Manual PCA in PHYSICAL space (not pixel space)
    
    # Extract voxel coordinates
    coords = np.array(np.where(mask > 0)).T  # Shape: (N, 3) - [z, y, x]
    
    if len(coords) > 3:  # Need at least 4 points for PCA
        try:
            # Scale coordinates to physical space (microns)
            voxel_size_array = np.array(voxel_size)  # [vz, vy, vx]
            coords_physical = coords * voxel_size_array
            
            # Center the coordinates
            centroid = np.mean(coords_physical, axis=0)
            coords_centered = coords_physical - centroid
            
            # Compute covariance matrix in PHYSICAL space
            cov_matrix = np.cov(coords_centered.T)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            
            # Axis lengths = 2 * standard deviations along principal axes
            major_axis = 2 * np.sqrt(max(0, eigenvalues[0]))
            intermediate_axis = 2 * np.sqrt(max(0, eigenvalues[1]))
            minor_axis = 2 * np.sqrt(max(0, eigenvalues[2]))
            
            # Fractional Anisotropy
            mean_eval = np.mean(eigenvalues)
            if np.sum(eigenvalues**2) > 0:
                numerator = np.sum((eigenvalues - mean_eval)**2)
                denominator = np.sum(eigenvalues**2)
                fractional_anisotropy = np.sqrt(1.5 * numerator / denominator)
            else:
                fractional_anisotropy = 0.0
                
        except Exception as e:
            import traceback
            print(f"Exception in PCA calculation: {e}")
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
        'Skeleton_Length': skeleton_length,
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
    
    # FILTER: Exclude tiny nuclei (likely segmentation artifacts)
    if total_volume < 200.0:
        return None
        
    # Surface Area for Sphericity — pad so z-boundary caps are closed
    try:
        mask_padded = np.pad(mask, 1, mode='constant')
        verts, faces, normals, values = measure.marching_cubes(mask_padded, spacing=voxel_size)
        surface_area = measure.mesh_surface_area(verts, faces)
    except (ValueError, RuntimeError):
        surface_area = 0.0

    sphericity = calculate_sphericity(total_volume, surface_area)
    
    # Circularity (2D metric from maximum area slice - more robust than 3D sphericity)
    circularity = calculate_circularity(mask)
    
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
        'Circularity': circularity,
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
                        help='Voxel size in Z Y X (µm). Manual override.')
    parser.add_argument('--raw_nrrd_dir', type=str, default=None,
                        help='Path to original NRRD files directory (recommended for accurate voxel sizes)')
    
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
            # Initialize variables to prevent UnboundLocalError
            voxel_size = None
            voxel_source = None
            
            # Load file
            if file_path.name.endswith('.nii.gz'):
                nii = nib.load(file_path)
                mask = nii.get_fdata()
            else:
                mask = tifffile.imread(file_path)
            
            # CRITICAL FIX: Determine voxel size with proper priority chain
            
            if args.voxel_size is not None:
                # Priority 1: Manual override
                voxel_size = args.voxel_size
                voxel_source = 'Manual'
            elif args.raw_nrrd_dir is not None:
                # Priority 2: NRRD lookup (MOST ACCURATE)
                voxel_size = get_voxel_size_from_nrrd(file_path, args.raw_nrrd_dir)
                if voxel_size is not None:
                    voxel_source = 'NRRD'
                    print(f"  Voxel size from NRRD: {voxel_size} µm")
            
            # Priority 3: NIfTI header (often corrupted)
            if voxel_size is None and file_path.name.endswith('.nii.gz'):
                zooms = nii.header.get_zooms()
                voxel_size = list(zooms[:3])
                voxel_source = 'NIfTI_Header'
                print(f"  WARNING: Using NIfTI header voxel size: {voxel_size} µm (may be inaccurate)")
            
            # Priority 4: Default fallback (last resort)
            if voxel_size is None:
                voxel_size = [1.0, 1.0, 1.0]
                voxel_source = 'Default'
                print(f"  ERROR: Using default voxel size {voxel_size} µm (INACCURATE!)")
            
            # AXIS ORDER FIX:
            # The raw NRRD files are stored as (Z, Y, X) and get_voxel_size_from_nrrd
            # returns [vz, vy, vx] matching that order.
            # However, nnU-Net's NIfTI output reverses the axes to (X, Y, Z).
            # Confirmed: NRRD shape (18, 930, 964) → NIfTI shape (964, 930, 18).
            # So for NIfTI files we must reverse the voxel_size to [vx, vy, vz]
            # so that spacing axis 0 matches X, axis 1 matches Y, axis 2 matches Z.
            if file_path.name.endswith('.nii.gz') and voxel_source in ('NRRD', 'Manual'):
                voxel_size = [voxel_size[2], voxel_size[1], voxel_size[0]]
                print(f"  Voxel size reordered for NIfTI (X,Y,Z): {voxel_size} µm")

            # Ensure binary (0 and 1)
            mask = (mask > 0).astype(np.uint8)

            # Binary closing: Fill small holes to improve surface area / sphericity
            from scipy.ndimage import binary_closing
            mask = binary_closing(mask, structure=np.ones((3, 3, 3))).astype(np.uint8)
            
            # Analyze based on structure
            if args.structure == 'mito':
                metrics = analyze_mito(mask, voxel_size)
            elif args.structure == 'actin':
                metrics = analyze_actin(mask, voxel_size)
            elif args.structure == 'nucleus':
                metrics = analyze_nucleus(mask, voxel_size)
            
            # If metrics is None (quality control failure), skip
            if metrics is None:
                print(f"  Skipping {file_path.name}: QC Failure (e.g. too small)")
                continue
            
            # Store voxel size and source for auditing
            metrics['Voxel_Z'] = voxel_size[2]
            metrics['Voxel_Y'] = voxel_size[1]
            metrics['Voxel_X'] = voxel_size[0]
            metrics['Voxel_Source'] = voxel_source
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
        cols = ['Filename', 'Voxel_Source', 'Voxel_Z', 'Voxel_Y', 'Voxel_X'] + [c for c in df.columns if c not in ['Filename', 'Voxel_Source', 'Voxel_Z', 'Voxel_Y', 'Voxel_X']]
        df = df[cols]
        df.to_csv(args.output_csv, index=False)
        print(f"Saved results to {args.output_csv}")
    else:
        print("No results to save.")
        
if __name__ == "__main__":
    main()
