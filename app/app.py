import os
import io
import time
import base64
import shutil
import subprocess
import numpy as np
import nibabel as nib
from PIL import Image
from flask import Flask, render_template, request, jsonify
from aicsimageio import AICSImage
from skimage import measure  

app = Flask(__name__)

# State to hold the loaded image in memory
viewer_state = {
    "data": None,       # numpy array shape (C, Z, Y, X)
    "channels": 0,
    "slices": 0,
    "width": 0,
    "height": 0,
    "masks": {}         # Dictionary to hold masks: key=channel_idx, val=numpy_mask
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/viewer')
def viewer():
    return render_template('viewer.html')

@app.route('/segmentation')
def segmentation():
    return render_template('segmentation.html')

@app.route('/mesh/<int:c>')
def get_mesh(c):
    """
    Returns 3D mesh data (vertices, faces) for a given channel mask.
    Uses Marching Cubes algorithm.
    """
    print(f"[DEBUG] Request for mesh channel {c}")
    
    if c not in viewer_state["masks"]:
        print(f"[ERROR] No mask found for channel {c}")
        return jsonify({"error": "No mask found for this channel"})
    
    try:
        mask = viewer_state["masks"][c]
        print(f"[DEBUG] Mask shape: {mask.shape}, Max Val: {mask.max()}, Min Val: {mask.min()}")

        if mask.max() == 0:
            print(f"[WARN] Mask is empty (all zeros) for channel {c}")
            return jsonify({"error": "Mask is empty (no segmentation found)"})
        
        # Optimize: Downsample big masks (2x) to prevent browser crash
        step = 2 
        mask_small = mask[::step, ::step, ::step]
        
        # Marching Cubes: Create surface mesh
        # Level 0.5 means the boundary between 0 (background) and 1 (mask)
        verts, faces, normals, values = measure.marching_cubes(mask_small, level=0.5)
        
        # Scale vertices back to original size
        verts = verts * step
        
        print(f"[SUCCESS] Generated mesh with {len(verts)} vertices and {len(faces)} faces")
        
        return jsonify({
            "vertices": verts.tolist(),
            "faces": faces.tolist(),
            "status": "success"
        })
        
    except Exception as e:
        print(f"[ERROR] Mesh generation failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


# ----------------------------------------
# Viewer Backend (API Routes)
# ----------------------------------------

@app.route("/load", methods=["POST"])
def load_oir():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"})

    try:
        ext = os.path.splitext(file.filename)[1] or ".oir"
        temp_file = "/tmp/temp_oir_upload" + ext
        file.save(temp_file)

        print(f"[INFO] Loading file: {file.filename}")

        # Read with AICSImage (uses BioFormats for .oir)
        img = AICSImage(temp_file)

        # Get data in CZYX order
        data = img.get_image_data("CZYX")
        data = data.astype(np.float32)

        raw_shape = str(data.shape)
        c, z, h, w = data.shape
        print(f"[INFO] Shape (C,Z,Y,X): ({c}, {z}, {h}, {w})")

        # Store in global state
        viewer_state["data"] = data
        viewer_state["channels"] = c
        viewer_state["slices"] = z
        viewer_state["width"] = w
        viewer_state["height"] = h

        return jsonify({
            "channels": c,
            "slices": z,
            "width": w,
            "height": h,
            "raw_shape": raw_shape,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/slice/<int:c>/<int:z>")
def get_slice(c, z):
    data = viewer_state["data"]
    if data is None:
        return jsonify({"error": "No data loaded"})

    try:
        if c < 0 or c >= viewer_state["channels"]:
            return jsonify({"error": "Invalid channel"})
        if z < 0 or z >= viewer_state["slices"]:
            return jsonify({"error": "Invalid slice"})

        sl = data[c, z]

        # Simple Auto-Scaling: map min..max to 0..255 for display
        vmin, vmax = sl.min(), sl.max()
        
        # Avoid division by zero
        if vmax - vmin < 1e-6:
            disp = np.zeros_like(sl, dtype=np.uint8)
        else:
            disp = ((sl - vmin) / (vmax - vmin) * 255).astype(np.uint8)

        # Create blank Red, Green, Blue channels
        zeros = np.zeros_like(disp)
        
        # Color Mapping (Fiji Style)
        if c == 0:   # Channel 1 (Nucleus) -> Blue
            rgb = np.dstack((zeros, zeros, disp))
        elif c == 1: # Channel 2 (Actin) -> Green
            rgb = np.dstack((zeros, disp, zeros))
        elif c == 2: # Channel 3 (Mito) -> Red
            rgb = np.dstack((disp, zeros, zeros))
        else:        # Fallback for other channels -> Grayscale
            rgb = np.dstack((disp, disp, disp))

        # Check for Segmentation Mask
        if c in viewer_state["masks"]:
            mask_vol = viewer_state["masks"][c]
            
            # DEBUG 
            print(f"[DEBUG] Slice z={z}: Image Shape={sl.shape}, Mask Vol Shape={mask_vol.shape}")
            
            # Ensure mask exists for this slice (Z dimension is index 0)
            if z < mask_vol.shape[0]: 
                mask_slice = mask_vol[z]
                
                # Check shapes match DEBUG
                if mask_slice.shape != sl.shape:
                    print(f"[ERROR] Shape Mismatch! Img {sl.shape} vs Mask {mask_slice.shape}")
                else:
                    # Where mask is 1, overlay Yellow (approximate)
                    should_overlay = (mask_slice > 0)
                    
                    if np.any(should_overlay):
                         print(f"[DEBUG] Overlaying {np.sum(should_overlay)} pixels")
                    
                    # We blend directly into the RGB array
                    # Yellow = Red + Green
                    rgb[:,:,0] = np.where(should_overlay, 255, rgb[:,:,0]) # Red channel -> Max
                    rgb[:,:,1] = np.where(should_overlay, 255, rgb[:,:,1]) # Green channel -> Max
                    # Blue channel -> Keep or Dim? Let's dim it to make yellow pop
                    rgb[:,:,2] = np.where(should_overlay, 0, rgb[:,:,2])

        # Create the image from the COLOR array
        pil_img = Image.fromarray(rgb.astype(np.uint8))
        
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")

        return jsonify({
            "image": base64.b64encode(buf.getvalue()).decode(),
            "pixels": sl.astype(int).flatten().tolist()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

# ----------------------------------------
# Segmentation Helper Functions
# ----------------------------------------

def save_nifti_for_nnunet(data_channel, filepath):
    """Saves a 3D numpy array (Z, Y, X) as NIfTI (X, Y, Z)"""
    data_transposed = data_channel.transpose(2, 1, 0)
    img_nifti = nib.Nifti1Image(data_transposed, affine=np.eye(4))
    nib.save(img_nifti, filepath)

def load_nifti_mask(filepath):
    """Loads NIfTI mask and converts back to Z, Y, X"""
    if not os.path.exists(filepath): return None
    img = nib.load(filepath)
    return img.get_fdata().transpose(2, 1, 0)

def run_nnunet_predict(input_folder, output_folder, task_id):
    """Runs nnUNetv2_predict command """
    
    # Define the environment variables needed by nnU-Net
    # (Based on your aneurysm_project paths)
    env = os.environ.copy()
    env["nnUNet_raw"] = "/home/luisa/aneurysm_project/data/nnUNet/nnUNet_raw"
    env["nnUNet_preprocessed"] = "/home/luisa/aneurysm_project/data/nnUNet/nnUNet_preprocessed"
    env["nnUNet_results"] = "/home/luisa/aneurysm_project/data/nnUNet/nnUNet_results"

    # Use ABSOLUTE PATH to the executable in the project venv
    nnunet_exe = "/home/luisa/aneurysm_project/venv/bin/nnUNetv2_predict"

    cmd = [
        nnunet_exe,
        "-i", input_folder,
        "-o", output_folder,
        "-d", str(task_id),   # Dataset ID
        "-c", "3d_fullres",   # Configuration
        "-f", "0"             # Fold 0
    ]
    
    print(f"[INFO] Running {nnunet_exe} for Task {task_id}...")
    # Run in the specific environment
    subprocess.run(cmd, check=True, env=env)

@app.route('/run_segmentation', methods=['POST'])
def run_segmentation():
    if viewer_state["data"] is None:
        return jsonify({"error": "No data loaded"})

    try:
        print("[INFO] Starting Segmentation Pipeline...")
        data = viewer_state["data"]
        results = {}
        
        # Temp folders
        base_tmp = "/tmp/oir_segmentation"
        if os.path.exists(base_tmp): shutil.rmtree(base_tmp)
        os.makedirs(base_tmp, exist_ok=True)
        
        # Mapping: Channel Index -> nnU-Net Dataset ID
        # CONFIRMED IDs:
        # Channel 0 (Nucleus) -> Dataset 003
        # Channel 1 (Actin)   -> Dataset 001
        # Channel 2 (Mito)    -> Dataset 002
        tasks = {
            0: 3,  # Nucleus
            1: 1,  # Actin
            2: 2   # Mito
        }

        for ch_idx, task_id in tasks.items():
            if ch_idx >= data.shape[0]: continue
            
            print(f"[INFO] Processing Channel {ch_idx} (Task {task_id})")
            
            inp_dir = os.path.join(base_tmp, f"task{task_id}_in")
            out_dir = os.path.join(base_tmp, f"task{task_id}_out")
            os.makedirs(inp_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            
            # Save Input
            save_nifti_for_nnunet(data[ch_idx], os.path.join(inp_dir, "case_000_0000.nii.gz"))
            
            try:
                # RUN REAL AI
                run_nnunet_predict(inp_dir, out_dir, task_id)
                
                # Load Result
                output_filename = os.path.join(out_dir, "case_000.nii.gz")
                mask = load_nifti_mask(output_filename)
                
                if mask is not None:
                    viewer_state["masks"][ch_idx] = mask
                    results[ch_idx] = "Success"
                else:
                    results[ch_idx] = "Failed (No Output File)"
                
            except Exception as e:
                print(f"[ERROR] Task {task_id} failed: {e}")
                results[ch_idx] = f"Error: {e}"

        return jsonify({"status": "success", "results": results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("OIR Analysis Platform")
    print("=" * 50)
    print("\nOpen: http://localhost:5001")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)
