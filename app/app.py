import os
import io
import time
import base64
import threading
import shutil
import subprocess
import numpy as np
import nibabel as nib
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from aicsimageio import AICSImage
from skimage import measure
import scipy.ndimage as ndimage
import zipfile
import numpy as np
import nibabel as nib

# --- Add Project Root to Path for Imports ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import pandas as pd

# --- Import Feature Extractors ---
# We wrap this in try-except in case of dependency issues
try:
    from src.postprocessing.analyze_structures import analyze_nucleus, analyze_actin, analyze_mito
    print("[INFO] Successfully imported analysis modules.")
except ImportError as e:
    print(f"[WARN] Could not import analysis modules: {e}")
    # Define dummy functions so app doesn't crash if imports fail
    def analyze_nucleus(*args): return {}
    def analyze_actin(*args): return {}
    def analyze_mito(*args): return {}

app = Flask(__name__)

# State to hold the loaded image in memory
viewer_state = {
    "data": None,       # numpy array shape (C, Z, Y, X)
    "channels": 0,
    "slices": 0,
    "width": 0,
    "height": 0,
    "spacing": (1.0, 1.0, 1.0), # (Z, Y, X) physical size
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

@app.route('/classifier')
def classifier_page():
    return render_template('classifier.html')

@app.route('/api/classify_subtypes', methods=['POST'])
def classify_subtypes():
    """Run TAV-ATAA vs BAV-ATAA classifier on current cell's features."""
    if viewer_state["data"] is None:
        return jsonify({"error": "No cell loaded. Go to Segmentation first."})
    if not viewer_state["masks"]:
        return jsonify({"error": "No segmentation masks. Run segmentation first."})

    try:
        spacing = viewer_state["spacing"]
        features = {}

        if 0 in viewer_state["masks"]:
            nuc = analyze_nucleus(viewer_state["masks"][0], spacing)
            if nuc:
                features["Nucleus_Volume_µm³"]        = nuc.get('Volume', 0)
                features["Nucleus_Sphericity_ratio"]   = nuc.get('Sphericity', 0)
                features["Nucleus_Elongation_ratio"]   = nuc.get('Elongation', 0)
                features["Nucleus_Flatness_ratio"]     = nuc.get('Flatness', 0)
                features["Nucleus_Solidity_ratio"]     = nuc.get('Solidity', 0)
                features["Nucleus_Circularity_ratio"]  = nuc.get('Circularity', 0)

        if 1 in viewer_state["masks"]:
            act = analyze_actin(viewer_state["masks"][1], spacing)
            if act:
                features["Actin_Volume_µm³"]                   = act.get('Volume', 0)
                features["Actin_Skeleton_Length_µm"]           = act.get('Skeleton_Length', 0)
                features["Actin_Convex_Hull_Volume_µm³"]       = act.get('Convex_Hull_Volume', 0)
                features["Actin_Solidity_ratio"]               = act.get('Solidity', 0)
                features["Actin_Extent_ratio"]                 = act.get('Extent', 0)
                features["Actin_Fractional_Anisotropy_ratio"]  = act.get('Fractional_Anisotropy', 0)
                features["Actin_Major_Axis_µm"]                = act.get('Major_Axis', 0)
                features["Actin_Intermediate_Axis_µm"]         = act.get('Intermediate_Axis', 0)
                features["Actin_Minor_Axis_µm"]                = act.get('Minor_Axis', 0)

        if 2 in viewer_state["masks"]:
            mit = analyze_mito(viewer_state["masks"][2], spacing)
            if mit:
                features["Mito_Volume_µm³"]                    = mit.get('Volume', 0)
                features["Mito_Surface_Area_µm²"]              = mit.get('Surface_Area', 0)
                features["Mito_Sphericity_ratio"]              = mit.get('Sphericity', 0)
                features["Mito_Fragment_Count_n"]              = mit.get('Fragment_Count', 0)
                features["Mito_Mean_Fragment_Sphericity_ratio"]= mit.get('Mean_Fragment_Sphericity', 0)
                features["Mito_Std_Fragment_Sphericity_ratio"] = mit.get('Std_Fragment_Sphericity', 0)
                features["Mito_Min_Fragment_Sphericity_ratio"] = mit.get('Min_Fragment_Sphericity', 0)
                features["Mito_Max_Fragment_Sphericity_ratio"] = mit.get('Max_Fragment_Sphericity', 0)
                features["Mito_Mean_Fragment_Volume_µm³"]      = mit.get('Mean_Fragment_Volume', 0)
                features["Mito_Junction_Count_n"]              = mit.get('Junction_Count', 0)
                features["Mito_Branch_Count_n"]                = mit.get('Branch_Count', 0)
                features["Mito_Mean_Branch_Length_µm"]         = mit.get('Mean_Branch_Length', 0)
                features["Mito_Total_Network_Length_µm"]       = mit.get('Total_Network_Length', 0)
                features["Mito_Mean_Tortuosity_ratio"]         = mit.get('Mean_Tortuosity', 0)
                features["Mito_Cyclomatic_Number_n"]           = mit.get('Cyclomatic_Number', 0)

        model_path = os.path.join(os.path.dirname(__file__), 'models', 'tav_bav_classifier.joblib')
        if not os.path.exists(model_path):
            return jsonify({"error": "TAV/BAV classifier model not found."})

        bundle    = joblib.load(model_path)
        pipe      = bundle['pipeline']
        feat_list = bundle['features']
        labels    = bundle['label_names']

        df_pred = pd.DataFrame([features])
        for col in feat_list:
            if col not in df_pred.columns:
                df_pred[col] = 0.0
        df_pred = df_pred[feat_list].fillna(0)

        probas    = pipe.predict_proba(df_pred)[0]
        # align probas with label_names order
        clf_classes = list(pipe.classes_)
        prob_dict = {cls: float(probas[i]) for i, cls in enumerate(clf_classes)}
        pred      = max(prob_dict, key=prob_dict.get)
        confidence= prob_dict[pred]

        return jsonify({
            "status":     "success",
            "prediction": pred,
            "confidence": round(confidence * 100, 1),
            "probabilities": {
                "TAV-ATAA": round(prob_dict.get("TAV-ATAA", 0) * 100, 1),
                "BAV-ATAA": round(prob_dict.get("BAV-ATAA", 0) * 100, 1),
            },
            "features": {k: round(v, 4) for k, v in features.items()},
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)})

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

        mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        # Optimize: Downsample big masks (2x) to prevent browser crash
        step = 2 
        mask_small = mask[::step, ::step, ::step]
        
        # Marching Cubes: Create surface mesh
        # Level 0.5 means the boundary between 0 (background) and 1 (mask)
        verts, faces, normals, values = measure.marching_cubes(mask_small, level=0.5)
        
        # Scale vertices back to original size (undo downsampling)
        verts = verts * step

        # APPLY PHYSICAL SCALING
        # verts columns are (Axis0, Axis1, Axis2) corresponding to (Z, Y, X) of the mask
        spacing = viewer_state["spacing"] # (Z, Y, X)
        
        # Apply scaling
        verts[:, 0] *= spacing[0] # Scale Z
        verts[:, 1] *= spacing[1] # Scale Y
        verts[:, 2] *= spacing[2] # Scale X

        # REORDER FOR THREE.JS (X, Y, Z)
        # Currently it is (Z, Y, X). 
        # We want Three.js X = Image X, Three.js Y = Image Y, Three.js Z = Image Z
        # So we swap columns: (Z, Y, X) -> (X, Y, Z)
        # New Column 0 = Old Column 2
        # New Column 1 = Old Column 1
        # New Column 2 = Old Column 0
        verts_xyz = np.zeros_like(verts)
        verts_xyz[:, 0] = verts[:, 2] # X
        verts_xyz[:, 1] = verts[:, 1] # Y
        verts_xyz[:, 2] = verts[:, 0] # Z
        
        print(f"[SUCCESS] Generated mesh with {len(verts)} vertices. Spacing applied: {spacing}")
        
        return jsonify({
            "vertices": verts_xyz.tolist(),
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

        import platform as _platform
        if _platform.machine() == 'arm64':
            # Mac ARM64: JPype crashes → use Java subprocess with BioFormats JARs
            import glob as _glob
            _bf_candidates = _glob.glob(os.path.expanduser("~/.jgo/ome/formats-gpl/LATEST/*/"))
            if not _bf_candidates:
                raise RuntimeError(
                    "BioFormats JARs not found. Run: "
                    "pip install bioformats_jar && python -c 'import bioformats_jar'"
                )
            BF_DIR = _bf_candidates[0].rstrip('/')
            JAVA_CLS = os.path.join(os.path.dirname(__file__), "java")
            classpath = f"{BF_DIR}/*:{JAVA_CLS}"
            out_bin = "/tmp/oir_data.bin"

            result = subprocess.run(
                ["java", "-cp", classpath, "OirReader", temp_file, out_bin],
                capture_output=True, text=True, timeout=120
            )
            ok_line = [l for l in result.stdout.strip().splitlines() if l.startswith("OK")]
            if not ok_line:
                raise RuntimeError(f"OirReader failed: {result.stderr[-500:]}")
            parts = ok_line[-1].split()
            c, z, h, w = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            pz, py, px = float(parts[5]), float(parts[6]), float(parts[7])

            with open(out_bin, "rb") as f:
                f.seek(20 + 24)
                raw = np.frombuffer(f.read(), dtype=np.uint16)
            data = raw.reshape(c, z, h, w).astype(np.float32)
        else:
            # Linux x86_64: AICSImage + bioformats_jar works fine via jpype
            from aicsimageio import AICSImage
            img = AICSImage(temp_file)
            data = img.get_image_data("CZYX", S=0, T=0).astype(np.float32)
            c, z, h, w = data.shape
            ps = img.physical_pixel_sizes
            pz = ps.Z if ps.Z else 1.0
            py = ps.Y if ps.Y else 1.0
            px = ps.X if ps.X else 1.0
        raw_shape = str(data.shape)
        print(f"[INFO] Shape (C,Z,Y,X): ({c}, {z}, {h}, {w})")

        viewer_state["data"] = data
        viewer_state["channels"] = c
        viewer_state["slices"] = z
        viewer_state["width"] = w
        viewer_state["height"] = h
        viewer_state["masks"] = {}
        viewer_state["spacing"] = (pz if pz > 0 else 1.0, py if py > 0 else 1.0, px if px > 0 else 1.0)
        print(f"[INFO] Physical Spacing (Z, Y, X): {viewer_state['spacing']}")

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
    # This function returns a base64 PNG of the current slice for the frontend
    
    data = viewer_state["data"]
    if data is None: return jsonify({"error": "No data loaded"})

    # Helper: Normalize 0..255
    def normalize(sl):
        vmin, vmax = sl.min(), sl.max()
        if vmax - vmin < 1e-6: return np.zeros_like(sl, dtype=np.uint8)
        return ((sl - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    try:
        h, w = viewer_state["height"], viewer_state["width"]
        if z < 0 or z >= viewer_state["slices"]: return jsonify({"error": "Invalid slice"})

        # Initialize RGB Canvas (Height, Width, 3)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        # --- Mode 1: Composite View (All 3 Ch + Masks) ---
        if c == 99:
            # Add Blue Channel (Nucleus - Ch 0)
            if 0 < viewer_state["channels"]: 
                rgb[:,:,2] += normalize(data[0, z])
            # Add Green Channel (Actin - Ch 1)
            if 1 < viewer_state["channels"]: 
                rgb[:,:,1] += normalize(data[1, z])
            # Add Red Channel (Mito - Ch 2)
            if 2 < viewer_state["channels"]: 
                rgb[:,:,0] += normalize(data[2, z])
            
            # Overlay Masks (Semi-transparent)
            # Nucleus Mask (0) -> Cyan (0, 255, 255)
            if 0 in viewer_state["masks"] and z < viewer_state["masks"][0].shape[0]:
                m = viewer_state["masks"][0][z] > 0
                rgb[m] = rgb[m]*0.5 + np.array([0, 255, 255])*0.5
            
            # Actin Mask (1) -> Yellow (255, 255, 0)
            if 1 in viewer_state["masks"] and z < viewer_state["masks"][1].shape[0]:
                m = viewer_state["masks"][1][z] > 0
                rgb[m] = rgb[m]*0.5 + np.array([255, 255, 0])*0.5
                
            # Mito Mask (2) -> Magenta (255, 0, 255)
            if 2 in viewer_state["masks"] and z < viewer_state["masks"][2].shape[0]:
                m = viewer_state["masks"][2][z] > 0
                rgb[m] = rgb[m]*0.5 + np.array([255, 0, 255])*0.5
        
        # --- Mode 2: Single Channel View ---
        else:
            if c < 0 or c >= viewer_state["channels"]: return jsonify({"error": "Invalid channel"})
            
            sl = data[c, z]
            norm_sl = normalize(sl)
            
            # Apply Color
            if c == 0:   rgb[:,:,2] = norm_sl # Blue
            elif c == 1: rgb[:,:,1] = norm_sl # Green
            elif c == 2: rgb[:,:,0] = norm_sl # Red
            else:        rgb = np.dstack((norm_sl, norm_sl, norm_sl)) # Grayscale

            # Overlay Mask (Yellow)
            if c in viewer_state["masks"] and z < viewer_state["masks"][c].shape[0]:
                m = viewer_state["masks"][c][z] > 0
                rgb[m] = [255, 255, 0] # Solid Yellow

        # Return as PNG
        pil_img = Image.fromarray(rgb.astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        
        return jsonify({
            "image": base64.b64encode(buf.getvalue()).decode()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

# ----------------------------------------
# Analysis & Prediction Route
# ----------------------------------------

@app.route('/analyze_cell', methods=['POST'])
def analyze_cell():
    if viewer_state["data"] is None:
        return jsonify({"error": "No data loaded"})
    
    # 1. Check if we have masks
    if not viewer_state["masks"]:
        return jsonify({"error": "No segmentation masks found. Run Segmentation first."})
        
    try:
        print("[INFO] Starting Feature Extraction...")
        
        # Get Spacing
        spacing = viewer_state["spacing"] # (Z, Y, X)
        
        # 2. Extract Features
        # We aggregate features from all available channels into a single dictionary
        # Keys must match the training script prefixes EXACTLY, including units!
        features = {}
        
        # Add Voxel Sizes (Required by Model)
        features["Voxel_Z_µm"] = spacing[0]
        features["Voxel_Y_µm"] = spacing[1]
        features["Voxel_X_µm"] = spacing[2]
        
        # Nucleus (Channel 0)
        if 0 in viewer_state["masks"]:
            print("[INFO] Analyzing Nucleus...")
            nuc_feats = analyze_nucleus(viewer_state["masks"][0], spacing)
            if nuc_feats:
                # Map plain keys to Model keys (with units)
                features["Nucleus_Volume_µm³"] = nuc_feats.get('Volume', 0)
                features["Nucleus_Sphericity_ratio"] = nuc_feats.get('Sphericity', 0)
                features["Nucleus_Elongation_ratio"] = nuc_feats.get('Elongation', 0)
                features["Nucleus_Flatness_ratio"] = nuc_feats.get('Flatness', 0)
                features["Nucleus_Solidity_ratio"] = nuc_feats.get('Solidity', 0)
                features["Nucleus_Circularity_ratio"] = nuc_feats.get('Circularity', 0)
                
        # Actin (Channel 1)
        if 1 in viewer_state["masks"]:
            print("[INFO] Analyzing Actin...")
            actin_feats = analyze_actin(viewer_state["masks"][1], spacing)
            if actin_feats:
                features["Actin_Volume_µm³"] = actin_feats.get('Volume', 0)
                features["Actin_Skeleton_Length_µm"] = actin_feats.get('Skeleton_Length', 0)
                features["Actin_Convex_Hull_Volume_µm³"] = actin_feats.get('Convex_Hull_Volume', 0)
                features["Actin_Solidity_ratio"] = actin_feats.get('Solidity', 0)
                features["Actin_Extent_ratio"] = actin_feats.get('Extent', 0)
                features["Actin_Fractional_Anisotropy_ratio"] = actin_feats.get('Fractional_Anisotropy', 0)
                features["Actin_Major_Axis_µm"] = actin_feats.get('Major_Axis', 0)
                features["Actin_Intermediate_Axis_µm"] = actin_feats.get('Intermediate_Axis', 0)
                features["Actin_Minor_Axis_µm"] = actin_feats.get('Minor_Axis', 0)
                
        # Mito (Channel 2)
        if 2 in viewer_state["masks"]:
            print("[INFO] Analyzing Mitochondria...")
            mito_feats = analyze_mito(viewer_state["masks"][2], spacing)
            if mito_feats:
                # Map plain keys to Model keys (with units)
                features["Mito_Volume_µm³"] = mito_feats.get('Volume', 0)
                features["Mito_Surface_Area_µm²"] = mito_feats.get('Surface_Area', 0)
                features["Mito_Sphericity_ratio"] = mito_feats.get('Sphericity', 0)
                features["Mito_Fragment_Count_n"] = mito_feats.get('Fragment_Count', 0)
                features["Mito_Mean_Fragment_Sphericity_ratio"] = mito_feats.get('Mean_Fragment_Sphericity', 0)
                features["Mito_Std_Fragment_Sphericity_ratio"] = mito_feats.get('Std_Fragment_Sphericity', 0)
                features["Mito_Min_Fragment_Sphericity_ratio"] = mito_feats.get('Min_Fragment_Sphericity', 0)
                features["Mito_Max_Fragment_Sphericity_ratio"] = mito_feats.get('Max_Fragment_Sphericity', 0)
                features["Mito_Mean_Fragment_Volume_µm³"] = mito_feats.get('Mean_Fragment_Volume', 0)
                features["Mito_Junction_Count_n"] = mito_feats.get('Junction_Count', 0)
                features["Mito_Branch_Count_n"] = mito_feats.get('Branch_Count', 0)
                features["Mito_Mean_Branch_Length_µm"] = mito_feats.get('Mean_Branch_Length', 0)
                features["Mito_Total_Network_Length_µm"] = mito_feats.get('Total_Network_Length', 0)
                features["Mito_Mean_Tortuosity_ratio"] = mito_feats.get('Mean_Tortuosity', 0)
                features["Mito_Cyclomatic_Number_n"] = mito_feats.get('Cyclomatic_Number', 0)

        # 3. Load Prediction Model
        # Ensure the user has placed their trained model here
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'classifier.joblib')
        
        prediction = "Model Not Found"
        confidence = 0.0
        
        if os.path.exists(model_path):
            try:
                bundle    = joblib.load(model_path)
                # Support both plain pipeline and {pipeline, features, threshold} bundle
                if isinstance(bundle, dict):
                    clf       = bundle['pipeline']
                    feat_list = bundle.get('features', None)
                    threshold = bundle.get('threshold', 0.5)
                else:
                    clf       = bundle
                    feat_list = list(clf.feature_names_in_) if hasattr(clf, 'feature_names_in_') else None
                    threshold = 0.5

                df_pred = pd.DataFrame([features])

                # Select only the features the model expects
                if feat_list:
                    for col in feat_list:
                        if col not in df_pred.columns:
                            df_pred[col] = 0.0
                    df_pred = df_pred[feat_list]

                df_pred = df_pred.fillna(0)

                probas      = clf.predict_proba(df_pred)[0]
                label_names = bundle.get('label_names', ['Healthy', 'TAA']) if isinstance(bundle, dict) else ['Healthy', 'TAA']

                if len(label_names) == 3:
                    # 3-class: Healthy / TAA / BAV — pick argmax
                    pred_idx   = int(probas.argmax())
                    prediction = label_names[pred_idx]
                    confidence = float(probas[pred_idx])
                    print(f"[INFO] probas={dict(zip(label_names, probas.round(3)))} → {prediction}")
                else:
                    # Binary fallback with threshold
                    proba      = float(probas[1])
                    pred_cls   = int(proba >= threshold)
                    prediction = "Diseased (TAA)" if pred_cls == 1 else "Healthy"
                    confidence = proba
                    print(f"[INFO] TAA prob={proba:.3f} threshold={threshold} → {prediction}")

            except Exception as e:
                err_msg = str(e)
                print(f"[WARN] Prediction failed: {err_msg}")
                import traceback; traceback.print_exc()
                prediction = f"Prediction Error: {err_msg[:50]}"
        else:
            print(f"[WARN] Model not found at {model_path}")
            prediction = "No Model File (Upload 'classifier.joblib')"
        
        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "features": features
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

# ----------------------------------------
# Download Route
# ----------------------------------------
@app.route('/download_segmentation')
def download_segmentation():
    if not viewer_state["masks"]:
        return "No masks to download. Run segmentation first.", 400
        
    try:
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # For each mask, save as NII and add to zip
            for c, mask in viewer_state["masks"].items():
                # mask is (Z, Y, X).
                # Create NIfTI image. 
                # Ideally we set the affine matrix correctly using spacing.
                # Spacing is (Z, Y, X).
                # Nibabel expects (X, Y, Z, T).
                
                # Simple identity affine for now.
                affine = np.eye(4)
                
                # Swap axes for NiBabel? 
                # AICSImage returns (Z, Y, X). NiBabel is (X, Y, Z).
                # Let's transpose.
                mask_t = np.transpose(mask, (2, 1, 0))
                
                nii = nib.Nifti1Image(mask_t, affine)
                
                # Set spacing
                # viewer_state["spacing"] is (Z, Y, X).
                # nibabel zooms should be (X, Y, Z).
                if "spacing" in viewer_state and viewer_state["spacing"]:
                    sp = viewer_state["spacing"]
                    zooms = (sp[2], sp[1], sp[0]) 
                    nii.header.set_zooms(zooms)
                
                # Save to temp buffer? Or file. Nifti1Image needs file usually? no.
                # Actually we can write to a BytesIO but nibabel.save takes filename.
                # So we use a generic filename object or just a tmp file.
                tmp_name = f"/tmp/mask_{c}.nii.gz"
                nib.save(nii, tmp_name)
                zf.write(tmp_name, arcname=f"mask_channel_{c}.nii.gz")
                
        memory_file.seek(0)
        return send_file(memory_file, download_name='segmentation_masks.zip', as_attachment=True)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error creating zip: {e}", 500

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

# ----------------------------------------
# Segmentation Task (Threaded)
# ----------------------------------------
seg_progress = {"val": 0, "status": "Idle", "result": None}

def seg_task_runner():
    global seg_progress
    try:
        data = viewer_state["data"]
        results = {}
        
        # Temp folders
        base_tmp = "/tmp/oir_segmentation"
        if os.path.exists(base_tmp): shutil.rmtree(base_tmp)
        os.makedirs(base_tmp, exist_ok=True)
        
        tasks_map = {0: 3, 1: 1, 2: 2} # Nuc, Actin, Mito
        names = {0:"Nucleus", 1:"Actin", 2:"Mitochondria"}
        
        total_steps = len(tasks_map) * 3 
        current_step = 0
        
        seg_progress["val"] = 5
        seg_progress["status"] = "Initializing..."
        
        for ch_idx, task_id in tasks_map.items():
            if ch_idx >= data.shape[0]: continue
            
            # Step 1: Prep
            seg_progress["status"] = f"Processing {names.get(ch_idx,'Channel')}..."
            
            inp_dir = os.path.join(base_tmp, f"task{task_id}_in")
            out_dir = os.path.join(base_tmp, f"task{task_id}_out")
            os.makedirs(inp_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            
            save_nifti_for_nnunet(data[ch_idx], os.path.join(inp_dir, "case_000_0000.nii.gz"))
            current_step += 1
            seg_progress["val"] = int((current_step / total_steps) * 100)
            
            # Step 2: Run AI
            try:
                run_nnunet_predict(inp_dir, out_dir, task_id)
            except Exception as e:
                print(f"[ERROR] Task {task_id} failed: {e}")
                results[ch_idx] = f"Error: {e}"
                continue
                
            current_step += 1
            seg_progress["val"] = int((current_step / total_steps) * 100)
            
            # Step 3: Load Result
            output_filename = os.path.join(out_dir, "case_000.nii.gz")
            if os.path. exists(output_filename):
                mask = load_nifti_mask(output_filename)
                if mask is not None:
                    viewer_state["masks"][ch_idx] = mask
                    results[ch_idx] = "Success"
                else:
                    results[ch_idx] = "Failed (Empty)"
            else:
                 results[ch_idx] = "Failed (No Output)"
                 
            current_step += 1
            seg_progress["val"] = int((current_step / total_steps) * 100)

        seg_progress["val"] = 100
        seg_progress["status"] = "Complete"
        seg_progress["result"] = results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        seg_progress["status"] = f"Error: {e}"
        seg_progress["val"] = 0

@app.route('/run_segmentation', methods=['POST'])
def run_segmentation():
    if viewer_state["data"] is None:
        return jsonify({"error": "No data loaded"})
        
    # Start Thread
    thread = threading.Thread(target=seg_task_runner)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started"})

@app.route('/seg_status')
def get_seg_status():
    return jsonify(seg_progress)


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("OIR Analysis Platform")
    print("=" * 50)
    print("\nOpen: http://localhost:5051")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=8765, debug=False, use_reloader=False)
