"""
OIR Microscopy Web Viewer
Uses aicsimageio + BioFormats to read Olympus .oir files.
Features: 
- Load .oir file
- View Z-stack slices
- Switch Channels
- Auto-scaled display (min-max)
"""

import os
import sys
import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, render_template_string, jsonify, request

# aicsimageio for .oir reading (BioFormats-based)
from aicsimageio import AICSImage

app = Flask(__name__)

# State to hold the loaded image in memory
viewer_state = {
    "data": None,       # numpy array shape (C, Z, Y, X)
    "channels": 0,
    "slices": 0,
    "width": 0,
    "height": 0,
}

HTML_TEMPLATE = r'''
<!DOCTYPE html>
<html>
<head>
    <title>OIR Microscopy Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #fff5f8; color: #5e4b56; }
        .container { max-width: 1400px; margin: 0 auto; padding: 10px; }
        h1 { text-align: center; margin-bottom: 10px; color: #d65db1; font-size: 24px; font-weight: 600; }

        .top-controls {
            background: #ffe4e9;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 15px;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
            box-shadow: 0 4px 6px rgba(214, 93, 177, 0.1);
        }

        button {
            background: #ffb7ce;
            color: #5e4b56;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.2s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        button:hover { background: #ff9eb9; color: #fff; }

        input[type="range"] { width: 150px; accent-color: #d65db1; }
        input[type="file"] { color: #5e4b56; }
        
        select { 
            background: #fff; 
            color: #5e4b56; 
            border: 2px solid #ffb7ce; 
            padding: 6px; 
            border-radius: 8px; 
            outline: none;
        }

        label { font-size: 13px; color: #8c7b85; font-weight: 600; }
        
        .main { display: flex; gap: 15px; }
        .image-area { 
            flex: 1; 
            background: #ffe4e9; 
            border-radius: 12px; 
            padding: 10px; 
            min-height: 550px; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            position: relative;
            box-shadow: 0 4px 6px rgba(214, 93, 177, 0.05);
        }
        #canvas { max-width: 100%; border-radius: 4px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }

        .panel {
            width: 320px;
            background: #ffe4e9;
            border-radius: 12px;
            padding: 15px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 4px 6px rgba(214, 93, 177, 0.1);
        }
        .section { margin-bottom: 20px; }
        .section-title { 
            color: #d65db1; 
            font-size: 14px; 
            font-weight: bold; 
            margin-bottom: 10px; 
            border-bottom: 2px solid #ffccd5; 
            padding-bottom: 5px; 
        }
        
        .info-box { 
            background: #fff; 
            padding: 10px; 
            border-radius: 8px; 
            margin-bottom: 8px;
            border: 1px solid #ffccd5;
        }
        .info-row { display: flex; justify-content: space-between; font-size: 13px; padding: 4px 0; }
        .info-val { color: #d65db1; font-weight: bold; }

        #status { color: #8c7b85; position: absolute; }
        .nav-btn { padding: 6px 12px; }

        .tip { 
            background: #fff0f5; 
            border-left: 4px solid #ffb7ce; 
            padding: 10px; 
            font-size: 12px; 
            color: #5e4b56; 
            margin-top: 10px; 
            border-radius: 0 4px 4px 0;
        }
        .tip b { color: #d65db1; }

        /* Status Box Centering & Styling */
        #status-container {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            
            /* Flebox for perfect centering */
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            
            text-align: center;
            background: rgba(255, 255, 255, 0.9);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(214, 93, 177, 0.15);
            min-width: 350px; /* Wider to prevent wrapping */
        }

        #status {
            font-size: 18px;
            color: #d65db1;
            font-weight: 500;
            margin-top: 15px;
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            width: 60px;
            height: 60px;
            margin: 0 auto;
            border: 5px solid rgba(255, 183, 206, 0.3);
            border-radius: 50%;
            border-top: 5px solid #d65db1;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>OIR Microscopy Viewer</h1>

        <div class="top-controls">
            <!-- 1. File Loader -->
            <input type="file" id="file-input" accept=".oir,.OIR">
            <button onclick="loadFile()">Load OIR</button>

            <!-- 2. Channel Selector -->
            <label>Channel:</label>
            <select id="channel-select" onchange="setChannel(this.value)">
                <option value="0">Ch 1</option>
            </select>

            <!-- 3. Z-Slice Navigation -->
            <button class="nav-btn" onclick="prevSlice()">&lt;</button>
            <span id="slice-info" style="min-width:70px;text-align:center">0/0</span>
            <button class="nav-btn" onclick="nextSlice()">&gt;</button>
            <input type="range" id="slice-slider" min="0" max="0" value="0" oninput="setSlice(this.value)">
        </div>

        <div class="main">
            <!-- Left: Image Display -->
            <div class="image-area">
                
                <!-- Status & Spinner Container -->
                <div id="status-container">
                    <div id="spinner" class="spinner"></div>
                    <div id="status">Load an .oir file to begin</div>
                </div>

                <canvas id="canvas" style="display:none"></canvas>
            </div>

            <!-- Right: Info Panel -->
            <div class="panel">
                <div class="section">
                    <div class="section-title">Image Info</div>
                    <div class="info-box">
                        <div class="info-row"><span>Position:</span><span class="info-val" id="pos-val">(-, -)</span></div>
                        <div class="info-row"><span>Intensity:</span><span class="info-val" id="int-val">-</span></div>
                        <div class="info-row"><span>Dimensions:</span><span class="info-val" id="dim-val">-</span></div>
                        <div class="info-row"><span>Raw Shape:</span><span class="info-val" id="shape-val">-</span></div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Instructions</div>
                    <p style="font-size:12px; color:#aaa; line-height: 1.6;">
                        1. <b style="color:#00d4ff">Load OIR</b> to upload a file.<br>
                        2. Use the slider or mouse wheel to browse Z-slices.<br>
                        3. Switch channels to see different stains (Nucleus, Actin, Mitochondria).
                    </p>
                </div>
            </div>
        </div>
    </div>

<script>
let totalSlices = 0, currentSlice = 0, currentChannel = 0;
let width = 0, height = 0;
let pixels = null; // Store raw pixel data for mouseover

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

async function loadFile() {
    const f = document.getElementById('file-input').files[0];
    if (!f) return alert('Select a file first');

    document.getElementById('status').textContent = 'Loading... (this may take a moment)';
    
    // Show spinner
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('status-container').style.display = 'block';

    canvas.style.display = 'none';

    const fd = new FormData();
    fd.append('file', f);

    try {
        const res = await fetch('/load', { method: 'POST', body: fd });
        const data = await res.json();

        if (data.error) throw new Error(data.error);

        // Hide spinner on success
        document.getElementById('spinner').style.display = 'none';
        document.getElementById('status').textContent = ''; // Clear status text
        document.getElementById('status-container').style.display = 'none';

        totalSlices = data.slices;
        width = data.width;
        height = data.height;
        let channels = data.channels;

        // Reset to middle slice
        currentSlice = Math.floor(totalSlices / 2);
        currentChannel = 0;

        // Update UI
        document.getElementById('dim-val').textContent = `${width}x${height}x${totalSlices} (${channels}ch)`;
        document.getElementById('shape-val').textContent = data.raw_shape;
        
        const slider = document.getElementById('slice-slider');
        slider.max = totalSlices - 1;
        slider.value = currentSlice;

        const chSelect = document.getElementById('channel-select');
        chSelect.innerHTML = '';
        for (let i = 0; i < channels; i++) {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = `Ch ${i + 1}`;
            chSelect.appendChild(opt);
        }

        document.getElementById('status').style.display = 'none';
        canvas.style.display = 'block';

        updateDisplay();
    } catch (e) {
        alert("Error loading file: " + e.message);
        document.getElementById('status').textContent = 'Error loading file. Check terminal.';
        document.getElementById('spinner').style.display = 'none'; // Hide spinner
    }
}

async function updateDisplay() {
    document.getElementById('slice-info').textContent = `${parseInt(currentSlice) + 1}/${totalSlices}`;

    try {
        // Request the specific channel and slice image
        const res = await fetch(`/slice/${currentChannel}/${currentSlice}`);
        const data = await res.json();
        
        if (data.error) { console.error(data.error); return; }

        pixels = data.pixels;

        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        img.src = 'data:image/png;base64,' + data.image;
    } catch (e) {
        console.error(e);
    }
}

function setSlice(val) {
    currentSlice = parseInt(val);
    updateDisplay();
}

function prevSlice() {
    if (currentSlice > 0) { 
        currentSlice--; 
        document.getElementById('slice-slider').value = currentSlice; 
        updateDisplay(); 
    }
}

function nextSlice() {
    if (currentSlice < totalSlices - 1) { 
        currentSlice++; 
        document.getElementById('slice-slider').value = currentSlice; 
        updateDisplay(); 
    }
}

function setChannel(val) {
    currentChannel = parseInt(val);
    updateDisplay();
}

// Mouse interaction: show pixel position + intensity
canvas.addEventListener('mousemove', e => {
    if (!pixels) return;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    const x = Math.floor((e.clientX - rect.left) * sx);
    const y = Math.floor((e.clientY - rect.top) * sy);

    if (x >= 0 && x < width && y >= 0 && y < height) {
        document.getElementById('pos-val').textContent = `(${x}, ${y})`;
        document.getElementById('int-val').textContent = pixels[y * width + x];
    }
});

// Mouse wheel: scroll through slices
canvas.addEventListener('wheel', e => {
    e.preventDefault();
    if (e.deltaY > 0) nextSlice();
    else prevSlice();
});

// Arrow keys: navigate slices
document.addEventListener('keydown', e => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') prevSlice();
    else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') nextSlice();
});
</script>
</body>
</html>
'''


# ----------------------------
# Backend Routes
# ----------------------------
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


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
        # This ensures the image is always visible without manual tuning
        vmin, vmax = sl.min(), sl.max()
        
        # Avoid division by zero if image is blank
        if vmax - vmin < 1e-6:
            disp = np.zeros_like(sl, dtype=np.uint8)
        else:
            disp = ((sl - vmin) / (vmax - vmin) * 255).astype(np.uint8)

        pil_img = Image.fromarray(disp)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")

        return jsonify({
            # Send the image as base64 string
            "image": base64.b64encode(buf.getvalue()).decode(),
            # Send raw pixel values for mouseover inspection
            "pixels": sl.astype(int).flatten().tolist()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("OIR Microscopy Web Viewer")
    print("=" * 50)
    print("\nOpen: http://localhost:5001")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
