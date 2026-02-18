"""
DICOM Angiography Vessel Segmentation (Improved)
Optimized for X-ray angiography where vessels are DARK on bright background

Upgrades vs your version:
- Median background subtraction across frames (big boost for angio)
- Robust normalization + CLAHE
- Hybrid vessel enhancement (Frangi + top-hat)
- Seed-guided grayscale reconstruction (grows ONLY from seeds along vesselness)
- Cleaner morphology
"""

import os
import sys
import base64
import io
import numpy as np

# ----------------------------
# Lazy installs (keep your style)
# ----------------------------
try:
    import pydicom
except ImportError:
    os.system(f"{sys.executable} -m pip install pydicom -q")
    import pydicom

try:
    from PIL import Image
except ImportError:
    os.system(f"{sys.executable} -m pip install Pillow -q")
    from PIL import Image

try:
    from flask import Flask, render_template_string, jsonify, request
except ImportError:
    os.system(f"{sys.executable} -m pip install flask -q")
    from flask import Flask, render_template_string, jsonify, request

try:
    from skimage.filters import frangi, threshold_otsu
    from skimage.exposure import equalize_adapthist
    from skimage.morphology import (
        disk, black_tophat, reconstruction,
        binary_opening, binary_closing, binary_dilation
    )
    from scipy.ndimage import binary_fill_holes
except ImportError:
    os.system(f"{sys.executable} -m pip install scikit-image scipy -q")
    from skimage.filters import frangi, threshold_otsu
    from skimage.exposure import equalize_adapthist
    from skimage.morphology import (
        disk, black_tophat, reconstruction,
        binary_opening, binary_closing, binary_dilation
    )
    from scipy.ndimage import binary_fill_holes


app = Flask(__name__)

viewer_state = {
    "pixel_array": None,
    "total_frames": 0,
}

HTML_TEMPLATE = r'''
<!DOCTYPE html>
<html>
<head>
    <title>Angiography Vessel Segmentation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; }
        .container { max-width: 1400px; margin: 0 auto; padding: 10px; }
        h1 { text-align: center; margin-bottom: 10px; color: #00d4ff; font-size: 22px; }

        .top-controls {
            background: #16213e;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }

        button {
            background: #00d4ff;
            color: #000;
            border: none;
            padding: 8px 14px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { background: #00a8cc; }
        button.active { background: #ff6b6b; color: #fff; }
        button.green { background: #00b894; }
        button.purple { background: #6c5ce7; color: #fff; }

        input[type="range"] { width: 100px; }
        input[type="file"] { color: #eee; }
        label { font-size: 12px; color: #aaa; }
        .slider-val { color: #00d4ff; min-width: 30px; display: inline-block; }

        .main { display: flex; gap: 10px; }
        .image-area { flex: 1; background: #000; border-radius: 8px; padding: 10px; min-height: 550px; display: flex; align-items: center; justify-content: center; }
        #canvas { max-width: 100%; cursor: crosshair; }

        .panel {
            width: 320px;
            background: #16213e;
            border-radius: 8px;
            padding: 12px;
            max-height: 80vh;
            overflow-y: auto;
        }
        .section { margin-bottom: 15px; }
        .section-title { color: #00d4ff; font-size: 13px; font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid #0f3460; padding-bottom: 5px; }

        .ctrl-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
        .ctrl-row label { min-width: 90px; font-size: 12px; }

        .info-box { background: #0f3460; padding: 8px; border-radius: 4px; margin-bottom: 8px; }
        .info-row { display: flex; justify-content: space-between; font-size: 12px; padding: 2px 0; }
        .info-val { color: #00d4ff; }

        .seed-list { background: #0a1628; padding: 5px; border-radius: 4px; max-height: 80px; overflow-y: auto; font-size: 11px; }
        .seed-item { padding: 2px 5px; display: flex; justify-content: space-between; }
        .seed-x { color: #ff6b6b; cursor: pointer; }

        #status { color: #888; text-align: center; }
        .nav-btn { padding: 6px 10px; }

        .tip { background: #0a2a1a; border-left: 3px solid #00b894; padding: 8px; font-size: 11px; color: #aaa; margin-top: 8px; }
        .tip b { color: #00b894; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Angiography Vessel Segmentation</h1>

        <div class="top-controls">
            <input type="file" id="file-input" accept=".dcm,.DCM,*">
            <button onclick="loadFile()">Load DICOM</button>

            <button class="nav-btn" onclick="goFrame(0)">&lt;&lt;</button>
            <button class="nav-btn" onclick="prevFrame()">&lt;</button>
            <span id="frame-info" style="min-width:70px;text-align:center">0/0</span>
            <button class="nav-btn" onclick="nextFrame()">&gt;</button>
            <button class="nav-btn" onclick="goFrame(-1)">&gt;&gt;</button>
            <input type="range" id="frame-slider" min="0" max="0" value="0" onchange="setFrame(this.value)">

            <label>WC:</label>
            <input type="range" id="wc" min="0" max="4095" value="128" onchange="updateDisplay()">
            <span class="slider-val" id="wc-val">128</span>

            <label>WW:</label>
            <input type="range" id="ww" min="1" max="4095" value="256" onchange="updateDisplay()">
            <span class="slider-val" id="ww-val">256</span>

            <button onclick="autoWL()">Auto</button>
        </div>

        <div class="main">
            <div class="image-area">
                <div id="status">Load a DICOM file</div>
                <canvas id="canvas" style="display:none"></canvas>
            </div>

            <div class="panel">
                <div class="section">
                    <div class="section-title">Image Info</div>
                    <div class="info-box">
                        <div class="info-row"><span>Position:</span><span class="info-val" id="pos-val">(-, -)</span></div>
                        <div class="info-row"><span>Intensity:</span><span class="info-val" id="int-val">-</span></div>
                        <div class="info-row"><span>Frame:</span><span class="info-val" id="frm-val">-</span></div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Step 1: Mark Vessel Points</div>
                    <div class="ctrl-row">
                        <button id="seed-btn" onclick="toggleSeed()">Start Marking</button>
                        <button onclick="clearSeeds()">Clear</button>
                    </div>
                    <div class="seed-list" id="seed-list">Click "Start Marking" then click on vessel</div>
                    <div class="tip"><b>Tip:</b> Mark 5-15 points along the vessel centerline</div>
                </div>

                <div class="section">
                    <div class="section-title">Step 2: Segmentation Settings</div>

                    <div class="ctrl-row">
                        <label>Method:</label>
                        <select id="method" style="flex:1;padding:5px;background:#0f3460;color:#fff;border:1px solid #00d4ff;border-radius:4px">
                            <option value="hybrid" selected>Hybrid (Recommended)</option>
                            <option value="frangi">Frangi Only</option>
                            <option value="tophat">Top-Hat Only</option>
                            <option value="intensity">Intensity Only</option>
                        </select>
                    </div>

                    <div class="ctrl-row">
                        <label>Vessel Width:</label>
                        <input type="range" id="vessel-width" min="3" max="25" value="10" oninput="document.getElementById('vw-val').textContent=this.value">
                        <span class="slider-val" id="vw-val">10</span>
                    </div>

                    <div class="ctrl-row">
                        <label>Threshold:</label>
                        <input type="range" id="threshold" min="1" max="100" value="35" oninput="document.getElementById('th-val').textContent=this.value">
                        <span class="slider-val" id="th-val">35</span>
                    </div>

                    <div class="ctrl-row">
                        <label>Expand:</label>
                        <input type="range" id="expand" min="0" max="10" value="2" oninput="document.getElementById('ex-val').textContent=this.value">
                        <span class="slider-val" id="ex-val">2</span>
                    </div>

                    <button onclick="segmentFrame()" class="purple" style="width:100%;margin-top:5px">Segment This Frame</button>

                    <div class="tip"><b>Hybrid</b> uses temporal background subtraction + vesselness + seed-growing (best overall).</div>
                </div>

                <div class="section">
                    <div class="section-title">Step 3: Apply to All Frames</div>
                    <button onclick="segmentAll()" class="green" style="width:100%">Segment ALL Frames</button>
                    <div id="progress" style="margin-top:8px;display:none">
                        <div style="background:#0a1628;border-radius:4px;height:20px;overflow:hidden">
                            <div id="prog-bar" style="background:linear-gradient(90deg,#00d4ff,#00b894);height:100%;width:0%;transition:width 0.3s;text-align:center;font-size:11px;line-height:20px"></div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Display</div>
                    <div class="ctrl-row">
                        <label><input type="checkbox" id="show-overlay" checked onchange="updateDisplay()"> Show Overlay</label>
                        <input type="color" id="overlay-color" value="#00ff00" onchange="updateDisplay()">
                    </div>
                    <div class="ctrl-row">
                        <label>Opacity:</label>
                        <input type="range" id="opacity" min="10" max="90" value="50" onchange="updateDisplay()">
                    </div>
                    <div class="ctrl-row">
                        <button onclick="clearAll()">Clear All</button>
                        <button onclick="exportMasks()" class="purple">Export</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

<script>
let pixels = null, masks = {}, totalFrames = 0, frame = 0, w = 0, h = 0;
let seedMode = false, seeds = [];
const canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');

async function loadFile() {
    const f = document.getElementById('file-input').files[0];
    if (!f) return alert('Select a file first');

    document.getElementById('status').textContent = 'Loading...';
    document.getElementById('status').style.display = 'block';
    canvas.style.display = 'none';

    const fd = new FormData();
    fd.append('file', f);

    const res = await fetch('/load', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) return alert(data.error);

    totalFrames = data.frames;
    w = data.width; h = data.height;
    frame = 0;
    masks = {};
    seeds = [];
    updateSeedList();

    document.getElementById('frame-slider').max = totalFrames - 1;
    document.getElementById('wc').value = data.wc;
    document.getElementById('ww').value = data.ww;
    document.getElementById('wc-val').textContent = data.wc;
    document.getElementById('ww-val').textContent = data.ww;

    document.getElementById('status').style.display = 'none';
    canvas.style.display = 'block';
    updateDisplay();
}

async function updateDisplay() {
    const wc = document.getElementById('wc').value;
    const ww = document.getElementById('ww').value;
    document.getElementById('wc-val').textContent = wc;
    document.getElementById('ww-val').textContent = ww;

    const res = await fetch(`/frame/${frame}?wc=${wc}&ww=${ww}`);
    const data = await res.json();
    if (data.error) return;

    pixels = data.pixels;

    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        if (masks[frame] && document.getElementById('show-overlay').checked) {
            drawMask(masks[frame]);
        }
        drawSeeds();
    };
    img.src = 'data:image/png;base64,' + data.image;

    document.getElementById('frame-info').textContent = `${frame + 1}/${totalFrames}`;
    document.getElementById('frm-val').textContent = `${frame + 1}/${totalFrames}`;
    document.getElementById('frame-slider').value = frame;
}

function drawMask(mask) {
    const color = document.getElementById('overlay-color').value;
    const op = document.getElementById('opacity').value / 100;
    const r = parseInt(color.substr(1, 2), 16);
    const g = parseInt(color.substr(3, 2), 16);
    const b = parseInt(color.substr(5, 2), 16);

    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const px = imgData.data;

    for (let i = 0; i < mask.length; i++) {
        if (mask[i]) {
            const j = i * 4;
            px[j]     = px[j]     * (1 - op) + r * op;
            px[j + 1] = px[j + 1] * (1 - op) + g * op;
            px[j + 2] = px[j + 2] * (1 - op) + b * op;
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

function drawSeeds() {
    seeds.forEach((s, i) => {
        ctx.beginPath();
        ctx.arc(s.x, s.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ff0000';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px Arial';
        ctx.fillText(i + 1, s.x + 7, s.y + 3);
    });
}

function setFrame(f) { frame = parseInt(f); updateDisplay(); }
function prevFrame() { if (frame > 0) { frame--; updateDisplay(); } }
function nextFrame() { if (frame < totalFrames - 1) { frame++; updateDisplay(); } }
function goFrame(f) { frame = f === -1 ? totalFrames - 1 : 0; updateDisplay(); }

async function autoWL() {
    const res = await fetch(`/auto_wl/${frame}`);
    const data = await res.json();
    document.getElementById('wc').value = data.wc;
    document.getElementById('ww').value = data.ww;
    updateDisplay();
}

function toggleSeed() {
    seedMode = !seedMode;
    const btn = document.getElementById('seed-btn');
    btn.textContent = seedMode ? 'Stop Marking' : 'Start Marking';
    btn.classList.toggle('active', seedMode);
}

function clearSeeds() {
    seeds = [];
    updateSeedList();
    updateDisplay();
}

function updateSeedList() {
    const el = document.getElementById('seed-list');
    if (seeds.length === 0) {
        el.innerHTML = 'No points marked yet';
    } else {
        el.innerHTML = seeds.map((s, i) =>
            `<div class="seed-item"><span>${i + 1}: (${s.x}, ${s.y})</span><span class="seed-x" onclick="removeSeed(${i})">X</span></div>`
        ).join('');
    }
}

function removeSeed(i) {
    seeds.splice(i, 1);
    updateSeedList();
    updateDisplay();
}

async function segmentFrame() {
    if (seeds.length < 2) return alert('Mark at least 2 points on the vessel');

    document.getElementById('status').textContent = 'Segmenting...';
    document.getElementById('status').style.display = 'block';

    const res = await fetch('/segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            frame: frame,
            seeds: seeds,
            method: document.getElementById('method').value,
            vessel_width: parseInt(document.getElementById('vessel-width').value),
            threshold: parseInt(document.getElementById('threshold').value),
            expand: parseInt(document.getElementById('expand').value)
        })
    });

    const data = await res.json();
    document.getElementById('status').style.display = 'none';

    if (data.error) return alert(data.error);
    masks[frame] = data.mask;
    updateDisplay();
}

async function segmentAll() {
    if (seeds.length < 2) return alert('Mark at least 2 points first');

    document.getElementById('progress').style.display = 'block';
    document.getElementById('prog-bar').style.width = '0%';
    document.getElementById('prog-bar').textContent = '';

    const res = await fetch('/segment_all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            seeds: seeds,
            method: document.getElementById('method').value,
            vessel_width: parseInt(document.getElementById('vessel-width').value),
            threshold: parseInt(document.getElementById('threshold').value),
            expand: parseInt(document.getElementById('expand').value)
        })
    });

    const data = await res.json();
    if (data.error) return alert(data.error);

    masks = {};
    data.masks.forEach((m, i) => masks[i] = m);

    document.getElementById('prog-bar').style.width = '100%';
    document.getElementById('prog-bar').textContent = 'Done!';
    updateDisplay();
}

function clearAll() {
    masks = {};
    document.getElementById('progress').style.display = 'none';
    updateDisplay();
}

async function exportMasks() {
    const keys = Object.keys(masks);
    if (keys.length === 0) return alert('No masks to export');

    for (const f of keys) {
        const m = masks[f];
        const tc = document.createElement('canvas');
        tc.width = w;
        tc.height = h;
        const tctx = tc.getContext('2d');
        const imgData = tctx.createImageData(w, h);

        for (let i = 0; i < m.length; i++) {
            const v = m[i] ? 255 : 0;
            const j = i * 4;
            imgData.data[j] = v;
            imgData.data[j + 1] = v;
            imgData.data[j + 2] = v;
            imgData.data[j + 3] = 255;
        }

        tctx.putImageData(imgData, 0, 0);
        const link = document.createElement('a');
        link.download = `mask_${f.toString().padStart(3, '0')}.png`;
        link.href = tc.toDataURL();
        link.click();
        await new Promise(r => setTimeout(r, 100));
    }
}

canvas.addEventListener('mousemove', e => {
    if (!pixels) return;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    const x = Math.floor((e.clientX - rect.left) * sx);
    const y = Math.floor((e.clientY - rect.top) * sy);

    if (x >= 0 && x < w && y >= 0 && y < h) {
        document.getElementById('pos-val').textContent = `(${x}, ${y})`;
        document.getElementById('int-val').textContent = pixels[y * w + x];
    }
});

canvas.addEventListener('click', e => {
    if (!seedMode || !pixels) return;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    const x = Math.floor((e.clientX - rect.left) * sx);
    const y = Math.floor((e.clientY - rect.top) * sy);

    if (x >= 0 && x < w && y >= 0 && y < h) {
        seeds.push({ x, y });
        updateSeedList();
        updateDisplay();
    }
});

document.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft') prevFrame();
    else if (e.key === 'ArrowRight') nextFrame();
    else if (e.key === 'Escape' && seedMode) toggleSeed();
});

canvas.addEventListener('wheel', e => {
    e.preventDefault();
    if (e.deltaY > 0) nextFrame();
    else prevFrame();
});
</script>
</body>
</html>
'''


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/load", methods=["POST"])
def load_dicom():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"})

    try:
        file.save("/tmp/temp.dcm")
        dcm = pydicom.dcmread("/tmp/temp.dcm")
        arr = dcm.pixel_array.astype(np.float32)

        # Normalize frames layout
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[2] == 3:
            arr = np.mean(arr, axis=2)[np.newaxis, ...]
        elif arr.ndim == 4:
            arr = np.mean(arr, axis=3)

        viewer_state["pixel_array"] = arr
        viewer_state["total_frames"] = arr.shape[0]

        f = arr[0]
        wc = int(np.mean(f))
        ww = max(int(np.std(f) * 4), 256)

        return jsonify({
            "frames": arr.shape[0],
            "width": arr.shape[2],
            "height": arr.shape[1],
            "wc": wc,
            "ww": ww
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/frame/<int:n>")
def get_frame(n):
    arr = viewer_state["pixel_array"]
    if arr is None:
        return jsonify({"error": "No file"})

    wc = int(request.args.get("wc", 128))
    ww = int(request.args.get("ww", 256))

    f = arr[n]
    lo, hi = wc - ww / 2, wc + ww / 2
    disp = np.clip(f, lo, hi)
    disp = ((disp - lo) / (hi - lo) * 255).astype(np.uint8)

    img = Image.fromarray(disp)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return jsonify({
        "image": base64.b64encode(buf.getvalue()).decode(),
        "pixels": f.astype(int).flatten().tolist()
    })


@app.route("/auto_wl/<int:n>")
def auto_wl(n):
    f = viewer_state["pixel_array"][n]
    return jsonify({"wc": int(np.mean(f)), "ww": max(int(np.std(f) * 4), 256)})


# ----------------------------
# Improved segmentation core
# ----------------------------
def _robust_norm(img, p_low=1.0, p_high=99.5):
    lo, hi = np.percentile(img, [p_low, p_high])
    out = np.clip(img, lo, hi)
    out = (out - lo) / (hi - lo + 1e-8)
    return out


def segment_vessel(frame, seeds, method="hybrid", vessel_width=10, threshold=35, expand=2, background=None):
    """
    Segments dark coronary vessels on bright X-ray background using:
    - optional temporal background subtraction
    - CLAHE
    - hybrid vesselness (Frangi + top-hat)
    - seed-guided grayscale reconstruction
    """
    h, w = frame.shape
    frame = frame.astype(np.float32)

    # 0) temporal background subtraction
    if background is not None:
        x = background.astype(np.float32) - frame  # vessels become brighter
    else:
        x = frame.copy()

    # 1) normalize + CLAHE
    x = _robust_norm(x, 1.0, 99.5)
    x = equalize_adapthist(x, clip_limit=0.02)

    # ensure bright-vessel convention
    if background is None:
        x = 1.0 - x

    r = max(2, int(vessel_width))
    se = disk(r)

    # 2) vessel enhancement
    if method in ("tophat", "hybrid"):
        # using black_tophat on inverted gives "bright tube" emphasis on x
        tophat = black_tophat(1.0 - x, se)
        tophat = tophat / (tophat.max() + 1e-8)
    else:
        tophat = np.zeros_like(x, dtype=np.float32)

    if method in ("frangi", "hybrid"):
        sig_max = max(2, r // 2)
        sigmas = np.linspace(1, sig_max, num=max(2, sig_max))
        vness = frangi(x, sigmas=sigmas, black_ridges=False)
        vness = vness / (vness.max() + 1e-8)
    else:
        vness = np.zeros_like(x, dtype=np.float32)

    if method == "intensity":
        score = x
    elif method == "tophat":
        score = tophat
    elif method == "frangi":
        score = vness
    else:
        score = 0.55 * vness + 0.45 * tophat

    score = np.clip(score, 0, 1)

    # 3) seed-guided reconstruction (grow from seeds along score)
    marker = np.zeros_like(score, dtype=np.float32)
    for s in seeds:
        yy = int(np.clip(s["y"], 0, h - 1))
        xx = int(np.clip(s["x"], 0, w - 1))
        marker[yy, xx] = max(marker[yy, xx], score[yy, xx])

    if marker.max() <= 0:
        return np.zeros((h, w), dtype=int)

    recon = reconstruction(marker, score, method="dilation")
    recon = recon / (recon.max() + 1e-8)

    # 4) threshold (seed-guided + global Otsu fallback)
    rad = max(2, r // 2)
    seed_vals = []
    for s in seeds:
        yy = int(np.clip(s["y"], 0, h - 1))
        xx = int(np.clip(s["x"], 0, w - 1))
        y0, y1 = max(0, yy - rad), min(h, yy + rad + 1)
        x0, x1 = max(0, xx - rad), min(w, xx + rad + 1)
        seed_vals.append(np.median(recon[y0:y1, x0:x1]))
    seed_med = float(np.median(seed_vals))

    try:
        otsu_t = float(threshold_otsu(recon))
    except Exception:
        otsu_t = 0.2

    base_t = 0.65 * seed_med + 0.35 * otsu_t
    # slider: higher => stricter
    t = base_t * (1.15 - 0.8 * (threshold / 100.0))
    t = float(np.clip(t, 0.02, 0.95))

    mask = recon > t

    # 5) cleanup
    mask = binary_closing(mask, disk(1))
    mask = binary_opening(mask, disk(1))
    mask = binary_fill_holes(mask)

    if expand > 0:
        mask = binary_dilation(mask, disk(int(expand)))

    return mask.astype(int)


@app.route("/segment", methods=["POST"])
def segment():
    data = request.json
    arr = viewer_state["pixel_array"]
    if arr is None:
        return jsonify({"error": "No file"})

    try:
        frame_idx = int(data["frame"])
        frame = arr[frame_idx]

        bg = np.median(arr.astype(np.float32), axis=0) if arr.shape[0] > 1 else None

        mask = segment_vessel(
            frame,
            data["seeds"],
            data.get("method", "hybrid"),
            int(data.get("vessel_width", 10)),
            int(data.get("threshold", 35)),
            int(data.get("expand", 2)),
            background=bg
        )
        return jsonify({"mask": mask.flatten().tolist()})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/segment_all", methods=["POST"])
def segment_all():
    data = request.json
    arr = viewer_state["pixel_array"]
    if arr is None:
        return jsonify({"error": "No file"})

    try:
        bg = np.median(arr.astype(np.float32), axis=0)

        masks = []
        for i in range(arr.shape[0]):
            mask = segment_vessel(
                arr[i],
                data["seeds"],
                data.get("method", "hybrid"),
                int(data.get("vessel_width", 10)),
                int(data.get("threshold", 35)),
                int(data.get("expand", 2)),
                background=bg
            )
            masks.append(mask.flatten().tolist())

        return jsonify({"masks": masks})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Angiography Vessel Segmentation (Improved)")
    print("=" * 50)
    print("\nOptimized for X-ray angiography")
    print("(dark vessels on bright background)")
    print("\nOpen: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
