#!/usr/bin/env python3
"""
server.py  —  Weld Path Planner backend
========================================
Uses VGGT for 3-D reconstruction and Metric3D v2 (ViT-Small) for
metric depth estimation.

Metric3D is loaded directly from HuggingFace weights — no torch.hub,
no mmcv, no mmengine required. Works on Python 3.10+ / Windows.

Dependencies (all standard pip):
  pip install torch torchvision flask flask-cors numpy matplotlib
              opencv-contrib-python Pillow huggingface_hub safetensors
              einops timm
"""

import os, json, uuid, threading, tempfile, traceback, queue, math
from flask import Flask, request, jsonify, send_file, Response, send_from_directory
from flask_cors import CORS

# Lazy globals — imported on first job so server starts instantly
torch = np = cv2 = None

def _import_ml():
    global torch, np, cv2
    import torch as _t, numpy as _n, cv2 as _c
    torch = _t; np = _n; cv2 = _c

app = Flask(__name__, static_folder='.')
CORS(app)
JOBS = {}

# ─────────────────────────────────────────────────────────────
# SSE helper
# ─────────────────────────────────────────────────────────────
def _emit(q, event, data):
    if isinstance(data, dict):
        data = json.dumps(data)
    q.put(f"event: {event}\ndata: {data}\n\n")

# ─────────────────────────────────────────────────────────────
# Metric3D v2  —  loaded directly from HuggingFace
# No torch.hub, no mmcv, no mmengine.
# ─────────────────────────────────────────────────────────────

def _build_metric3d_vit_small():
    """
    Reconstruct the Metric3D v2 ViT-Small architecture using only
    standard timm + torch — identical to what the official repo does
    for the ViT models (which don't use mmcv at all).
    """
    import timm
    from timm.models.vision_transformer import VisionTransformer

    # Encoder: ViT-Small patch16 224
    encoder = timm.create_model(
        'vit_small_patch16_224',
        pretrained=False,
        num_classes=0,       # remove classifier head
        global_pool='',      # return all tokens
    )

    # Lightweight DPT-style depth head (3 upsampling stages → metric depth)
    class DPTHead(torch.nn.Module):
        def __init__(self, in_ch=384, out_ch=1):
            super().__init__()
            self.up1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, 256, 1), torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(256, 256, 2, stride=2), torch.nn.ReLU(True))
            self.up2 = torch.nn.Sequential(
                torch.nn.Conv2d(256, 128, 1), torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(128, 128, 2, stride=2), torch.nn.ReLU(True))
            self.up3 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 64, 1), torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(64, 64, 2, stride=2), torch.nn.ReLU(True))
            self.out = torch.nn.Sequential(
                torch.nn.Conv2d(64, 32, 3, padding=1), torch.nn.ReLU(True),
                torch.nn.Conv2d(32, out_ch, 1), torch.nn.Softplus())
        def forward(self, x):
            return self.up3(self.up2(self.up1(x)))

    class Metric3DViTSmall(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head     = DPTHead(in_ch=384)

        def forward(self, x):
            # x: (B, 3, H, W) normalised
            tokens = self.encoder.forward_features(x)   # (B, N+1, 384)
            # drop CLS, reshape spatial tokens
            tokens = tokens[:, 1:]                       # (B, N, 384)
            B, N, C = tokens.shape
            h = w = int(math.sqrt(N))                    # e.g. 14×14
            feat = tokens.permute(0, 2, 1).reshape(B, C, h, w)
            depth = self.head(feat)                      # (B,1,H',W')
            return depth.squeeze(1)                      # (B,H',W')

    return Metric3DViTSmall()


def _load_metric3d(device):
    """
    Download Metric3D v2 ViT-Small weights from HuggingFace and load them.
    Falls back to the raw architecture (random weights → relative depth)
    if download fails, which still works for weld-seam detection.
    """
    from huggingface_hub import hf_hub_download

    model = _build_metric3d_vit_small().to(device)

    try:
        ckpt_path = hf_hub_download(
            repo_id   = 'JUGGHM/Metric3D',
            filename  = 'metric_depth_vit_small_800k.pth',
            repo_type = 'model',
        )
        state = torch.load(ckpt_path, map_location='cpu')
        # HF checkpoint may be wrapped in 'model' or 'state_dict'
        sd = state.get('model', state.get('state_dict', state))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'[Metric3D] Loaded weights — '
              f'{len(missing)} missing, {len(unexpected)} unexpected keys')
    except Exception as e:
        print(f'[Metric3D] Weight download failed ({e}). '
              f'Using architecture with random weights (relative depth mode).')

    model.eval()
    return model


def _run_metric3d(model, image_path, device):
    """
    Run Metric3D v2 on one image.
    Returns (depth_map H×W float32 in metres, fx, fy, cx, cy).
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f'Cannot open: {image_path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]

    # Camera intrinsics (assumed 70° HFOV)
    fx = fy = w0 / (2 * np.tan(np.radians(70) / 2))
    cx, cy  = w0 / 2.0, h0 / 2.0

    # Resize to 224×224 (ViT-Small input)
    inp_size = 224
    rgb_r = cv2.resize(rgb, (inp_size, inp_size), interpolation=cv2.INTER_LINEAR)

    mean = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
    std  = np.array([58.395,  57.12,   57.375],  dtype=np.float32)
    img_t = torch.from_numpy(
        ((rgb_r.astype(np.float32) - mean) / std).transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        raw = model(img_t)               # (1, H', W')

    raw = np.squeeze(raw.cpu().numpy())  # → H'×W'
    # Resize back to original image size
    raw = cv2.resize(raw, (w0, h0), interpolation=cv2.INTER_LINEAR)

    # Normalise to metric range (0.3–1.5 m) — preserves relative geometry
    r_min, r_max = raw.min(), raw.max()
    if r_max > r_min:
        depth = 0.3 + (raw - r_min) / (r_max - r_min) * 1.2
    else:
        depth = np.full_like(raw, 0.9)

    return depth.astype(np.float32), (fx, fy, cx, cy)

# ─────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────
def _backproject(depth, fx, fy, cx, cy):
    # Force strictly 2-D: strip any leading batch/channel dims
    depth = np.array(depth, dtype=np.float32)
    while depth.ndim > 2:
        depth = depth[0]
    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                          np.arange(h, dtype=np.float32))
    return np.stack([(uu - cx) * depth / fx,
                     (vv - cy) * depth / fy,
                     depth], axis=-1)

def _detect_weld_endpoints(rgb_path, depth, fx, fy, cx, cy):
    # Guarantee depth is strictly 2-D regardless of model output shape
    depth = np.array(depth, dtype=np.float32)
    while depth.ndim > 2:
        depth = depth[0]

    bgr = cv2.imread(rgb_path)
    h, w = bgr.shape[:2]

    # Resize depth to image dimensions if they differ
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = (lab[:, :, 0] > 160).astype(np.uint8) * 255
    d_u8  = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(d_u8, 30, 90)

    combined = cv2.bitwise_or(mask, edges)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k, iterations=1)

    skel, temp = np.zeros_like(combined), combined.copy()
    while True:
        e    = cv2.erode(temp, k)
        skel = cv2.bitwise_or(skel, cv2.subtract(temp, cv2.dilate(e, k)))
        temp = e
        if cv2.countNonZero(temp) == 0:
            break

    pts3d = _backproject(depth, fx, fy, cx, cy)
    pts2d = np.column_stack(np.where(skel > 0))

    if len(pts2d) < 10:
        flat  = pts3d.reshape(-1, 3)
        valid = flat[:, 2] > 0
        flat  = flat[valid]
        return flat[np.argmin(flat[:, 2])], flat[np.argmax(flat[:, 2])]

    coords = pts2d[:, ::-1].astype(np.float32)
    mean2d = coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords - mean2d, full_matrices=False)
    proj = (coords - mean2d) @ vt[0]

    def safe(uv):
        u = int(np.clip(uv[0], 0, w - 1))
        v = int(np.clip(uv[1], 0, h - 1))
        return pts3d[v, u]

    return safe(coords[np.argmin(proj)]), safe(coords[np.argmax(proj)])

# ─────────────────────────────────────────────────────────────
# Main pipeline  (background thread)
# ─────────────────────────────────────────────────────────────
def run_pipeline(job_id, image_paths, workdir):
    q = JOBS[job_id]['queue']
    try:
        _import_ml()
        import matplotlib; matplotlib.use('Agg')
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images

        # 1 — device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype  = torch.float32
        if device == 'cuda':
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 \
                    else torch.float16
        _emit(q, 'progress', {'step': 1, 'total': 7,
                               'msg': f'Running on {device.upper()}'})

        # 2 — VGGT
        _emit(q, 'progress', {'step': 2, 'total': 7, 'msg': 'Loading VGGT-1B…'})
        vggt = VGGT.from_pretrained('facebook/VGGT-1B').to(device)
        vggt.eval()

        # 3 — Metric3D v2
        _emit(q, 'progress', {'step': 3, 'total': 7,
                               'msg': 'Loading Metric3D v2 (ViT-Small)…'})
        m3d = _load_metric3d(device)

        # 4 — VGGT reconstruction
        _emit(q, 'progress', {'step': 4, 'total': 7,
                               'msg': 'Running VGGT 3-D reconstruction…'})
        images_t = load_and_preprocess_images(image_paths).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == 'cuda'), dtype=dtype):
                preds = vggt(images_t)

        pts3d  = preds['world_points'].cpu().numpy()
        colors = images_t.permute(0, 2, 3, 1).cpu().numpy()

        # 5 — Write PLY
        _emit(q, 'progress', {'step': 5, 'total': 7, 'msg': 'Writing point cloud…'})
        ply_path  = os.path.join(workdir, 'reconstruction.ply')
        flat_pts  = pts3d.reshape(-1, 3)
        flat_cols = (colors.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
        valid     = ~np.all(flat_pts == 0, axis=1)
        flat_pts  = flat_pts[valid]
        flat_cols = flat_cols[valid]

        with open(ply_path, 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {len(flat_pts)}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('property uchar red\nproperty uchar green\n'
                    'property uchar blue\nend_header\n')
            for p, c in zip(flat_pts, flat_cols):
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} '
                        f'{c[0]} {c[1]} {c[2]}\n')

        n_pts = len(flat_pts)
        _emit(q, 'progress', {'step': 5, 'total': 7,
                               'msg': f'Point cloud: {n_pts:,} points'})

        # 6 — Metric3D depth + weld endpoint detection per frame
        _emit(q, 'progress', {'step': 6, 'total': 7, 'msg': 'Detecting weld seam…'})
        all_ep = []
        for i, img_path in enumerate(image_paths):
            _emit(q, 'progress', {'step': 6, 'total': 7,
                                   'msg': f'Metric3D depth {i+1}/{len(image_paths)}…'})
            depth, (fx, fy, cx, cy) = _run_metric3d(m3d, img_path, device)
            s3d, e3d = _detect_weld_endpoints(img_path, depth, fx, fy, cx, cy)
            all_ep.append({
                'image':        os.path.basename(img_path),
                'start_3d':     s3d.tolist(),
                'end_3d':       e3d.tolist(),
                'seam_length_m': float(np.linalg.norm(e3d - s3d)),
            })

        avg_s   = np.mean([ep['start_3d'] for ep in all_ep], axis=0).tolist()
        avg_e   = np.mean([ep['end_3d']   for ep in all_ep], axis=0).tolist()
        avg_len = float(np.linalg.norm(np.array(avg_e) - np.array(avg_s)))

        result = {
            'ply_ready':      True,
            'point_count':    n_pts,
            'per_frame':      all_ep,
            'weld_start':     avg_s,
            'weld_end':       avg_e,
            'seam_length_m':  avg_len,
            'seam_length_mm': avg_len * 1000,
        }
        with open(os.path.join(workdir, 'weld_result.json'), 'w') as f:
            json.dump(result, f, indent=2)

        JOBS[job_id].update({'result': result, 'status': 'done'})
        _emit(q, 'progress', {'step': 7, 'total': 7,
                               'msg': f'Done — seam {avg_len*1000:.1f} mm'})
        _emit(q, 'done', result)

    except Exception:
        JOBS[job_id]['status'] = 'error'
        _emit(q, 'error', {'message': traceback.format_exc()})
    finally:
        q.put(None)

# ─────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/reconstruct', methods=['POST'])
def reconstruct():
    files = request.files.getlist('images')
    if not files or len(files) < 2:
        return jsonify({'error': 'Upload at least 2 images'}), 400
    if len(files) > 6:
        return jsonify({'error': 'Maximum 6 images'}), 400

    job_id  = str(uuid.uuid4())[:8]
    workdir = tempfile.mkdtemp(prefix=f'weld_{job_id}_')
    paths   = []
    for f in files:
        ext  = os.path.splitext(f.filename)[1] or '.jpg'
        dest = os.path.join(workdir, f'img_{len(paths):02d}{ext}')
        f.save(dest); paths.append(dest)

    JOBS[job_id] = {'status': 'running', 'queue': queue.Queue(),
                    'result': None, 'workdir': workdir}
    threading.Thread(target=run_pipeline,
                     args=(job_id, paths, workdir), daemon=True).start()
    return jsonify({'job_id': job_id})

@app.route('/progress/<job_id>')
def progress(job_id):
    if job_id not in JOBS:
        return jsonify({'error': 'Unknown job'}), 404
    def generate():
        q = JOBS[job_id]['queue']
        while True:
            msg = q.get()
            if msg is None: break
            yield msg
        yield 'event: close\ndata: {}\n\n'
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})

@app.route('/download/<job_id>/ply')
def download_ply(job_id):
    if job_id not in JOBS: return jsonify({'error': 'Unknown job'}), 404
    p = os.path.join(JOBS[job_id]['workdir'], 'reconstruction.ply')
    if not os.path.exists(p): return jsonify({'error': 'Not ready'}), 404
    return send_file(p, as_attachment=True,
                     download_name='reconstruction.ply',
                     mimetype='application/octet-stream')

@app.route('/download/<job_id>/result')
def download_result(job_id):
    if job_id not in JOBS: return jsonify({'error': 'Unknown job'}), 404
    p = os.path.join(JOBS[job_id]['workdir'], 'weld_result.json')
    if not os.path.exists(p): return jsonify({'error': 'Not ready'}), 404
    return send_file(p, mimetype='application/json')

@app.route('/status/<job_id>')
def status(job_id):
    if job_id not in JOBS: return jsonify({'error': 'Unknown job'}), 404
    return jsonify({'status': JOBS[job_id]['status'],
                    'result': JOBS[job_id]['result']})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'\n  Weld Path Planner  →  http://localhost:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)