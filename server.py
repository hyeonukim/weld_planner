#!/usr/bin/env python3
"""
server.py  —  Weld Path Planner backend
========================================
VGGT (multi-view 3-D reconstruction) provides the shared world frame.
Weld-seam endpoints are detected per view in 2-D, lifted into VGGT's
world frame via `world_points`, then fused across views with a single
PCA fit — no per-view averaging across different camera frames.

Metric3D v2 (YvanYin/Metric3D, torch.hub) is *optional* and used only
to derive an absolute (metres) scale factor by comparing its depth
output to VGGT depth at masked pixels. If Metric3D is unavailable, the
pipeline still returns valid results in VGGT units (no metric scale),
together with a `scale_factor: null` flag.

Dependencies:
  pip install torch torchvision flask flask-cors numpy matplotlib
              opencv-contrib-python Pillow timm
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
# Module-level model cache. The first request pays the load cost
# (~4 GB VGGT + Metric3D download/init); every subsequent request
# reuses the same in-memory models. Avoids Windows page-file OOM
# from re-allocating VGGT-1B per job, and cuts ~30 s off each run.
# ─────────────────────────────────────────────────────────────
_models_lock        = threading.Lock()
_VGGT_MODEL         = None
_M3D_MODEL          = None
_VGGT_LOAD_DEVICE   = None     # device the cached VGGT lives on
_M3D_LOAD_DEVICE    = None     # ditto for Metric3D

# Tunables for the PLY emitted to the browser. Keep these conservative —
# the 2-D canvas viewer chokes well before 200 k points.
PLY_CONF_THRESH = 1.0   # drop VGGT points below this confidence
PLY_DOWNSAMPLE  = 4     # keep every Nth point after filtering (1 = no downsample)

# Fixture-table hole grid: holes are 50 mm centre-to-centre. Hough params
# are guessed from VGGT-native image width (518 px); they auto-scale with W.
HOLE_SPACING_M = 0.050

# ─────────────────────────────────────────────────────────────
# SSE helper
# ─────────────────────────────────────────────────────────────
def _emit(q, event, data):
    if isinstance(data, dict):
        data = json.dumps(data)
    q.put(f"event: {event}\ndata: {data}\n\n")

# ─────────────────────────────────────────────────────────────
# Metric3D v2 — loaded from the official YvanYin/Metric3D torch.hub.
# Used only to derive an absolute (metres) scale factor; endpoint
# detection itself runs in VGGT's shared world frame.
# Loading is optional: if the hub download fails we still return a
# valid (unscaled) result using VGGT alone.
# ─────────────────────────────────────────────────────────────
_M3D_INPUT = (616, 1064)   # required input for Metric3D ViT-L / ViT-S


def _load_metric3d(device):
    try:
        model = torch.hub.load(
            'YvanYin/Metric3D', 'metric3d_vit_large', pretrain=True)
        print('[Metric3D] Loaded ViT-L from torch.hub')
    except Exception as e_l:
        print(f'[Metric3D] ViT-L unavailable ({e_l}), trying ViT-S...')
        try:
            model = torch.hub.load(
                'YvanYin/Metric3D', 'metric3d_vit_small', pretrain=True)
            print('[Metric3D] Loaded ViT-S from torch.hub')
        except Exception as e_s:
            print(f'[Metric3D] Unavailable ({e_s}); '
                  f'continuing without absolute scale.')
            return None
    model.to(device).eval()
    return model


def _run_metric3d(model, image_path, device, intrinsics=None):
    """
    Returns (depth H×W float32 in metres, (fx, fy, cx, cy) at original size).
    Returns (None, None) if model is None.
    """
    if model is None:
        return None, None
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f'Cannot open: {image_path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]

    if intrinsics and len(intrinsics) == 4:
        fx, fy, cx, cy = [float(v) for v in intrinsics]
    else:
        # 70° HFOV fallback — acceptable for ViT-L on phone-camera framing
        fx = fy = w0 / (2 * np.tan(np.radians(70) / 2))
        cx, cy  = w0 / 2.0, h0 / 2.0

    in_h, in_w = _M3D_INPUT
    s_h, s_w = in_h / h0, in_w / w0
    fx_s, fy_s = fx * s_w, fy * s_h
    cx_s, cy_s = cx * s_w, cy * s_h
    rgb_r = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std  = np.array([58.395,  57.12,  57.375], dtype=np.float32)
    img_t = torch.from_numpy(
        ((rgb_r.astype(np.float32) - mean) / std).transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model.inference(
            {'input': img_t},
            cam_model=None,
            pad_info=None,
            normalize_scale=None,
            cam_intrinsic={'fx': fx_s, 'fy': fy_s, 'cx': cx_s, 'cy': cy_s},
        )
    depth_s = pred['prediction'].squeeze().cpu().numpy()    # (in_h, in_w)
    depth   = cv2.resize(depth_s, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return depth.astype(np.float32), (fx, fy, cx, cy)

# ─────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────
def _seam_mask_2d(rgb_uint8):
    """
    Per-view 2-D seam-candidate mask: brightness in LAB + gray Canny edges,
    morphology, then skeletonised to a 1-px line. Heuristic is kept
    intentionally simple here — option #3 (valley / plane-intersection)
    will replace it.
    """
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bright = (lab[:, :, 0] > 160).astype(np.uint8) * 255

    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    combined = cv2.bitwise_or(bright, edges)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k, iterations=1)

    skel, temp = np.zeros_like(combined), combined.copy()
    while True:
        e = cv2.erode(temp, k)
        skel = cv2.bitwise_or(skel, cv2.subtract(temp, cv2.dilate(e, k)))
        temp = e
        if cv2.countNonZero(temp) == 0:
            break
    return skel


def _fit_line_endpoints(pts_xyz, trim_pct=0.05):
    """
    PCA-fit a line through pts_xyz (N,3) in world frame and return the two
    extreme points along the principal axis, trimming `trim_pct` from each
    side to reject outliers. Returns (start_xyz, end_xyz, axis_unit) or
    (None, None, None) if not enough points.
    """
    if pts_xyz is None or len(pts_xyz) < 10:
        return None, None, None
    mean = pts_xyz.mean(axis=0)
    centred = pts_xyz - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    proj = centred @ axis
    lo, hi = np.percentile(proj, [trim_pct * 100, (1 - trim_pct) * 100])
    keep = (proj >= lo) & (proj <= hi)
    proj_k = proj[keep]; pts_k = pts_xyz[keep]
    if len(proj_k) < 2:
        i_min = int(np.argmin(proj)); i_max = int(np.argmax(proj))
        return pts_xyz[i_min], pts_xyz[i_max], axis
    i_min = int(np.argmin(proj_k)); i_max = int(np.argmax(proj_k))
    return pts_k[i_min], pts_k[i_max], axis


def _detect_holes_in_view(rgb_uint8):
    """
    Auto-detect circular dark fixture-table holes via cv2.HoughCircles.
    Returns list of (cx, cy, r) in image pixels. Empty list if none found.
    """
    H, W = rgb_uint8.shape[:2]
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 5)

    # Hole radius heuristics: ~1.5–4% of image width
    min_r = max(4, int(0.014 * W))
    max_r = max(min_r + 4, int(0.040 * W))
    min_dist = max(min_r * 2, int(0.035 * W))

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min_dist,
        param1=80, param2=18,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return []

    out = []
    for cx, cy, r in np.round(circles[0]).astype(int):
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        # Sanity check: hole interior should be noticeably darker than its
        # surroundings (rejects shadows on the workpiece / specular dots).
        y0, y1 = max(0, cy - 2), min(H, cy + 3)
        x0, x1 = max(0, cx - 2), min(W, cx + 3)
        interior = float(gray[y0:y1, x0:x1].mean())
        ry0, ry1 = max(0, cy - r - 4), min(H, cy + r + 5)
        rx0, rx1 = max(0, cx - r - 4), min(W, cx + r + 5)
        ring = float(gray[ry0:ry1, rx0:rx1].mean())
        if ring > 0 and interior < ring * 0.85:
            out.append((int(cx), int(cy), int(r)))
    return out


def _holes_to_scale(world_pts_v, hole_pixels, world_conf_v=None,
                    conf_thresh=1.0, hole_spacing_m=HOLE_SPACING_M):
    """
    From detected holes in one view, recover a (metres per VGGT unit) scale
    by taking the median nearest-neighbour 3-D distance between hole centres
    and dividing by the known 50 mm physical spacing.
    Returns (scale, n_holes_used) or (None, 0).
    """
    if len(hole_pixels) < 2:
        return None, 0
    H, W = world_pts_v.shape[0], world_pts_v.shape[1]
    pts3d = []
    for cx, cy, _r in hole_pixels:
        if not (0 <= cy < H and 0 <= cx < W):
            continue
        p = world_pts_v[cy, cx]
        if np.all(p == 0):
            continue
        if world_conf_v is not None and world_conf_v[cy, cx] < conf_thresh:
            continue
        pts3d.append(p)
    if len(pts3d) < 2:
        return None, 0
    pts3d = np.array(pts3d, np.float64)

    # Nearest-neighbour distance per hole → median = adjacent-hole spacing
    nn = []
    for i in range(len(pts3d)):
        d = np.linalg.norm(pts3d - pts3d[i], axis=1)
        d[i] = np.inf
        nn.append(float(d.min()))
    median_nn = float(np.median(nn))
    if median_nn < 1e-6:
        return None, 0
    return hole_spacing_m / median_nn, len(pts3d)


def _fit_table_frame(hole_pts_world, table_n, scale_m_per_unit):
    """
    Build a right-handed table coordinate frame from detected hole centres:
       origin = centroid (in metres)
       z      = table-plane normal (flipped so workpiece side is +z)
       x      = densest in-plane lattice direction (hole-row axis)
       y      = z × x

    Returns a dict with `R` (3×3 world→table rotation, no scale),
    `origin_m` (3,), `axes` (each 3-vector in world frame). Returns None
    if the hole geometry is degenerate.
    """
    if len(hole_pts_world) < 4 or table_n is None:
        return None
    pts = np.asarray(hole_pts_world, np.float64)
    n = np.asarray(table_n, np.float64)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return None
    n = n / nn

    origin = pts.mean(axis=0)
    centred = pts - origin
    in_plane = centred - (centred @ n)[:, None] * n  # remove normal comp

    # All pairwise in-plane vectors, sorted by length — shortest ones are
    # the lattice rows/cols.
    diffs = []
    for i in range(len(in_plane)):
        for j in range(i + 1, len(in_plane)):
            d = in_plane[j] - in_plane[i]
            m = float(np.linalg.norm(d))
            if m > 1e-6:
                diffs.append((m, d / m))
    diffs.sort(key=lambda t: t[0])
    n_keep = max(4, len(diffs) // 3)
    short_dirs = np.array([d for _, d in diffs[:n_keep]])

    # PCA on outer products → main lattice direction (sign-agnostic)
    M = np.zeros((3, 3))
    for d in short_dirs:
        M += np.outer(d, d)
    _evals, evecs = np.linalg.eigh(M)
    x_axis = evecs[:, -1]
    x_axis = x_axis - (x_axis @ n) * n
    xn = float(np.linalg.norm(x_axis))
    if xn < 1e-6:
        return None
    x_axis = x_axis / xn
    y_axis = np.cross(n, x_axis)

    return {
        'origin_m':   (origin * scale_m_per_unit).tolist(),
        'x_axis':     x_axis.tolist(),
        'y_axis':     y_axis.tolist(),
        'z_axis':     n.tolist(),
        'R_world_to_table': np.column_stack([x_axis, y_axis, n]).T.tolist(),
    }


def _ransac_plane(pts, n_iters=300, thresh=0.005, seed=0):
    """
    Find the largest planar inlier set in `pts` (N, 3).
    Returns (n_unit, d, inlier_mask) such that  n·p + d ≈ 0  on the plane.
    """
    N = len(pts)
    if N < 3:
        return None, None, np.zeros(N, bool)
    rng = np.random.default_rng(seed)
    best_count = 0
    best_n = best_d = best_mask = None
    for _ in range(n_iters):
        i = rng.choice(N, 3, replace=False)
        a, b, c = pts[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -float(n @ a)
        dists = np.abs(pts @ n + d)
        mask = dists < thresh
        cnt = int(mask.sum())
        if cnt > best_count:
            best_count = cnt
            best_n, best_d, best_mask = n, d, mask
    if best_mask is None or best_mask.sum() < 10:
        return best_n, best_d, (best_mask if best_mask is not None
                                else np.zeros(N, bool))
    # Refit on inliers via PCA (smallest-eigenvector = normal)
    inl = pts[best_mask]
    centroid = inl.mean(axis=0)
    cov = np.cov(inl.T)
    evals, evecs = np.linalg.eigh(cov)
    n = evecs[:, 0]
    d = -float(n @ centroid)
    mask = np.abs(pts @ n + d) < thresh
    return n, d, mask


def _segment_workpiece(world_pts, conf=None, conf_thresh=1.0):
    """
    RANSAC the dominant plane (= the fixture table) and return the points
    that lie significantly off it (= the workpiece) plus the table plane
    params.

    Returns:
      table_n, table_d            — plane params (or None on failure)
      workpiece_pts (M, 3)        — points well off the table plane
      scene_diag                  — bbox diagonal of all valid points
    """
    valid = ~np.all(world_pts == 0, axis=1)
    if conf is not None:
        valid &= conf > conf_thresh
    pts = world_pts[valid].astype(np.float64)
    if len(pts) < 100:
        return None, None, pts, 1.0

    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0
    thresh = 0.003 * diag    # table flatness tolerance (~0.3% of diagonal)

    ss_n = min(50000, len(pts))
    if ss_n < len(pts):
        idx = np.random.default_rng(0).choice(len(pts), ss_n, replace=False)
        ss = pts[idx]
    else:
        ss = pts

    n, d, _ = _ransac_plane(ss, n_iters=300, thresh=thresh)
    if n is None:
        return None, None, pts, diag

    # Orient the normal so it points to the workpiece side: take points that
    # are far off the plane and flip the normal if their mean signed distance
    # is negative. This keeps table_frame.z_axis = "up out of the table".
    signed = pts @ n + d
    off_mask = np.abs(signed) > thresh * 3
    if int(off_mask.sum()) and float(signed[off_mask].mean()) < 0:
        n, d = -n, -d
        signed = -signed

    workpiece = pts[signed > thresh * 3]
    return n, d, workpiece, diag


def _detect_seam_geometric(workpiece_pts, scene_diag):
    """
    Auto-pick t-joint vs groove and return (start_3d, end_3d, method).

      t-joint  : two strong workpiece planes → seam = plane1 ∩ plane2
      groove   : one plane + offset ridge    → seam = PCA on offset points

    Returns (None, None, 'failed') if neither path produces a seam.
    """
    if len(workpiece_pts) < 50:
        return None, None, 'failed:too-few-workpiece-pts'

    thresh = 0.005 * scene_diag

    ss_n = min(30000, len(workpiece_pts))
    if ss_n < len(workpiece_pts):
        idx = np.random.default_rng(1).choice(len(workpiece_pts), ss_n, replace=False)
        ss = workpiece_pts[idx]
    else:
        ss = workpiece_pts

    n1, d1, mask1 = _ransac_plane(ss, n_iters=300, thresh=thresh, seed=1)
    if n1 is None:
        return None, None, 'failed:no-plane'
    plane1_count = int(mask1.sum())

    n2 = d2 = None
    plane2_count = 0
    rem = ss[~mask1]
    if len(rem) >= 100:
        n2, d2, mask2 = _ransac_plane(rem, n_iters=300, thresh=thresh, seed=2)
        plane2_count = int(mask2.sum()) if mask2 is not None else 0

    is_tjoint = (n2 is not None
                 and plane2_count > 0.30 * plane1_count
                 and abs(float(n1 @ n2)) < 0.95)  # planes not parallel

    if is_tjoint:
        line_dir = np.cross(n1, n2)
        ln = np.linalg.norm(line_dir)
        if ln > 1e-6:
            line_dir = line_dir / ln
            # Closest point of two planes to origin (Moore–Penrose on 2×3 system)
            A = np.vstack([n1, n2])
            b = -np.array([d1, d2])
            try:
                p0, *_ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                p0 = None
            if p0 is not None:
                # Seam-zone = workpiece points near BOTH planes
                d1_all = np.abs(workpiece_pts @ n1 + d1)
                d2_all = np.abs(workpiece_pts @ n2 + d2)
                zone = (d1_all < thresh * 2.0) & (d2_all < thresh * 2.0)
                seam_pts = workpiece_pts[zone]
                if len(seam_pts) < 4:
                    seam_pts = workpiece_pts
                rel = seam_pts - p0
                proj = rel @ line_dir
                lo, hi = np.percentile(proj, [5, 95])
                keep = (proj >= lo) & (proj <= hi)
                if keep.sum() >= 2:
                    proj_k = proj[keep]; pts_k = seam_pts[keep]
                    return (pts_k[int(np.argmin(proj_k))],
                            pts_k[int(np.argmax(proj_k))],
                            f't-joint:plane-intersection,'
                            f'p1={plane1_count},p2={plane2_count},seam={len(seam_pts)}')

    # Groove path: ridge points = workpiece points significantly off plane1
    signed = workpiece_pts @ n1 + d1
    ridge_mask = np.abs(signed) > thresh * 1.5
    ridge_pts = workpiece_pts[ridge_mask]
    if len(ridge_pts) < 10:
        ridge_pts = workpiece_pts
    s, e, _axis = _fit_line_endpoints(ridge_pts, trim_pct=0.05)
    if s is None:
        return None, None, 'failed:groove-pca'
    return s, e, (f'groove:ridge-pca,'
                  f'plane={plane1_count},ridge={len(ridge_pts)}')


def _vggt_to_metric_scale(vggt_depth_v, metric_depth_v, mask_uint8,
                          conf_v=None, conf_thresh=1.0):
    """
    Estimate a single scale factor s such that  metres ≈ s · VGGT_units,
    by comparing VGGT depth to Metric3D depth at masked pixels of one view.
    Returns (scale, n_used) or (None, 0) if Metric3D depth is None.
    """
    if metric_depth_v is None:
        return None, 0
    # Resize Metric3D depth to VGGT-native (H, W) so we can index by mask
    H, W = vggt_depth_v.shape
    if metric_depth_v.shape != (H, W):
        metric_depth_v = cv2.resize(metric_depth_v, (W, H),
                                    interpolation=cv2.INTER_LINEAR)
    ys, xs = np.where(mask_uint8 > 0)
    if len(ys) == 0:
        ys, xs = np.where(vggt_depth_v > 0)
    v = vggt_depth_v[ys, xs].astype(np.float64)
    m = metric_depth_v[ys, xs].astype(np.float64)
    good = (v > 1e-6) & (m > 0.05) & (m < 5.0)
    if conf_v is not None:
        good &= conf_v[ys, xs] > conf_thresh
    if good.sum() < 50:
        return None, int(good.sum())
    ratios = m[good] / v[good]
    # Median is robust to seam highlights and Metric3D outliers
    return float(np.median(ratios)), int(good.sum())

# ─────────────────────────────────────────────────────────────
# Cached model accessors. Returns (model, was_loaded_now: bool)
# so the caller can emit a different progress message on cache hit.
# ─────────────────────────────────────────────────────────────
def _get_vggt(device):
    global _VGGT_MODEL, _VGGT_LOAD_DEVICE
    with _models_lock:
        if _VGGT_MODEL is not None and _VGGT_LOAD_DEVICE == device:
            return _VGGT_MODEL, False
        from vggt.models.vggt import VGGT
        m = VGGT.from_pretrained('facebook/VGGT-1B').to(device)
        m.eval()
        _VGGT_MODEL, _VGGT_LOAD_DEVICE = m, device
        return _VGGT_MODEL, True


def _get_metric3d_cached(device):
    global _M3D_MODEL, _M3D_LOAD_DEVICE
    with _models_lock:
        if _M3D_LOAD_DEVICE == device:
            # Cached value may legitimately be None (load failed once).
            return _M3D_MODEL, False
        _M3D_MODEL       = _load_metric3d(device)
        _M3D_LOAD_DEVICE = device
        return _M3D_MODEL, True


# ─────────────────────────────────────────────────────────────
# Main pipeline  (background thread)
# ─────────────────────────────────────────────────────────────
def run_pipeline(job_id, image_paths, workdir):
    q = JOBS[job_id]['queue']
    try:
        _import_ml()
        import matplotlib; matplotlib.use('Agg')
        from vggt.utils.load_fn import load_and_preprocess_images

        # 1 — device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype  = torch.float32
        if device == 'cuda':
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 \
                    else torch.float16
        _emit(q, 'progress', {'step': 1, 'total': 7,
                               'msg': f'Running on {device.upper()}'})

        # 2 — VGGT (cached after first request)
        vggt, vggt_loaded = _get_vggt(device)
        _emit(q, 'progress', {'step': 2, 'total': 7,
                              'msg': ('Loading VGGT-1B…' if vggt_loaded
                                      else 'VGGT-1B ready (cached)')})

        # 3 — Metric3D v2 (cached after first request)
        m3d, m3d_loaded = _get_metric3d_cached(device)
        _emit(q, 'progress', {'step': 3, 'total': 7,
                              'msg': ('Loading Metric3D v2…' if m3d_loaded
                                      else 'Metric3D v2 ready (cached)')})

        # 4 — VGGT reconstruction
        _emit(q, 'progress', {'step': 4, 'total': 7,
                               'msg': 'Running VGGT 3-D reconstruction…'})
        images_t = load_and_preprocess_images(image_paths).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == 'cuda'), dtype=dtype):
                preds = vggt(images_t)

        # VGGT outputs:
        #   world_points       [1, V, H, W, 3]
        #   world_points_conf  [1, V, H, W]
        #   depth              [1, V, H, W, 1]
        pts3d  = preds['world_points'].cpu().numpy()
        colors = images_t.permute(0, 2, 3, 1).cpu().numpy()       # (V, H, W, 3)

        wp = pts3d[0] if pts3d.ndim == 5 else pts3d                # (V, H, W, 3)
        wpc = preds.get('world_points_conf', None)
        if wpc is not None:
            wpc = wpc.cpu().numpy()
            if wpc.ndim == 4: wpc = wpc[0]                         # (V, H, W)
        vggt_depth = preds.get('depth', None)
        if vggt_depth is not None:
            vggt_depth = vggt_depth.cpu().numpy()
            while vggt_depth.ndim > 3: vggt_depth = vggt_depth[0]  # (V, H, W)

        V = wp.shape[0]
        CONF_THRESH = 1.0

        # 4b — Fixture-hole detection (per view) → absolute scale
        _emit(q, 'progress', {'step': 4, 'total': 7,
                              'msg': 'Detecting 50 mm fixture holes for scale…'})
        hole_scale_samples = []        # list of (scale_m_per_unit, n_holes)
        hole_counts        = []        # per-view detected count
        holes_world_all    = []        # accumulated 3-D hole centres
        for v in range(V):
            rgb_v = (colors[v] * 255).clip(0, 255).astype(np.uint8)
            holes = _detect_holes_in_view(rgb_v)
            hole_counts.append(len(holes))
            if not holes:
                continue
            s, n = _holes_to_scale(
                wp[v], holes,
                world_conf_v=(wpc[v] if wpc is not None else None),
                conf_thresh=CONF_THRESH)
            if s is not None:
                hole_scale_samples.append((s, n))
            for cx, cy, _r in holes:
                p = wp[v, cy, cx]
                if np.all(p == 0):
                    continue
                if wpc is not None and wpc[v, cy, cx] < CONF_THRESH:
                    continue
                holes_world_all.append(p)

        hole_scale_factor = None
        if hole_scale_samples:
            arr = np.array(hole_scale_samples)
            order = np.argsort(arr[:, 0])
            sorted_s = arr[order, 0]; sorted_n = arr[order, 1]
            cum = np.cumsum(sorted_n)
            hole_scale_factor = float(sorted_s[np.searchsorted(cum, cum[-1] / 2.0)])

        _emit(q, 'progress', {'step': 4, 'total': 7,
                              'msg': f'Holes per view: {hole_counts} → scale '
                                     + (f'{hole_scale_factor:.4f} m/unit'
                                        if hole_scale_factor else 'unavailable')})

        # 5 — Write PLY (binary little-endian, conf-filtered, downsampled)
        _emit(q, 'progress', {'step': 5, 'total': 7, 'msg': 'Writing point cloud…'})
        ply_path  = os.path.join(workdir, 'reconstruction.ply')
        flat_pts  = pts3d.reshape(-1, 3).astype(np.float32)
        flat_cols = (colors.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)

        keep = ~np.all(flat_pts == 0, axis=1)
        n_raw = int(keep.sum())
        if wpc is not None:
            keep &= wpc.reshape(-1) > PLY_CONF_THRESH
        n_conf = int(keep.sum())

        flat_pts  = flat_pts[keep]
        flat_cols = flat_cols[keep]

        if PLY_DOWNSAMPLE > 1 and len(flat_pts) > 0:
            flat_pts  = flat_pts[::PLY_DOWNSAMPLE]
            flat_cols = flat_cols[::PLY_DOWNSAMPLE]
        n_pts = len(flat_pts)

        # Pack as binary LE vertex stream (3·float32 + 3·uint8 = 15 B/vertex)
        vert_dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                            ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        verts = np.empty(n_pts, dtype=vert_dt)
        verts['x'] = flat_pts[:, 0]; verts['y'] = flat_pts[:, 1]; verts['z'] = flat_pts[:, 2]
        verts['r'] = flat_cols[:, 0]; verts['g'] = flat_cols[:, 1]; verts['b'] = flat_cols[:, 2]

        with open(ply_path, 'wb') as f:
            f.write(b'ply\n')
            f.write(b'format binary_little_endian 1.0\n')
            f.write(f'comment vggt-weldplanner conf>{PLY_CONF_THRESH} '
                    f'downsample={PLY_DOWNSAMPLE} raw={n_raw} '
                    f'conf_kept={n_conf} written={n_pts}\n'.encode('ascii'))
            f.write(f'element vertex {n_pts}\n'.encode('ascii'))
            f.write(b'property float x\nproperty float y\nproperty float z\n')
            f.write(b'property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write(b'end_header\n')
            f.write(verts.tobytes())

        _emit(q, 'progress', {'step': 5, 'total': 7,
                               'msg': f'Point cloud: {n_pts:,} pts '
                                      f'(raw {n_raw:,} → conf {n_conf:,} → {n_pts:,})'})

        # 6 — Geometric seam detection in VGGT world frame
        #     Strategy: segment workpiece (above table plane), then either
        #     intersect two workpiece planes (t-joint) or PCA-fit ridge
        #     points (groove). Per-view 2-D mask is kept as a fallback.
        _emit(q, 'progress', {'step': 6, 'total': 7,
                              'msg': 'Segmenting workpiece (table-plane RANSAC)…'})

        H, W       = wp.shape[1], wp.shape[2]
        per_frame  = []
        scale_samples = []   # Metric3D fallback samples

        flat_pts3d = wp.reshape(-1, 3)
        flat_conf  = wpc.reshape(-1) if wpc is not None else None
        table_n, table_d, workpiece_pts, scene_diag = _segment_workpiece(
            flat_pts3d, conf=flat_conf, conf_thresh=CONF_THRESH)

        _emit(q, 'progress', {'step': 6, 'total': 7,
                              'msg': f'Workpiece: {len(workpiece_pts):,} pts '
                                     f'(scene diag {scene_diag:.3f})'})

        detect_method = 'none'
        weld_s_w = weld_e_w = None
        if table_n is not None and len(workpiece_pts) >= 50:
            _emit(q, 'progress', {'step': 6, 'total': 7,
                                  'msg': 'Geometric seam (plane intersection / ridge PCA)…'})
            weld_s_w, weld_e_w, detect_method = _detect_seam_geometric(
                workpiece_pts, scene_diag)

        # Fallback: per-view 2-D mask → world-frame PCA (option-#1 path)
        if weld_s_w is None:
            _emit(q, 'progress', {'step': 6, 'total': 7,
                                  'msg': 'Geometric failed — falling back to 2-D mask fusion…'})
            seam_world = []
            for i in range(V):
                rgb_v  = (colors[i] * 255).clip(0, 255).astype(np.uint8)
                mask_v = _seam_mask_2d(rgb_v)
                ys, xs = np.where(mask_v > 0)
                if not len(ys):
                    continue
                pts_v = wp[i, ys, xs]
                if wpc is not None:
                    keep = (wpc[i, ys, xs] > CONF_THRESH) & ~np.all(pts_v == 0, axis=1)
                else:
                    keep = ~np.all(pts_v == 0, axis=1)
                pts_v = pts_v[keep]
                if len(pts_v):
                    seam_world.append(pts_v.astype(np.float64))
            all_pts = (np.concatenate(seam_world, axis=0)
                       if seam_world else np.zeros((0, 3), np.float64))
            weld_s_w, weld_e_w, _ax = _fit_line_endpoints(all_pts, trim_pct=0.05)
            if weld_s_w is not None:
                detect_method = f'fallback:2d-mask,seam_pts={len(all_pts)}'

        # Per-view Metric3D scale samples (independent of detection method)
        if m3d is not None and vggt_depth is not None:
            for i, img_path in enumerate(image_paths):
                _emit(q, 'progress', {'step': 6, 'total': 7,
                                      'msg': f'Metric3D scale {i+1}/{V}…'})
                m_depth, _ = _run_metric3d(m3d, img_path, device,
                                           intrinsics=JOBS[job_id].get('intrinsics'))
                # Use the whole image (not seam mask) for a stable scale estimate
                full_mask = np.ones(vggt_depth[i].shape, np.uint8) * 255
                s, n = _vggt_to_metric_scale(vggt_depth[i], m_depth, full_mask,
                                             conf_v=wpc[i] if wpc is not None else None,
                                             conf_thresh=CONF_THRESH)
                if s is not None:
                    scale_samples.append((s, n))
                per_frame.append({
                    'image':       os.path.basename(img_path),
                    'scale_ratio': s,
                    'scale_n':     n,
                })
        else:
            for i, img_path in enumerate(image_paths):
                per_frame.append({'image': os.path.basename(img_path)})

        if weld_s_w is None:
            # Ultimate fallback: depth extremes (preserves prior behaviour)
            flat_all = wp.reshape(-1, 3)
            if wpc is not None:
                hi = (wpc.reshape(-1) > CONF_THRESH) & ~np.all(flat_all == 0, axis=1)
                flat_all = flat_all[hi]
            if len(flat_all):
                weld_s_w = flat_all[np.argmin(flat_all[:, 2])]
                weld_e_w = flat_all[np.argmax(flat_all[:, 2])]
                detect_method = 'fallback:depth-extremes'
            else:
                weld_s_w = weld_e_w = np.zeros(3)
                detect_method = 'fallback:empty'

        seam_len_vggt = float(np.linalg.norm(weld_e_w - weld_s_w))

        # Absolute-scale factor: holes (preferred) > Metric3D depth-ratio
        metric3d_scale = None
        if scale_samples:
            arr = np.array(scale_samples)            # (K, 2): scale, n_used
            order = np.argsort(arr[:, 0])
            sorted_s = arr[order, 0]; sorted_n = arr[order, 1]
            cum = np.cumsum(sorted_n)
            metric3d_scale = float(sorted_s[np.searchsorted(cum, cum[-1] / 2.0)])

        if hole_scale_factor is not None:
            scale_factor = hole_scale_factor
            scale_source = 'holes'
        elif metric3d_scale is not None:
            scale_factor = metric3d_scale
            scale_source = 'metric3d'
        else:
            scale_factor = None
            scale_source = None

        seam_len_m = (seam_len_vggt * scale_factor) if scale_factor else None

        # Table coordinate frame (origin + axes) from detected hole lattice.
        table_frame = None
        if hole_scale_factor and table_n is not None and len(holes_world_all) >= 4:
            table_frame = _fit_table_frame(
                np.array(holes_world_all), table_n, hole_scale_factor)

        # Keep weld_start/end in raw VGGT units so they live in the same
        # coordinate frame as the PLY the viewer renders. Metric and table
        # frame versions go in separate fields.
        s_out   = weld_s_w.tolist()
        e_out   = weld_e_w.tolist()
        s_out_m = (weld_s_w * scale_factor).tolist() if scale_factor else None
        e_out_m = (weld_e_w * scale_factor).tolist() if scale_factor else None

        # Also project endpoints into the table frame when available.
        weld_start_table = weld_end_table = None
        if table_frame is not None and s_out_m is not None:
            R = np.array(table_frame['R_world_to_table'])
            origin_m = np.array(table_frame['origin_m'])
            ws_m = np.array(s_out_m); we_m = np.array(e_out_m)
            weld_start_table = (R @ (ws_m - origin_m)).tolist()
            weld_end_table   = (R @ (we_m - origin_m)).tolist()

        result = {
            'ply_ready':      True,
            'point_count':    n_pts,
            'per_frame':      per_frame,
            # Cloud-frame (raw VGGT units) — matches the PLY served to the viewer.
            'weld_start':     s_out,
            'weld_end':       e_out,
            # Metric (world axes) versions — null if no scale factor.
            'weld_start_m':   s_out_m,
            'weld_end_m':     e_out_m,
            'seam_length_m':  (seam_len_m if seam_len_m is not None else seam_len_vggt),
            'seam_length_mm': ((seam_len_m if seam_len_m is not None
                                else seam_len_vggt) * 1000),
            'seam_length_vggt_units': seam_len_vggt,
            'scale_factor':   scale_factor,
            'scale_source':   scale_source,         # 'holes' | 'metric3d' | None
            'hole_scale_factor':    hole_scale_factor,
            'metric3d_scale_factor': metric3d_scale,
            'hole_counts_per_view':  hole_counts,
            'frame':          'world (VGGT units, matches PLY)',
            'detection_method': detect_method,
            'scene_diag_vggt':  scene_diag,
            'table_frame':       table_frame,        # null if not enough holes
            'weld_start_table_m': weld_start_table,  # null if no table_frame
            'weld_end_table_m':   weld_end_table,
        }
        avg_len = seam_len_m if seam_len_m is not None else seam_len_vggt
        with open(os.path.join(workdir, 'weld_result.json'), 'w') as f:
            json.dump(result, f, indent=2)

        JOBS[job_id].update({'result': result, 'status': 'done'})
        unit_msg = (f'Done — seam {avg_len*1000:.1f} mm' if scale_factor
                    else f'Done — seam {seam_len_vggt:.4f} VGGT units '
                         f'(no Metric3D scale)')
        _emit(q, 'progress', {'step': 7, 'total': 7, 'msg': unit_msg})
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

    # Optional real camera intrinsics — passed as JSON form field:
    # { "fx": 3200, "fy": 3200, "cx": 2016, "cy": 1512 }
    intrinsics = None
    intr_raw = request.form.get('intrinsics')
    if intr_raw:
        try:
            d = json.loads(intr_raw)
            intrinsics = [float(d['fx']), float(d['fy']),
                          float(d['cx']), float(d['cy'])]
        except Exception:
            pass   # bad JSON → fall back to 70° HFOV guess

    JOBS[job_id] = {'status': 'running', 'queue': queue.Queue(),
                    'result': None, 'workdir': workdir,
                    'intrinsics': intrinsics}
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

# ─────────────────────────────────────────────────────────────
# ROS 2 Bridge — publish weld path to robot via ros2 topic pub
# ─────────────────────────────────────────────────────────────
# POST /send_to_robot
# Body (JSON):
#   {
#     "start":     [x, y, z],   ← metres, camera frame
#     "end":       [x, y, z],
#     "armed":     false,        ← true = live weld, false = dry-run
#     "speed_mm_s": 5.0,
#     "procedure": 1,
#     "schedule":  1
#   }
#
# The server calls `ros2 topic pub` (one-shot) to publish to:
#   /weld/seam_start   geometry_msgs/msg/Point
#   /weld/seam_end     geometry_msgs/msg/Point
#   /weld/speed        std_msgs/msg/Float32
#   /weld/procedure    std_msgs/msg/Int32
#   /weld/schedule     std_msgs/msg/Int32
#   /gui/start_weld    std_msgs/msg/Bool   (only if armed=true)
#
# The robot_gui_listener node must be running and subscribed to
# /weld/seam_start and /weld/seam_end — add those subscriptions
# in robot_gui_listener if they don't exist yet (see weld_ros_bridge.py).
# ─────────────────────────────────────────────────────────────

import subprocess, shutil

def _ros2_pub_once(topic, msg_type, value_json):
    """
    Publish a single message via `ros2 topic pub --once`.
    Returns (success:bool, stderr:str).
    Requires ros2 to be on PATH (source /opt/ros/humble/setup.bash).
    """
    if not shutil.which('ros2'):
        return False, 'ros2 not found on PATH — source /opt/ros/humble/setup.bash'
    cmd = ['ros2', 'topic', 'pub', '--once', topic, msg_type, value_json]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.returncode == 0, r.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, 'ros2 pub timed out'
    except Exception as e:
        return False, str(e)

@app.route('/send_to_robot', methods=['POST'])
def send_to_robot():
    """
    Publish weld path to the ROS 2 network so the Fanuc CRx controller
    can execute it via robot_gui_listener.
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    try:
        start     = data['start']          # [x, y, z] metres
        end       = data['end']            # [x, y, z] metres
        armed     = bool(data.get('armed', False))
        speed     = float(data.get('speed_mm_s', 5.0))
        procedure = int(data.get('procedure', 1))
        schedule  = int(data.get('schedule',  1))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({'error': f'Bad payload: {e}'}), 400

    errors = []
    results = {}

    # Publish seam start (metres → kept as-is; listener converts to robot frame)
    ok, err = _ros2_pub_once(
        '/weld/seam_start',
        'geometry_msgs/msg/Point',
        f'{{x: {start[0]:.6f}, y: {start[1]:.6f}, z: {start[2]:.6f}}}'
    )
    results['seam_start'] = 'ok' if ok else err
    if not ok: errors.append(f'seam_start: {err}')

    ok, err = _ros2_pub_once(
        '/weld/seam_end',
        'geometry_msgs/msg/Point',
        f'{{x: {end[0]:.6f}, y: {end[1]:.6f}, z: {end[2]:.6f}}}'
    )
    results['seam_end'] = 'ok' if ok else err
    if not ok: errors.append(f'seam_end: {err}')

    # Publish motion parameters
    ok, err = _ros2_pub_once('/weld/speed',     'std_msgs/msg/Float32', f'{{data: {speed}}}')
    results['speed'] = 'ok' if ok else err

    ok, err = _ros2_pub_once('/weld/procedure', 'std_msgs/msg/Int32',   f'{{data: {procedure}}}')
    results['procedure'] = 'ok' if ok else err

    ok, err = _ros2_pub_once('/weld/schedule',  'std_msgs/msg/Int32',   f'{{data: {schedule}}}')
    results['schedule'] = 'ok' if ok else err

    # Trigger execution only when armed
    if armed:
        ok, err = _ros2_pub_once('/gui/start_weld', 'std_msgs/msg/Bool', '{data: true}')
        results['start_weld'] = 'ok' if ok else err
        if not ok: errors.append(f'start_weld: {err}')
    else:
        results['start_weld'] = 'skipped (armed=false — dry-run only)'

    status_code = 200 if not errors else 207   # 207 = partial success
    return jsonify({
        'armed':   armed,
        'results': results,
        'errors':  errors,
        'message': 'Path sent to robot.' if not errors else
                   'Path sent with some errors — check ROS2 connectivity.'
    }), status_code

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'\n  Weld Path Planner  →  http://localhost:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)