#!/usr/bin/env python3
"""
server.py  —  Weld Path Planner backend
========================================
VGGT (multi-view 3-D reconstruction) provides the shared world frame.
Seam candidates are extracted from shared world geometry and traced by
local connectivity. Planar intersections yield straight paths; narrow
ridges and explicitly selected base contacts can yield curved paths.

Metric3D v2 (YvanYin/Metric3D, torch.hub) is *optional* and used only
to derive an absolute (metres) scale factor by comparing its depth
output to VGGT depth at masked pixels. If Metric3D is unavailable, the
pipeline still returns valid results in VGGT units (no metric scale),
together with a `scale_factor: null` flag.

Dependencies:
  pip install torch torchvision flask flask-cors numpy matplotlib
              opencv-contrib-python Pillow timm
  # Metric3D (optional absolute scale) additionally needs:
  pip install mmengine mmcv-lite
"""

import os, json, uuid, threading, tempfile, traceback, queue, math
from seam_path import trace_path, trace_loop, trim_end_hooks, depth_views, align_depth
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

# Scene crop: keep only the fixture table and the object(s) resting on it,
# discarding the wall / floor / background and any hovering equipment. Applies
# to both the rendered point cloud and seam detection.
CROP_TO_TABLE    = True    # master switch; False → full-scene (legacy) behaviour
CROP_MARGIN      = 0.02    # expand the table footprint by this fraction
CROP_HEIGHT_FRAC = 0.5     # max object height as a fraction of table extent
CROP_VOXEL_FRAC  = 0.015   # cluster voxel size as a fraction of scene diagonal

# Workpiece height prominence: a cluster must rise at least this many multiples
# of the table-flatness tolerance above the table to count as workpiece.
# Rejects rust/discolouration patches whose noisy depth pokes a few mm above
# the plane and would otherwise drag the seam path onto fake corners.
PART_MIN_HEIGHT_MULT = 8

# Workpiece cut above the table, in robust standard deviations of the table's
# own surface noise (bounded to [0.1, 3] × the table-fit tolerance).
PART_CUT_SIGMA = 4
# Inter-part seam: members must meet at a real angle, and this fraction of the
# surface above the base must face along one of the two plates (measured:
# real upright T ≈ 0.48, pipes / curved walls ≤ 0.23).
MEMBER_MIN_ANGLE_DEG = 30
MEMBER_FLAT_FRAC = 0.4

# Fixture-table hole grid: holes are 50 mm centre-to-centre. Hough params
# are guessed from VGGT-native image width (518 px); they auto-scale with W.
HOLE_SPACING_M = 0.050

# Metric3D model variant: 'small' (ViT-S) is much faster to load/run; 'large'
# (ViT-L) is more accurate raw. Since Metric3D is anchored to the fixture holes,
# the small model's accuracy gap is absorbed by the learned calibration — so we
# default to the faster one. Speed > raw Metric3D accuracy here.
M3D_VARIANT = 'small'

# Headline measurement scale source:
#   'metric3d+holes' — Metric3D anchored to the fixture holes. Holes are treated
#                      as ground truth when present (Metric3D's scale snaps to
#                      them, and the correction is remembered); off-fixture,
#                      Metric3D — corrected by the saved calibration — carries the
#                      scale. Metric3D runs every job (~3 fast ViT-S passes).
#   'holes'          — holes preferred; Metric3D only when no holes are found
#                      (fast path — Metric3D skipped on fixture shots).
# VGGT supplies the seam *geometry* in both; the rendered cloud stays raw.
MEASUREMENT_SCALE = 'metric3d+holes'

# Metric3D scale calibration. Metric3D's raw metric scale has a systematic
# per-camera error (unknown focal length + model bias on shiny metal). When the
# 50 mm fixture holes are visible we treat their scale as ground truth, learn a
# multiplicative correction so Metric3D matches it, persist it, and reuse it on
# later fixture-less shots from the same camera. Pure scale multiplier →
# resolution-independent.
_M3D_CALIB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'camera_calib.json')
_M3D_CORRECTION = None      # multiplier applied to Metric3D's raw scale


def _load_m3d_calib():
    # A correction is only valid for the model variant it was learned with
    # (ViT-S and ViT-L have different raw bias), so ignore mismatches → re-learn.
    global _M3D_CORRECTION
    try:
        with open(_M3D_CALIB_PATH) as f:
            d = json.load(f)
        c = float(d.get('metric3d_correction'))
        if d.get('variant', M3D_VARIANT) == M3D_VARIANT and 0 < c < 1e6:
            _M3D_CORRECTION = c
        else:
            _M3D_CORRECTION = None
    except Exception:
        _M3D_CORRECTION = None


def _save_m3d_calib(correction):
    global _M3D_CORRECTION
    _M3D_CORRECTION = float(correction)
    try:
        with open(_M3D_CALIB_PATH, 'w') as f:
            json.dump({'metric3d_correction': _M3D_CORRECTION,
                       'variant': M3D_VARIANT}, f)
    except Exception as e:
        print(f'[Metric3D] could not persist calibration: {e}')


_load_m3d_calib()

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


def _load_metric3d(device, variant='small'):
    # NOTE: the hub repo needs `mmengine` + `mmcv-lite` importable (it imports
    # mmcv.utils / mmengine.Config). Without them the load raises ModuleNotFound
    # and Metric3D is silently skipped. Weights download once (ViT-S ~ a few
    # hundred MB, ViT-L ~1.3 GB). Try the requested variant first, then fall back.
    order = (['metric3d_vit_small', 'metric3d_vit_large'] if variant == 'small'
             else ['metric3d_vit_large', 'metric3d_vit_small'])
    model = None
    for name in order:
        try:
            model = torch.hub.load('YvanYin/Metric3D', name,
                                   pretrain=True, trust_repo=True)
            print(f'[Metric3D] Loaded {name} from torch.hub')
            break
        except Exception as e:
            print(f'[Metric3D] {name} unavailable ({e})')
    if model is None:
        print('[Metric3D] Unavailable; continuing without absolute scale.')
        return None
    model.to(device).eval()
    return model


_M3D_CANONICAL_FOCAL = 1000.0   # canonical-camera focal used by Metric3D ViT


def _run_metric3d(model, image_path, device, intrinsics=None, focal_norm=None):
    """
    Returns (depth H×W float32 in metres, (fx, fy, cx, cy) at original size).
    Returns (None, None) if model is None.

    Follows the official Metric3D ViT recipe: aspect-preserving resize + pad to
    the model input, run `inference(data) -> (pred_depth, conf, out)`, then undo
    the canonical-camera transform (depth_metric = depth_canonical · fx_resized /
    1000) so the result is in metres. Earlier code targeted a stale API
    (`inference(..., cam_intrinsic=...)['prediction']`) which raised on the
    current repo, leaving Metric3D silently disabled.

    Focal length priority: explicit `intrinsics` > `focal_norm` (fx/imageWidth,
    e.g. from a user FOV) > 70° HFOV guess. Metric3D depth scales linearly with
    the focal, so this directly controls the absolute scale.
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
    elif focal_norm:
        fx = fy = float(focal_norm) * w0
        cx, cy  = w0 / 2.0, h0 / 2.0
    else:
        # 70° HFOV fallback — acceptable for ViT on phone-camera framing
        fx = fy = w0 / (2 * np.tan(np.radians(70) / 2))
        cx, cy  = w0 / 2.0, h0 / 2.0

    # Aspect-preserving resize into the model input, then pad with the mean RGB.
    in_h, in_w = _M3D_INPUT
    scale = min(in_w / w0, in_h / h0)
    rw, rh = int(round(w0 * scale)), int(round(h0 * scale))
    rgb_r = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = in_h - rh, in_w - rw
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    pl, pr = pad_w // 2, pad_w - pad_w // 2
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std  = np.array([58.395,  57.12,  57.375], dtype=np.float32)
    rgb_p = cv2.copyMakeBorder(rgb_r, pt, pb, pl, pr,
                               cv2.BORDER_CONSTANT, value=mean.tolist())

    img_t = torch.from_numpy(
        ((rgb_p.astype(np.float32) - mean) / std).transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_depth, _conf, _out = model.inference({'input': img_t})

    pred_depth = pred_depth.squeeze()
    # Strip the padding, then undo the canonical-camera scaling → metres.
    pred_depth = pred_depth[pt:pred_depth.shape[0] - pb,
                            pl:pred_depth.shape[1] - pr]
    pred_depth = pred_depth * (fx * scale / _M3D_CANONICAL_FOCAL)
    depth_s = pred_depth.cpu().numpy()                       # (rh, rw)
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


def _fit_table(flat_pts, flat_conf=None, conf_thresh=1.0):
    """
    RANSAC the dominant plane (= the fixture table) over the valid points and
    return a dict describing the table frame, or None if no usable plane.

    Keys: n, d (oriented plane, normal → workpiece side); u, v (orthonormal
    in-plane basis); u_lo/u_hi/v_lo/v_hi (robust footprint bounds); thresh
    (flatness tolerance); height_cap (max object height); scene_diag.
    """
    valid = ~np.all(flat_pts == 0, axis=1)
    if flat_conf is not None:
        valid &= flat_conf > conf_thresh
    pts = flat_pts[valid].astype(np.float64)
    if len(pts) < 100:
        return None

    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0
    thresh = 0.003 * diag

    ss_n = min(50000, len(pts))
    if ss_n < len(pts):
        idx = np.random.default_rng(0).choice(len(pts), ss_n, replace=False)
        ss = pts[idx]
    else:
        ss = pts

    n, d, _ = _ransac_plane(ss, n_iters=300, thresh=thresh)
    if n is None:
        return None

    # Orient the normal toward the workpiece side (see _segment_workpiece).
    signed = pts @ n + d
    off_mask = np.abs(signed) > thresh * 3
    if int(off_mask.sum()) and float(signed[off_mask].mean()) < 0:
        n, d = -n, -d
        signed = -signed

    table_pts = pts[np.abs(signed) < thresh]
    if len(table_pts) < 10:
        return None   # can't define a footprint → caller falls back to no-crop

    # Orthonormal in-plane basis (u, v) spanning the table surface.
    a = np.array([1.0, 0.0, 0.0])
    if abs(float(n @ a)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = a - (a @ n) * n
    u = u / (np.linalg.norm(u) or 1.0)
    v = np.cross(n, u)

    tu, tv = table_pts @ u, table_pts @ v
    u_lo, u_hi = np.percentile(tu, [1, 99])   # robust table outline
    v_lo, v_hi = np.percentile(tv, [1, 99])
    height_cap = CROP_HEIGHT_FRAC * max(u_hi - u_lo, v_hi - v_lo)

    # Height above which a point belongs to the workpiece, from the table's
    # MEASURED surface noise. `thresh` scales with the whole scene (floor and
    # background included), so 3·thresh can sit centimetres above the table
    # and swallow a thin base plate together with the weld root standing on it.
    res = signed[np.abs(signed) < thresh]
    sigma = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    part_cut = min(3 * thresh, max(PART_CUT_SIGMA * sigma, 0.1 * thresh))

    return {'n': n, 'd': d, 'u': u, 'v': v,
            'u_lo': u_lo, 'u_hi': u_hi, 'v_lo': v_lo, 'v_hi': v_hi,
            'thresh': thresh, 'part_cut': part_cut,
            'height_cap': height_cap, 'scene_diag': diag}


def _voxel_components(band_pts, voxel):
    """
    Voxel-grid connected-components (26-neighbour, union-find). Returns an
    integer component label per point. Numpy-only — no scipy/sklearn.
    """
    keys = np.floor(band_pts / voxel).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    M = len(uniq)
    index = {(int(a), int(b), int(c)): i for i, (a, b, c) in enumerate(uniq)}

    parent = list(range(M))

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    offs = [(dx, dy, dz)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)]
    for i in range(M):
        a, b, c = int(uniq[i][0]), int(uniq[i][1]), int(uniq[i][2])
        for dx, dy, dz in offs:
            j = index.get((a + dx, b + dy, c + dz))
            if j is not None and j > i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    roots = np.array([find(i) for i in range(M)])
    return roots[inv]                                   # component per point


def _cluster_keep_mask(band_pts, signed, table):
    """
    Keep every component that has at least one point resting on the table
    (low signed height); drop free-floating components such as a hovering
    welding torch. Returns a boolean mask over band_pts.
    """
    N = len(band_pts)
    if N < 10:
        return np.ones(N, bool)
    voxel = max(CROP_VOXEL_FRAC * table['scene_diag'], 1e-9)
    comp = _voxel_components(band_pts, voxel)
    anchored = np.unique(comp[signed <= table['thresh'] * 3])
    if len(anchored) == 0:
        return np.ones(N, bool)                         # fallback: keep all
    return np.isin(comp, anchored)


def _tall_cluster_mask(pts, signed, table, min_height_mult=PART_MIN_HEIGHT_MULT):
    """
    Keep only clusters PROMINENT in DEPTH: a real workpiece rises centimetres
    above the table, while rust stains / discolouration only corrupt VGGT's
    depth by a few millimetres. Filtering fake 'workpiece' bumps here keeps
    the seam path anchored to genuine geometric edges instead of appearance
    artefacts. Falls back to keeping everything if no cluster qualifies.
    Returns a boolean mask over pts.
    """
    N = len(pts)
    if N < 10:
        return np.ones(N, bool)
    # Half the crop voxel: at the crop's resolution a rust patch NEXT TO the
    # part bridges into its cluster and inherits "tall" status. Finer voxels
    # keep low noise patches as separate (droppable) components.
    voxel = max(0.5 * CROP_VOXEL_FRAC * table['scene_diag'], 1e-9)
    comp = _voxel_components(pts, voxel)
    min_h = table['thresh'] * min_height_mult
    keep_ids = []
    for c in np.unique(comp):
        sel = signed[comp == c]
        # 95th percentile height: robust "how tall is this cluster really"
        if len(sel) >= 10 and float(np.percentile(sel, 95)) >= min_h:
            keep_ids.append(c)
    if not keep_ids:
        return np.ones(N, bool)                         # never empty the set
    return np.isin(comp, np.array(keep_ids))


def _scene_crop_mask(flat_pts, table):
    """
    Boolean keep-mask over flat_pts for "the table + the object(s) resting on
    it": points inside the (margin-expanded) table footprint, from the table
    surface up to height_cap, with free-floating clusters removed.
    """
    n, d = table['n'], table['d']
    signed = flat_pts @ n + d
    pu, pv = flat_pts @ table['u'], flat_pts @ table['v']
    mu = CROP_MARGIN * (table['u_hi'] - table['u_lo'])
    mv = CROP_MARGIN * (table['v_hi'] - table['v_lo'])
    on_fp = ((pu >= table['u_lo'] - mu) & (pu <= table['u_hi'] + mu) &
             (pv >= table['v_lo'] - mv) & (pv <= table['v_hi'] + mv))

    valid = ~np.all(flat_pts == 0, axis=1)
    band = (on_fp & valid &
            (signed >= -table['thresh']) & (signed < table['height_cap']))

    keep = np.zeros(len(flat_pts), bool)
    idx = np.where(band)[0]
    if len(idx) == 0:
        return keep
    cl = _cluster_keep_mask(flat_pts[idx].astype(np.float64), signed[idx], table)
    keep[idx[cl]] = True
    return keep


def _view_align_matrix(table):
    """
    3×3 world→aligned rotation that maps the table normal to the viewer's "up"
    (+y), so the front-end turntable spins horizontally about the table's true
    vertical instead of the arbitrary VGGT data axis. Row-major (rows = aligned
    x, y, z axes in world coords). Returns a nested list or None if degenerate.
    """
    ey = np.asarray(table['n'], np.float64)
    ex = np.asarray(table['u'], np.float64)
    ez = np.cross(ex, ey)
    nz = np.linalg.norm(ez)
    if nz < 1e-9:
        return None
    ez = ez / nz
    ex = np.cross(ey, ez)                       # re-orthogonalise
    ex = ex / (np.linalg.norm(ex) or 1.0)
    return np.vstack([ex, ey, ez]).tolist()


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

    # Restrict the workpiece to points that sit ABOVE the table AND OVER the
    # table's physical footprint. Without this, a wall / shelf / equipment in
    # the background is also "off the table plane" and gets mistaken for the
    # workpiece, sending the seam onto the wall. We bound candidates by the
    # in-plane extent of the table-plane inliers and cap their height.
    above = signed > thresh * 3
    table_mask = np.abs(signed) < thresh
    table_pts = pts[table_mask]
    if len(table_pts) >= 10:
        # Orthonormal in-plane basis (u, v) spanning the table surface.
        a = np.array([1.0, 0.0, 0.0])
        if abs(float(n @ a)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        u = a - (a @ n) * n
        u = u / (np.linalg.norm(u) or 1.0)
        v = np.cross(n, u)

        tu, tv = table_pts @ u, table_pts @ v
        # 1st/99th pct → robust table outline (ignores a few stray inliers)
        u_lo, u_hi = np.percentile(tu, [1, 99])
        v_lo, v_hi = np.percentile(tv, [1, 99])
        # Small margin so a workpiece touching the table edge isn't clipped.
        mu, mv = 0.02 * (u_hi - u_lo), 0.02 * (v_hi - v_lo)

        pu, pv = pts @ u, pts @ v
        on_table = ((pu >= u_lo - mu) & (pu <= u_hi + mu) &
                    (pv >= v_lo - mv) & (pv <= v_hi + mv))

        # Height cap: workpieces sit low on the table; reject anything taller
        # than half the table's in-plane extent (kills tall background walls
        # that still overlap the footprint in projection).
        height_cap = 0.5 * max(u_hi - u_lo, v_hi - v_lo)
        above = above & on_table & (signed < height_cap)

    workpiece = pts[above]
    return n, d, workpiece, diag


def _reject_far_outliers(pts, k=3.0):
    """
    Drop points far from the bulk of the workpiece using a robust median + MAD
    radius. Detached reconstruction noise (a floating clump off to the side)
    otherwise gets chosen as a seam endpoint. Conservative: only trims points
    well beyond the main mass, so a genuinely elongated seam is preserved.
    """
    if pts is None or len(pts) < 20:
        return pts
    c  = np.median(pts, axis=0)
    d  = np.linalg.norm(pts - c, axis=1)
    md = float(np.median(d))
    mad = float(np.median(np.abs(d - md)))
    if mad < 1e-9:
        return pts
    keep = d <= md + k * 1.4826 * mad
    return pts[keep] if int(keep.sum()) >= 20 else pts


def _rdp_simplify(pts, eps):
    """
    Douglas–Peucker polyline simplification. Removes waypoints that deviate
    less than `eps` from the local straight run. This is what keeps ARC LENGTH
    honest: a noisy-but-straight seam collapses to its chord instead of
    accumulating zigzag length, while a genuine curve keeps its shape.
    """
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    L = float(np.linalg.norm(ab))
    if L < 1e-12:
        d = np.linalg.norm(pts - a, axis=1)
    else:
        u = ab / L
        rel = pts - a
        d = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
    i = int(np.argmax(d))
    if d[i] > eps:
        left  = _rdp_simplify(pts[:i + 1], eps)
        right = _rdp_simplify(pts[i:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([a, b])


def _extract_seam_path(seam_pts, line_dir, p0, scene_diag,
                       n_bins=40, min_pts_per_bin=3):
    """Trace local connectivity instead of sorting a curved joint on one axis."""
    return trace_path(seam_pts, scene_diag)


def _path_arc_length(path):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _contact_seam(workpiece_pts, n, d, thresh, part_extent, scene_diag):
    """
    Trace where a standing member MEETS a base surface (plane n·p+d=0),
    following any curvature of the member.

    Points in a thin band just above the base plane that (a) sit over the
    base surface and (b) continue upward into a rising wall are the root of
    the joint. Dropping them onto the plane gives the weld-root footprint,
    which is traced by local connectivity (open seams) or angle (closed
    loops, e.g. pipe-to-plate). Straight walls give a 2-point path, curved
    walls a dense polyline.

    Returns (path, info) or (None, reason).
    """
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import connected_components
    n = np.asarray(n, np.float64)
    n = n / (np.linalg.norm(n) or 1.0)
    h = workpiece_pts @ n + d
    off = np.abs(h) > thresh * 3
    if int(off.sum()) < 30:
        return None, 'no-off-plane'
    if float(h[off].mean()) < 0:          # orient toward the standing member
        n, d, h = -n, -d, -h
    base = workpiece_pts[np.abs(h) < thresh * 1.5]
    if len(base) < 30:
        return None, 'no-base'

    lo = thresh * 1.5
    band_h = max(3 * thresh, 0.02 * part_extent)
    hi = lo + band_h
    band_m = (h > lo) & (h <= hi)
    upper_m = (h > hi) & (h <= hi + 3 * band_h)
    if band_m.sum() < 15 or upper_m.sum() < 15:
        return None, 'no-rising-wall'
    band, hb = workpiece_pts[band_m], h[band_m]

    # (a) the band must continue upward (a wall), not be a low skirt/warp
    r_up = 2.0 * band_h
    rising = cKDTree(workpiece_pts[upper_m]).query(band, distance_upper_bound=r_up)[0] < np.inf
    # (b) its footprint must land on / next to the base surface
    foot = band - np.outer(hb, n)
    base_foot = base - np.outer(base @ n + d, n)
    r_base = 1.5 * band_h
    on_base = cKDTree(base_foot).query(foot, distance_upper_bound=r_base)[0] < np.inf
    foot = foot[rising & on_base]
    if len(foot) < 15:
        return None, 'no-supported-contact'

    # Several standing members → keep the longest connected footprint.
    tree = cKDTree(foot)
    link = max(3 * band_h, 0.03 * part_extent)
    pairs = tree.query_pairs(link, output_type='ndarray')
    if len(pairs):
        from scipy.sparse import coo_matrix
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(foot), len(foot)))
        _, lab = connected_components(g, directed=False)
        best, best_ext = None, -1.0
        for c in np.unique(lab):
            sel = foot[lab == c]
            if len(sel) < 15:
                continue
            ext = float(np.linalg.norm(np.ptp(sel, axis=0)))
            if ext > best_ext:
                best, best_ext = sel, ext
        if best is None:
            return None, 'no-supported-contact'
        foot = best

    # A well-covered ring (pipe-to-plate) must be traced as a loop first: the
    # open tracer also "succeeds" on it, but keeps only part of the circle.
    path = trace_loop(foot)
    if path is not None:
        return path, f'closed,band={len(foot)}'
    path = trace_path(foot, scene_diag)
    if path is not None and len(path) > 2:
        # Thick plates: drop the wrap-around at the plate's end faces.
        eps = max(2 * thresh, 0.02 * float(np.linalg.norm(np.ptp(foot, axis=0))))
        path = trim_end_hooks(path, eps)
    if path is not None:
        # A seam only exists where BOTH the base face and the standing member
        # are present next to it; stray points can extend the footprint past
        # the part's real ends.
        up = workpiece_pts[upper_m]
        up_foot = up - np.outer(up @ n + d, n)
        # Base support is measured AWAY from the root (2–6 band heights), so
        # stray points that merely lie in the base plane next to the root
        # don't count as base surface.
        path = _clip_to_support(path, [(base_foot, 2.0 * band_h, 6.0 * band_h),
                                       (up_foot, 0.0, 2.0 * band_h)])
    if path is None:
        return None, 'untraceable-contact'
    return path, f'open,band={len(foot)}'


def _resample(path, step):
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    n = max(2, int(np.ceil(arc[-1] / max(step, 1e-12))) + 1)
    s = np.linspace(0, arc[-1], n)
    return s, np.column_stack([np.interp(s, arc, path[:, j]) for j in range(3)])


def _clip_to_support(path, supports, lo_pct=0.2, hi_pct=99.8):
    """Trim an open path to the arc range where every support set has points
    beside it. `supports` is a list of (points, r_min, r_max): only points
    whose distance to the path lies in [r_min, r_max] count. The path is left
    unchanged if a set gives no usable range, and is never cut below 20 % of
    its original length."""
    from scipy.spatial import cKDTree
    path = np.asarray(path, float)
    total = _path_arc_length(path)
    if total <= 0 or len(path) < 2:
        return path
    step = min(r for _, r, _ in [(None, rmax, None) for _, _, rmax in supports]) / 4
    s, dense = _resample(path, step)
    tree = cKDTree(dense)
    lo, hi = 0.0, s[-1]
    for pts, rmin, rmax in supports:
        if pts is None or len(pts) < 10:
            continue
        dist, idx = tree.query(pts, distance_upper_bound=rmax)
        near = np.isfinite(dist) & (dist >= rmin)
        if near.sum() < 10:
            continue
        a, b = np.percentile(s[idx[near]], [lo_pct, hi_pct])
        lo, hi = max(lo, a), min(hi, b)
    if hi - lo < 0.2 * total:
        return path
    keep = (s >= lo) & (s <= hi)
    if len(path) == 2:
        return dense[keep][[0, -1]]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    inner = path[(arc > lo) & (arc < hi)]
    ends = [np.array([np.interp(t, arc, path[:, j]) for j in range(3)]) for t in (lo, hi)]
    return np.vstack([ends[0], inner, ends[1]])


def _line_fit(path):
    """PCA line through a polyline (densified so vertex spacing doesn't bias it).
    Returns (centre, unit direction, 90th-pct perpendicular deviation)."""
    _, dense = _resample(path, _path_arc_length(path) / 200)
    c = dense.mean(axis=0)
    u = np.linalg.svd(dense - c, full_matrices=False)[2][0]
    rel = dense - c
    dev = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
    return c, u, float(np.percentile(dev, 90))


def _max_dev_from_line(path, p0, u):
    rel = path - p0
    return float(np.max(np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)))


def _member_seam(workpiece_pts, ss, n1, d1, mask1, table_n, table_d, thresh, scene_diag):
    """
    Seam between two flat members standing above the base (an upright T, a
    corner of two plates), ignoring the base itself. Returns the
    _detect_seam_geometric tuple, or None when the parts above the base are
    not two genuine plates (a pipe or curved wall also yields two RANSAC
    "planes", as chords, but its surface normals do not line up with them).
    """
    # Base: the dominant plane when it lies along the table, else the table.
    base_n, base_d = n1, d1
    if table_n is not None and table_d is not None:
        tn = np.asarray(table_n, np.float64)
        tnn = float(np.linalg.norm(tn)) or 1.0
        if abs(float(n1 @ tn)) / tnn < 0.9 or mask1.sum() < 0.1 * len(ss):
            base_n, base_d = tn / tnn, float(table_d) / tnn
    h = ss @ base_n + base_d
    if (h > 3 * thresh).sum() < (h < -3 * thresh).sum():
        base_n, base_d, h = -base_n, -base_d, -h
    mem = ss[h > 3 * thresh]
    if len(mem) < 200:
        return None

    na, da, ma = _ransac_plane(mem, n_iters=300, thresh=thresh, seed=1)
    if na is None or len(mem) - ma.sum() < 100:
        return None
    nb, db, mb = _ransac_plane(mem[~ma], n_iters=300, thresh=thresh, seed=2)
    if (nb is None or mb.sum() < max(100, 0.15 * ma.sum())
            or abs(float(na @ nb)) > np.cos(np.radians(MEMBER_MIN_ANGLE_DEG))):
        return None

    # Flat members: most local surface normals agree with one of the planes.
    from scipy.spatial import cKDTree
    sub = mem[np.random.default_rng(3).choice(len(mem), min(8000, len(mem)),
                                               replace=False)]
    _, nbr = cKDTree(sub).query(sub, k=min(30, len(sub)))
    q = sub[nbr] - sub[nbr].mean(axis=1, keepdims=True)
    local_n = np.linalg.svd(q, full_matrices=False)[2][:, 2]
    near = np.cos(np.radians(15))
    flat = float(((np.abs(local_n @ na) > near) | (np.abs(local_n @ nb) > near)).mean())
    if flat < MEMBER_FLAT_FRAC:
        return None

    line_dir = np.cross(na, nb)
    line_dir /= np.linalg.norm(line_dir)
    p0, *_ = np.linalg.lstsq(np.vstack([na, nb]), -np.array([da, db]), rcond=None)
    above = workpiece_pts @ base_n + base_d > 3 * thresh
    zone = (above & (np.abs(workpiece_pts @ na + da) < 2 * thresh)
            & (np.abs(workpiece_pts @ nb + db) < 2 * thresh))
    seam_pts = workpiece_pts[zone]
    if len(seam_pts) < 4:
        return None
    projected = p0 + np.outer((seam_pts - p0) @ line_dir, line_dir)
    path = _extract_seam_path(projected, line_dir, p0, scene_diag)
    if path is None:
        return None
    return (path[0], path[-1],
            f't-joint:members,pa={int(ma.sum())},pb={int(mb.sum())},'
            f'flat={flat:.2f},seam={len(seam_pts)},wp={len(path)}',
            path)


def _detect_seam_geometric(workpiece_pts, scene_diag, table_n=None, table_d=None, joint_mode="auto"):
    """
    Auto-pick the seam and return (start_3d, end_3d, method, path) where
    `path` is an ordered (N,3) waypoint array (N=2 for straight seams,
    N>2 when the seam curves).

      fillet   : standing plate ∩ KNOWN table plane → base weld line (preferred
                 when the table plane is supplied; most robust for T/fillet welds)
      t-joint  : two strong workpiece planes → seam = plane1 ∩ plane2
      groove   : one plane + offset ridge    → seam = PCA on offset points

    Returns (None, None, 'failed', None) if no strategy produces a seam.
    """
    if len(workpiece_pts) < 50:
        return None, None, 'failed:too-few-workpiece-pts', None

    # Drop detached floating noise before any fitting, so endpoints can't snap
    # to a stray clump off the workpiece.
    workpiece_pts = _reject_far_outliers(workpiece_pts)
    if len(workpiece_pts) < 50:
        return None, None, 'failed:too-few-workpiece-pts', None

    # Background extent must not set a centimetre-wide seam tolerance on a
    # small part. Bound the fit tolerance by the robust workpiece extent.
    part_extent = float(np.linalg.norm(np.percentile(workpiece_pts, 98, axis=0)
                                      - np.percentile(workpiece_pts, 2, axis=0)))
    thresh = max(1e-9, min(0.005 * scene_diag, 0.015 * part_extent))

    ss_n = min(30000, len(workpiece_pts))
    if ss_n < len(workpiece_pts):
        idx = np.random.default_rng(1).choice(len(workpiece_pts), ss_n, replace=False)
        ss = workpiece_pts[idx]
    else:
        ss = workpiece_pts

    n1, d1, mask1 = _ransac_plane(ss, n_iters=300, thresh=thresh, seed=1)
    if n1 is None:
        return None, None, 'failed:no-plane', None
    plane1_count = int(mask1.sum())

    n2 = d2 = None
    plane2_count = 0
    rem = ss[~mask1]
    if len(rem) >= 100:
        n2, d2, mask2 = _ransac_plane(rem, n_iters=300, thresh=thresh, seed=2)
        plane2_count = int(mask2.sum()) if mask2 is not None else 0

    is_tjoint = (n2 is not None
                 and plane2_count > max(100, 0.15 * plane1_count)
                 and abs(float(n1 @ n2)) < 0.95)  # planes not parallel

    if is_tjoint:
        line_dir = np.cross(n1, n2)
        ln = np.linalg.norm(line_dir)

    # ── Inter-part seam first: two flat members meeting above the base ──
    # A part standing on a plate (or on the table) has two candidate welds:
    # between its own members, and its footprint on the base. The member
    # seam wins; the footprint trace below is the fallback (pipes, single
    # walls, curved plates).
    if joint_mode != "base":
        member = _member_seam(workpiece_pts, ss, n1, d1, mask1, table_n, table_d,
                              thresh, scene_diag)
        if member is not None:
            return member

    # ── Contact trace: follow where a standing member meets a base face ──
    # Works for straight AND curved members (C/S-shaped plates, pipes on a
    # plate). A plane∩plane line can only ever be straight, so this decides
    # whether the joint is really curved before the t-joint line is trusted.
    contact = None
    if joint_mode != "base":
        bases = [(n1, d1)]
        if n2 is not None and plane2_count > max(100, 0.05 * plane1_count):
            bases.append((n2, d2))
        # Prefer the dominant plane as the base (normally the base plate):
        # using the standing wall as "base" makes the plate look like a wall
        # rising along the wall's infinite plane, past the wall's real ends.
        for bn, bd in bases:
            cpath, cinfo = _contact_seam(workpiece_pts, bn, bd, thresh,
                                         part_extent, scene_diag)
            if cpath is not None:
                contact = (cpath, cinfo)
                break

    # Decide straight vs curved from the traced footprint ITSELF. Comparing
    # it to a plane∩plane line is unreliable: two noisy faces of the same
    # plate can pass the "not parallel" test and give a meaningless line.
    contact_straight = None
    if contact is not None and joint_mode != "base":
        cpath, cinfo = contact
        closed = len(cpath) > 3 and np.allclose(cpath[0], cpath[-1])
        if not closed:
            seam_len = _path_arc_length(cpath)
            c, u, dev90 = _line_fit(cpath)
            if len(cpath) == 2 or dev90 <= max(2.5 * thresh, 0.02 * seam_len):
                ts = np.sort((cpath[[0, -1]] - c) @ u)
                contact_straight = (c, u, ts, dev90)

    if is_tjoint and joint_mode != "base":
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
                agrees = False
                if contact is not None and contact_straight is not None:
                    # Use the precise plane∩plane line only where the traced
                    # root actually lies on it (and runs along it).
                    c, u, ts, _ = contact_straight
                    _, dense = _resample(contact[0], _path_arc_length(contact[0]) / 100)
                    rel = dense - p0
                    off = np.linalg.norm(rel - np.outer(rel @ line_dir, line_dir), axis=1)
                    agrees = (abs(float(u @ line_dir)) > np.cos(np.radians(10))
                              and np.percentile(off, 90) < 2.0 * thresh)
                if contact is None and len(seam_pts) >= 4:
                    projected = p0 + np.outer((seam_pts - p0) @ line_dir, line_dir)
                    path = _extract_seam_path(projected, line_dir, p0, scene_diag)
                elif agrees:
                    # Clip the infinite line to where the wall actually stands.
                    tl = np.sort((contact[0][[0, -1]] - p0) @ line_dir)
                    path = p0 + np.outer(tl, line_dir)
                else:
                    path = None
                if path is not None:
                    return (path[0], path[-1],
                            f't-joint:plane-intersection,'
                            f'p1={plane1_count},p2={plane2_count},'
                            f'seam={len(seam_pts)},wp={len(path)}',
                            path)

    # ── Fillet strategy: trace where the standing workpiece MEETS the table ──
    # Only reached when there is no valid t-joint ridge. Take the workpiece's
    # BASE BAND (points just above the table), drop it onto the table plane,
    # and extract the ordered path through it. Unlike a plane∩plane
    # intersection this follows a CURVED plate's bend, because the per-bin
    # medians track wherever the base band actually runs. Skipped for flat
    # lap/groove parts (dominant plane ≈ parallel to table) → groove path.
    if joint_mode == "base" and table_n is not None and table_d is not None:
        tn = np.asarray(table_n, np.float64)
        tnn = float(np.linalg.norm(tn))
        if tnn > 1e-9:
            tn = tn / tnn
        if abs(float(n1 @ tn)) < 0.85:               # workpiece stands upright
            signed_t = workpiece_pts @ tn + float(table_d)
            band = workpiece_pts[np.abs(signed_t) < thresh * 8]   # low, near the table
            if len(band) >= 10:
                base = band - np.outer(band @ tn + float(table_d), tn)
                c0 = base.mean(axis=0)
                _, _, vt = np.linalg.svd(base - c0, full_matrices=False)
                path = _extract_seam_path(base, vt[0], c0, scene_diag)
                if path is not None:
                    return (path[0], path[-1],
                            f'fillet:base-band,band={len(band)},wp={len(path)}',
                            path)

    if joint_mode == "base":
        return None, None, "failed:no-supported-base-contact", None

    if contact is not None:
        cpath, cinfo = contact
        if contact_straight is not None:
            c, u, ts, dev90 = contact_straight
            line = c + np.outer(ts, u)
            return (line[0], line[-1],
                    f'straight-joint:contact-trace,{cinfo},p1={plane1_count},'
                    f'dev90={dev90:.4f},wp=2', line)
        return (cpath[0], cpath[-1],
                f'curved-joint:contact-trace,{cinfo},p1={plane1_count},wp={len(cpath)}',
                cpath)

    # Groove path: ridge points = workpiece points significantly off plane1
    signed = workpiece_pts @ n1 + d1
    ridge_mask = np.abs(signed) > thresh * 1.5
    ridge_pts = workpiece_pts[ridge_mask]
    if len(ridge_pts) < 10:
        return None, None, 'failed:no-ridge', None
    s, e, axis = _fit_line_endpoints(ridge_pts, trim_pct=0.05)
    if s is None:
        return None, None, 'failed:groove-pca', None
    path = _extract_seam_path(ridge_pts.astype(np.float64), axis,
                              ridge_pts.mean(axis=0), scene_diag)
    if path is None:
        return None, None, 'failed:unsupported-ridge', None
    return (path[0], path[-1],
            f'groove:ridge-pca,plane={plane1_count},'
            f'ridge={len(ridge_pts)},wp={len(path)}',
            path)


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
        metric_depth_v = align_depth(metric_depth_v, (H, W))
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
        m = None
        if str(device).startswith('cuda'):
            # Build the 1B-param model and map its weights straight onto the
            # GPU. The default path first allocates ~5 GB of CPU RAM and then
            # maps the 5 GB weight file, which can exceed the Windows commit
            # limit ("paging file is too small", os error 1455).
            try:
                with torch.device(device):
                    m = VGGT.from_pretrained('facebook/VGGT-1B',
                                             map_location=str(device))
            except (TypeError, AttributeError, RuntimeError):
                m = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if m is None:
            m = VGGT.from_pretrained('facebook/VGGT-1B')
        m = m.to(device)
        m.eval()
        _VGGT_MODEL, _VGGT_LOAD_DEVICE = m, device
        return _VGGT_MODEL, True


def _get_metric3d_cached(device):
    global _M3D_MODEL, _M3D_LOAD_DEVICE
    with _models_lock:
        if _M3D_LOAD_DEVICE == device:
            # Cached value may legitimately be None (load failed once).
            return _M3D_MODEL, False
        _M3D_MODEL       = _load_metric3d(device, M3D_VARIANT)
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

        # 3 — Metric3D is loaded lazily, AFTER hole detection, and only when it
        # can actually help (no holes, or holes present but not yet calibrated).
        # This skips the model entirely on the common fast path (fixture in
        # frame + saved calibration).
        m3d = None
        m3d_skipped_reason = None
        _emit(q, 'progress', {'step': 3, 'total': 7,
                              'msg': 'Metric3D: deferred (loads only if needed)'})

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
            vggt_depth = depth_views(vggt_depth, wp.shape[0])

        V = wp.shape[0]
        CONF_THRESH = 1.0

        # 4a — Fit the fixture table once and crop the scene to it. The same
        # table frame drives the PLY crop, workpiece/seam detection, and the
        # viewer's "up" alignment (view_align). Falls back to full scene if no
        # dominant plane is found.
        flat_wp  = wp.reshape(-1, 3)
        flat_wpc = wpc.reshape(-1) if wpc is not None else None
        table = (_fit_table(flat_wp, flat_wpc, CONF_THRESH)
                 if CROP_TO_TABLE else None)
        scene_keep = _scene_crop_mask(flat_wp, table) if table is not None else None
        if scene_keep is not None and int(scene_keep.sum()) < 500:
            scene_keep = None     # crop too aggressive → fall back to full scene
        if scene_keep is not None:
            _emit(q, 'progress', {'step': 4, 'total': 7,
                                  'msg': f'Cropped scene to table footprint: '
                                         f'{int(scene_keep.sum()):,} pts kept'})

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
        if scene_keep is not None:                # crop to table + on-table object
            keep &= scene_keep

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

        if table is not None and scene_keep is not None:
            # Reuse the cropped table frame: the workpiece is the on-table
            # object(s) above the surface — the floating torch is already gone
            # from scene_keep, so the seam can't latch onto background.
            table_n, table_d, scene_diag = table['n'], table['d'], table['scene_diag']
            valid = ~np.all(flat_wp == 0, axis=1)
            if flat_wpc is not None:
                valid &= flat_wpc > CONF_THRESH
            signed_all = flat_wp @ table_n + table_d
            wp_mask = valid & scene_keep & (signed_all > table['part_cut'])
            workpiece_pts = flat_wp[wp_mask].astype(np.float64)
            # Depth-prominence filter: only clusters that genuinely RISE off
            # the table are workpiece. Rust/brightness artefacts in the VGGT
            # depth (a few mm of bump) get dropped here, so the seam can only
            # attach to real geometric edges.
            tall = _tall_cluster_mask(workpiece_pts,
                                      signed_all[wp_mask].astype(np.float64),
                                      table)
            n_dropped = int(len(workpiece_pts) - tall.sum())
            workpiece_pts = workpiece_pts[tall]
            if n_dropped:
                _emit(q, 'progress', {'step': 6, 'total': 7,
                                      'msg': f'Dropped {n_dropped:,} low-prominence '
                                             f'pts (rust/noise bumps)'})
        else:
            # No table fit (or crop disabled) → legacy footprint segmentation.
            table_n, table_d, workpiece_pts, scene_diag = _segment_workpiece(
                flat_wp, conf=flat_wpc, conf_thresh=CONF_THRESH)

        _emit(q, 'progress', {'step': 6, 'total': 7,
                              'msg': f'Workpiece: {len(workpiece_pts):,} pts '
                                     f'(scene diag {scene_diag:.3f})'})

        detect_method = 'none'
        weld_s_w = weld_e_w = None
        weld_path_w = None                 # ordered (N,3) waypoints, N>=2
        if len(workpiece_pts) >= 50:
            _emit(q, 'progress', {'step': 6, 'total': 7,
                                  'msg': 'Geometric seam (contact trace / plane intersection / ridge)…'})
            weld_s_w, weld_e_w, detect_method, weld_path_w = _detect_seam_geometric(
                workpiece_pts, scene_diag, table_n=table_n, table_d=table_d,
                joint_mode=JOBS[job_id].get("joint_mode", "auto"))
            # Keep the detector's exact input so detection can be re-run and
            # debugged later without repeating the reconstruction.
            try:
                np.savez_compressed(
                    os.path.join(workdir, 'geometry.npz'),
                    part=np.asarray(workpiece_pts, np.float32),
                    diag=float(scene_diag),
                    table_n=(np.asarray(table_n, float) if table_n is not None
                             else np.full(3, np.nan)),
                    table_d=(float(table_d) if table_d is not None else np.nan))
            except Exception:
                pass

        # Decide whether to run Metric3D this job. In 'metric3d+holes' mode it
        # supplies the headline scale, so it runs every time. In 'holes' mode it
        # runs only when there are no holes (fast path). Loaded lazily.
        need_metric3d = (MEASUREMENT_SCALE == 'metric3d+holes') or (
            hole_scale_factor is None)
        m3d_skipped_holes = (not need_metric3d) and (hole_scale_factor is not None)
        if need_metric3d and vggt_depth is not None:
            m3d, m3d_loaded = _get_metric3d_cached(device)
            _emit(q, 'progress', {'step': 6, 'total': 7,
                                  'msg': ('Loading Metric3D ('
                                          + M3D_VARIANT + ')…' if m3d_loaded
                                          else 'Metric3D ready (cached)')})
        else:
            m3d_skipped_reason = ('holes used (Metric3D off for speed)'
                                  if m3d_skipped_holes else 'no depth output')
            _emit(q, 'progress', {'step': 6, 'total': 7,
                                  'msg': f'Metric3D skipped — {m3d_skipped_reason}'})

        # Per-view Metric3D scale samples (independent of detection method)
        # Focal for Metric3D: explicit intrinsics > user FOV (hfov_deg) > guess.
        manual_intr = JOBS[job_id].get('intrinsics')
        hfov_deg    = JOBS[job_id].get('hfov_deg')
        focal_norm  = (1.0 / (2.0 * np.tan(np.radians(float(hfov_deg)) / 2.0))
                       if hfov_deg else None)
        # Restrict the VGGT↔Metric3D comparison to the cropped table+part region
        # (where both reconstruct reliably) instead of the whole image.
        scene_keep_vhw = (scene_keep.reshape(V, H, W)
                          if scene_keep is not None else None)
        if m3d is not None and vggt_depth is not None:
            for i, img_path in enumerate(image_paths):
                _emit(q, 'progress', {'step': 6, 'total': 7,
                                      'msg': f'Metric3D scale {i+1}/{V}…'})
                m_depth, _ = _run_metric3d(m3d, img_path, device,
                                           intrinsics=manual_intr,
                                           focal_norm=focal_norm)
                if (scene_keep_vhw is not None
                        and scene_keep_vhw[i].shape == vggt_depth[i].shape):
                    mask_i = (scene_keep_vhw[i] > 0).astype(np.uint8) * 255
                else:
                    mask_i = np.ones(vggt_depth[i].shape, np.uint8) * 255
                s, n = _vggt_to_metric_scale(vggt_depth[i], m_depth, mask_i,
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

        detected = weld_path_w is not None and len(weld_path_w) >= 2
        if not detected:
            weld_path_w = np.empty((0, 3), dtype=float)
            detect_method = 'needs-selection: no supported narrow seam found'

        # Seam length = ARC length along the waypoint path. For a straight
        # (2-point) path this equals the start–end distance; for a curvy seam
        # it is the true weld length (the chord would under-measure it).
        seam_len_vggt = _path_arc_length(weld_path_w)

        # Metric3D raw scale (sample-weighted median across views).
        metric3d_scale_raw = None
        if scale_samples:
            arr = np.array(scale_samples)            # (K, 2): scale, n_used
            order = np.argsort(arr[:, 0])
            sorted_s = arr[order, 0]; sorted_n = arr[order, 1]
            cum = np.cumsum(sorted_n)
            metric3d_scale_raw = float(sorted_s[np.searchsorted(cum, cum[-1] / 2.0)])

        # Anchor Metric3D to the holes. When the holes are present they're ground
        # truth, so Metric3D's scale snaps to them (and we remember the
        # correction for off-fixture use). With no holes, Metric3D — corrected by
        # any saved calibration — carries the scale on its own.
        metric3d_scale     = None
        metric3d_calibrated = False
        metric3d_correction = None
        if metric3d_scale_raw is not None and metric3d_scale_raw > 1e-12:
            if hole_scale_factor is not None:
                corr = hole_scale_factor / metric3d_scale_raw
                if np.isfinite(corr) and corr > 0:
                    _save_m3d_calib(corr)
                    metric3d_correction = corr
                metric3d_scale = hole_scale_factor           # anchored to holes
                metric3d_calibrated = True
            elif _M3D_CORRECTION is not None:
                metric3d_correction = _M3D_CORRECTION
                metric3d_scale = metric3d_scale_raw * _M3D_CORRECTION
                metric3d_calibrated = True
            else:
                metric3d_scale = metric3d_scale_raw          # raw (no anchor yet)
        elif m3d_skipped_holes:
            metric3d_scale = hole_scale_factor
            metric3d_correction = _M3D_CORRECTION
            metric3d_calibrated = _M3D_CORRECTION is not None

        # Headline scale + source. In 'metric3d+holes' mode the headline is the
        # (holes-anchored) Metric3D scale; we fall back to holes only if Metric3D
        # is unavailable, then to nothing.
        if MEASUREMENT_SCALE == 'metric3d+holes' and metric3d_scale is not None:
            scale_factor = metric3d_scale
            scale_source = ('metric3d+holes' if hole_scale_factor is not None
                            else 'metric3d')
        elif hole_scale_factor is not None:
            scale_factor = hole_scale_factor
            scale_source = 'holes'
        elif metric3d_scale is not None:
            scale_factor = metric3d_scale
            scale_source = 'metric3d'
        else:
            scale_factor = None
            scale_source = None

        seam_len_m = (seam_len_vggt * scale_factor) if scale_factor else None

        # Metric3D as an INDEPENDENT measurement (shown in its own UI panel).
        seam_mm_metric3d = (seam_len_vggt * metric3d_scale * 1000.0
                            if metric3d_scale and detected else None)

        # Table coordinate frame (origin + axes) from detected hole lattice.
        table_frame = None
        if hole_scale_factor and table_n is not None and len(holes_world_all) >= 4:
            table_frame = _fit_table_frame(
                np.array(holes_world_all), table_n, hole_scale_factor)

        # Keep weld_start/end in raw VGGT units so they live in the same
        # coordinate frame as the PLY the viewer renders. Metric and table
        # frame versions go in separate fields.
        s_out   = weld_s_w.tolist() if detected else None
        e_out   = weld_e_w.tolist() if detected else None
        s_out_m = (weld_s_w * scale_factor).tolist() if scale_factor and detected else None
        e_out_m = (weld_e_w * scale_factor).tolist() if scale_factor and detected else None

        # Viewer "up" hint: 3×3 rotation that stands the table level so the
        # front-end turntable spins horizontally about the table's true
        # vertical. Pure render hint — does not alter weld_start/end or the PLY.
        view_align = _view_align_matrix(table) if table is not None else None

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
            'detection_status': 'candidate' if detected else 'needs_selection',
            'point_count':    n_pts,
            'per_frame':      per_frame,
            # Cloud-frame (raw VGGT units) — matches the PLY served to the viewer.
            'weld_start':     s_out,
            'weld_end':       e_out,
            # Ordered waypoint path (N>=2; N=2 means straight). Same frame as
            # weld_start/end; weld_path_m is the metres version (null unscaled).
            'weld_path':      weld_path_w.tolist(),
            'weld_path_m':    ((weld_path_w * scale_factor).tolist()
                               if scale_factor else None),
            'waypoint_count': int(len(weld_path_w)),
            'is_curved':      bool(len(weld_path_w) > 2),
            # Metric (world axes) versions — null if no scale factor.
            'weld_start_m':   s_out_m,
            'weld_end_m':     e_out_m,
            'seam_length_m':  (seam_len_m if detected else None),
            'seam_length_mm': (seam_len_m * 1000 if detected and seam_len_m is not None else None),
            'seam_length_vggt_units': seam_len_vggt,
            # Independent Metric3D measurement (its own UI panel). null if the
            # model didn't load or produced no usable scale samples this run.
            'seam_length_mm_metric3d': seam_mm_metric3d,
            'metric3d_available':      (m3d is not None or m3d_skipped_reason is not None),
            'metric3d_skipped':        m3d_skipped_reason,    # null if it ran
            'metric3d_scale_samples':  len(scale_samples),
            'metric3d_calibrated':     metric3d_calibrated,
            'metric3d_correction':     metric3d_correction,
            'metric3d_calib_source':   ('holes' if (metric3d_calibrated and len(scale_samples) and hole_scale_factor is not None)
                                        else 'saved' if metric3d_calibrated
                                        else 'uncalibrated'),
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
            'view_align':        view_align,          # 3×3 render hint, null if no table
        }
        avg_len = seam_len_m if seam_len_m is not None else seam_len_vggt
        with open(os.path.join(workdir, 'weld_result.json'), 'w') as f:
            json.dump(result, f, indent=2)

        JOBS[job_id].update({'result': result, 'status': 'done'})
        unit_msg = (f'Done — seam {avg_len*1000:.1f} mm' if scale_factor
                    else f'Done — seam {seam_len_vggt:.4f} VGGT units '
                         f'(no Metric3D scale)')
        if not detected:
            unit_msg = 'Reconstruction ready — select the joint with Draw Path'
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

@app.route('/examples/<name>.json')
def path_example(name):
    if name not in ('straight', 'curved'):
        return jsonify({'error': 'Unknown example'}), 404
    return send_from_directory(os.path.join(app.root_path, 'examples'), name + '.json')

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

    # Optional horizontal field-of-view (degrees) — simpler than full
    # intrinsics; sets the Metric3D focal when intrinsics aren't supplied.
    hfov_deg = None
    hfov_raw = request.form.get('hfov_deg')
    if hfov_raw:
        try:
            v = float(hfov_raw)
            if 10.0 < v < 170.0:
                hfov_deg = v
        except Exception:
            pass

    JOBS[job_id] = {'status': 'running', 'queue': queue.Queue(),
                    'result': None, 'workdir': workdir,
                    'intrinsics': intrinsics, 'hfov_deg': hfov_deg,
                    'joint_mode': request.form.get('joint_mode', 'auto')}
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
