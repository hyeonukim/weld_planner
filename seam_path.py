"""Topology-preserving tracing of narrow, unordered 3-D seam candidates."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def trace_path(points, scene_diag=None):
    """Return an open seam polyline, or None for insufficient/disconnected support.

    Local graph distances preserve bends that PCA sorting collapses. Inputs must
    already represent a narrow seam zone, not an entire workpiece surface.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    p = np.unique(p[np.isfinite(p).all(axis=1)], axis=0)
    if len(p) < 10:
        return None
    extent = np.linalg.norm(np.ptp(p, axis=0))
    if extent <= 1e-12:
        return None
    voxel = extent / 180
    keys = np.floor((p - p.min(axis=0)) / voxel).astype(int)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inv)
    p = np.column_stack([np.bincount(inv, weights=p[:, j]) / counts for j in range(3)])
    tree = cKDTree(p)
    dist, idx = tree.query(p, k=min(9, len(p)))
    radius = min(max(3 * np.median(dist[:, 1]), 2.5 * voxel), extent * .04)
    rows = np.repeat(np.arange(len(p)), idx.shape[1] - 1)
    cols = idx[:, 1:].ravel()
    weights = dist[:, 1:].ravel()
    keep = (weights > 0) & (weights <= radius)
    graph = coo_matrix((weights[keep], (rows[keep], cols[keep])), shape=(len(p), len(p))).tocsr()
    graph = graph.maximum(graph.T)
    _, labels = connected_components(graph)
    component = np.argmax(np.bincount(labels))
    chosen = np.flatnonzero(labels == component)
    if len(chosen) < max(10, .6 * len(p)):
        return None
    p = p[chosen]
    graph = graph[chosen][:, chosen]
    a = int(np.argmax(dijkstra(graph, indices=0)))
    distances, previous = dijkstra(graph, indices=a, return_predecessors=True)
    b = int(np.argmax(distances))
    order = [b]
    while order[-1] != a:
        order.append(int(previous[order[-1]]))
    path = p[order[::-1]]
    if len(path) < 3:
        return None
    # Reject broad surfaces: most candidates must lie in a narrow tube.
    support = cKDTree(path).query(p)[0]
    if np.percentile(support, 80) > max(4 * voxel, extent * .06):
        return None
    # Local smoothing preserves endpoint positions and U/S topology.
    for _ in range(3):
        path[1:-1] = (path[:-2] + 2 * path[1:-1] + path[2:]) / 4
    chord = path[-1] - path[0]
    length = np.linalg.norm(chord)
    if length > 0:
        deviation = np.linalg.norm(np.cross(path - path[0], chord / length), axis=1)
        if deviation.max() < max(2 * voxel, .008 * length):
            return path[[0, -1]]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    samples = np.linspace(0, arc[-1], min(200, max(16, len(path))))
    return np.column_stack([np.interp(samples, arc, path[:, j]) for j in range(3)])


def depth_views(depth, views):
    """Normalize VGGT [1,V,H,W,1] without accidentally dropping a view."""
    d = np.asarray(depth)
    if d.ndim == 5 and d.shape[0] == 1:
        d = d[0]
    if d.ndim == 4 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 3 or d.shape[0] != views:
        raise ValueError(f'Unexpected VGGT depth shape: {d.shape}')
    return d


def align_depth(depth, target_shape):
    """Match VGGT default width resize, centre crop and batch centre padding."""
    import cv2
    h, w = depth.shape
    rh = round(h * (518 / w) / 14) * 14
    resized = cv2.resize(depth, (518, rh), interpolation=cv2.INTER_LINEAR)
    if rh > 518:
        top = (rh - 518) // 2
        resized = resized[top:top + 518]
    th, tw = target_shape
    dh, dw = th - resized.shape[0], tw - resized.shape[1]
    if dh < 0 or dw < 0:
        raise ValueError('Depth target does not match VGGT crop preprocessing')
    return np.pad(resized, ((dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2)), constant_values=np.nan)


def trace_loop(points, min_coverage_deg=330, max_gap_deg=35):
    """Order a CLOSED seam (e.g. pipe-to-plate) by angle about its centre.

    trace_path() is an open-path tracer: on a ring it keeps only half the
    loop and its support check then rejects it. This handles roughly
    star-shaped loops (circles, ellipses, rounded rectangles). Returns a
    closed polyline (last point == first) or None if the points are not a
    narrow, well-covered loop.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 30:
        return None
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    uv = (p - c) @ vt[:2].T
    r = np.linalg.norm(uv, axis=1)
    r_med = float(np.median(r))
    if r_med <= 1e-12:
        return None
    # A loop is an annulus: nothing near the centre, narrow radial spread.
    if np.percentile(r, 5) < .5 * r_med:
        return None
    ang = np.degrees(np.arctan2(uv[:, 1], uv[:, 0])) % 360
    s = np.sort(ang)
    gaps = np.diff(np.r_[s, s[0] + 360])
    if gaps.max() > max_gap_deg or 360 - gaps.max() < min_coverage_deg:
        return None
    nb = int(np.clip(len(p) // 6, 36, 180))
    bins = np.minimum((ang / 360 * nb).astype(int), nb - 1)
    pts = []
    for b in range(nb):
        sel = p[bins == b]
        if len(sel):
            pts.append(np.median(sel, axis=0))
    path = np.asarray(pts)
    if len(path) < 12:
        return None
    # Narrowness check rejects a filled disc / broad surface.
    if np.percentile(cKDTree(path).query(p)[0], 80) > .15 * r_med:
        return None
    for _ in range(3):   # circular smoothing
        path = (np.roll(path, 1, 0) + 2 * path + np.roll(path, -1, 0)) / 4
    return np.vstack([path, path[:1]])


def trim_end_hooks(path, eps, window_frac=.12, min_turn_deg=135, end_frac=.4):
    """Cut U-turn hooks off both ends of an open seam polyline.

    A thick standing plate shows two faces joined by an end face, so a traced
    footprint can run along one face, wrap round the plate end and start back
    along the other face. That hook is not part of the seam and makes a
    straight joint look curved. A hook turns >= `min_turn_deg` within a short
    arc (`window_frac` of the length, at least 6*eps); real seam bends are far
    gentler, so only such turns inside the last `end_frac` of each end are
    cut, at the point of furthest progress before the turn.
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 8:
        return path
    cos_lim = np.cos(np.radians(min_turn_deg))
    for _ in range(2):                       # tail end, then (reversed) head end
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        total = arc[-1]
        if total <= 0:
            return path
        w = max(window_frac * total, 6 * eps)
        q = w / 4                                   # tangent chord length

        def tangent(i):
            j = min(np.searchsorted(arc, arc[i] + q), len(path) - 1)
            t = path[j] - path[i]
            n = np.linalg.norm(t)
            return t / n if n > 0 else None

        start = np.searchsorted(arc, (1 - end_frac) * total)
        for i in range(start, len(path) - 1):
            j = np.searchsorted(arc, arc[i] + w - q)
            if j >= len(path) - 1:
                break
            ti, tj = tangent(i), tangent(j)
            if ti is None or tj is None or ti @ tj > cos_lim:
                continue
            k_end = min(np.searchsorted(arc, arc[i] + w), len(path) - 1)
            prog = (path[i:k_end + 1] - path[i]) @ ti
            cut = i + int(np.argmax(prog))
            path = path[:cut + 1]
            break
        path = path[::-1]
    return path
