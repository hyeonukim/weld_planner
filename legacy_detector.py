"""Seam detector as it was BEFORE curved-joint support (kept for A/B comparison).

Uses the same helpers as server.py; only the strategy selection differs.
"""
import numpy as np
from server import (_reject_far_outliers, _ransac_plane, _fit_line_endpoints,
                    _extract_seam_path)


def detect_seam_legacy(workpiece_pts, scene_diag, table_n=None, table_d=None, joint_mode="auto"):
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
                if len(seam_pts) < 4:
                    return None, None, 'failed:unsupported-intersection', None
                # Two planar faces define a straight seam. Remove transverse
                # depth noise without changing its supported longitudinal extent.
                projected = p0 + np.outer((seam_pts - p0) @ line_dir, line_dir)
                path = _extract_seam_path(projected, line_dir, p0, scene_diag)
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
