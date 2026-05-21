import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


# ─────────────────────────────────────────────
# Metric3D helper: load model + predict depth
# ─────────────────────────────────────────────

def load_metric3d(device):
    """
    Load Metric3D v2 (ViT-Large backbone) from torch.hub.
    Falls back to ViT-Small if the large model is unavailable.
    Requires:  pip install timm
    """
    print(">>> Loading Metric3D v2 (ViT-L)...")
    try:
        model = torch.hub.load(
            "YvanYin/Metric3D",
            "metric3d_vit_large",
            pretrain=True,
        )
    except Exception:
        print("    ViT-L unavailable, falling back to ViT-Small")
        model = torch.hub.load(
            "YvanYin/Metric3D",
            "metric3d_vit_small",
            pretrain=True,
        )
    model.to(device).eval()
    return model


def metric3d_depth(model, image_path, device,
                   intrinsics=None,
                   input_size=(616, 1064)):
    """
    Run Metric3D on a single image and return a metric depth map (H×W, float32, metres).

    Parameters
    ----------
    model        : Metric3D model returned by load_metric3d()
    image_path   : path to an RGB image (any size)
    device       : torch device
    intrinsics   : [fx, fy, cx, cy] in pixels – pass your real camera intrinsics when
                   known; if None a reasonable guess is derived from image size
    input_size   : (H, W) the model expects (must match backbone; default for ViT-L/S)
    """
    # --- Read & normalise ---
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]

    # --- Derive intrinsics if not provided (assumes 70-deg HFOV) ---
    if intrinsics is None:
        fx = fy = w0 / (2 * np.tan(np.radians(70) / 2))
        cx, cy = w0 / 2.0, h0 / 2.0
    else:
        fx, fy, cx, cy = intrinsics

    # --- Resize to model input size (keep intrinsics consistent) ---
    scale_h = input_size[0] / h0
    scale_w = input_size[1] / w0
    fx_s = fx * scale_w
    fy_s = fy * scale_h
    cx_s = cx * scale_w
    cy_s = cy * scale_h
    rgb_r = cv2.resize(rgb, (input_size[1], input_size[0]),
                       interpolation=cv2.INTER_LINEAR)

    # --- Build tensor ---
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std  = np.array([58.395,  57.12,  57.375], dtype=np.float32)
    img_t = ((rgb_r.astype(np.float32) - mean) / std)
    img_t = torch.from_numpy(img_t.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # --- Inference ---
    with torch.no_grad():
        pred = model.inference(
            {"input": img_t},
            cam_model=None,
            pad_info=None,
            normalize_scale=None,
            cam_intrinsic={"fx": fx_s, "fy": fy_s, "cx": cx_s, "cy": cy_s},
        )
    # Metric3D returns depth in model-input space → resize back to original
    depth_s = pred["prediction"].squeeze().cpu().numpy()   # (H_in, W_in)
    depth   = cv2.resize(depth_s, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return depth.astype(np.float32), (fx, fy, cx, cy)


# ─────────────────────────────────────────────
# Weld-seam endpoint detection
# ─────────────────────────────────────────────

def backproject(depth_map, fx, fy, cx, cy):
    """Convert a metric depth map to a 3-D point cloud (H×W×3, metres)."""
    h, w = depth_map.shape
    us = np.arange(w, dtype=np.float32)
    vs = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    X = (uu - cx) * depth_map / fx
    Y = (vv - cy) * depth_map / fy
    Z = depth_map
    return np.stack([X, Y, Z], axis=-1)  # (H, W, 3)


def detect_weld_endpoints(rgb_image_path, depth_map, fx, fy, cx, cy,
                           visualise=True, save_prefix="weld"):
    """
    Detect the 3-D start and end coordinates of a weld seam.

    Strategy
    --------
    1. Convert the RGB image to LAB colour space and threshold on the 'a'
       channel, which captures the orange/amber glow typical of a weld bead.
    2. Apply Canny edge detection on the depth map to find geometric
       discontinuities (the raised bead ridge).
    3. Combine both masks and skeletonise to a thin seam line.
    4. Find the two extremal points along the principal axis of that line
       (via PCA) – these are the start and end.
    5. Back-project those pixel coordinates to 3-D using the metric depth.

    Returns
    -------
    start_3d, end_3d : np.ndarray of shape (3,), units = metres
    """
    bgr = cv2.imread(rgb_image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # ── Step 1: colour mask (weld bead tends to be brighter / warmer) ──
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # High L (bright) + slightly elevated a (warm) → likely weld bead
    brightness_mask = (lab[:, :, 0] > 160).astype(np.uint8) * 255

    # ── Step 2: depth-edge mask ──
    d_norm = cv2.normalize(depth_map, None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8)
    edges  = cv2.Canny(d_norm, 30, 90)

    # ── Step 3: combine & clean ──
    combined = cv2.bitwise_or(brightness_mask, edges)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Skeletonise (thin to 1-px wide line)
    skel   = np.zeros_like(combined)
    temp   = combined.copy()
    while True:
        eroded   = cv2.erode(temp, kernel)
        opened   = cv2.dilate(eroded, kernel)
        diff     = cv2.subtract(temp, opened)
        skel     = cv2.bitwise_or(skel, diff)
        temp     = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    # ── Step 4: PCA to find principal axis → endpoints ──
    pts = np.column_stack(np.where(skel > 0))   # (N, 2): rows = [row, col]
    if len(pts) < 10:
        print("  WARNING: very few seam pixels found – check colour/depth thresholds")
        # Fall back: use full point cloud extremes along Z axis
        pts3d_all = backproject(depth_map, fx, fy, cx, cy).reshape(-1, 3)
        valid     = pts3d_all[:, 2] > 0
        pts3d_v   = pts3d_all[valid]
        idx_min   = np.argmin(pts3d_v[:, 2])
        idx_max   = np.argmax(pts3d_v[:, 2])
        return pts3d_v[idx_min], pts3d_v[idx_max]

    coords = pts[:, ::-1].astype(np.float32)  # (N, 2): [col, row] = [x, y]
    mean   = coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords - mean, full_matrices=False)
    axis   = vt[0]  # principal direction (unit vector in image plane)

    # Project each point onto the axis and find extremes
    proj       = (coords - mean) @ axis
    idx_start  = np.argmin(proj)
    idx_end    = np.argmax(proj)
    px_start   = coords[idx_start].astype(int)   # [col, row]
    px_end     = coords[idx_end].astype(int)

    # ── Step 5: back-project to 3-D ──
    pts3d = backproject(depth_map, fx, fy, cx, cy)  # (H, W, 3)

    def safe_px(uv, pts3d, h, w):
        u = int(np.clip(uv[0], 0, w - 1))
        v = int(np.clip(uv[1], 0, h - 1))
        return pts3d[v, u]

    start_3d = safe_px(px_start, pts3d, h, w)
    end_3d   = safe_px(px_end,   pts3d, h, w)

    # ── Optional visualisation ──
    if visualise:
        vis = rgb.copy()
        # Draw skeleton in green
        vis[skel > 0] = [0, 220, 0]
        # Mark start (blue) and end (red)
        cv2.circle(vis, tuple(px_start), 10, (0, 0, 255),   -1)
        cv2.circle(vis, tuple(px_end),   10, (255, 0, 0),   -1)
        cv2.line(vis,   tuple(px_start), tuple(px_end), (255, 255, 0), 2)

        plt.figure(figsize=(10, 5))
        plt.imshow(vis)
        plt.title(f"Weld seam | START (blue) → END (red)\n"
                  f"3D start: ({start_3d[0]:.3f}, {start_3d[1]:.3f}, {start_3d[2]:.3f}) m\n"
                  f"3D end:   ({end_3d[0]:.3f},   {end_3d[1]:.3f},   {end_3d[2]:.3f}) m")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_endpoints.png", dpi=150)
        plt.show()
        print(f"  Saved endpoint visualisation → {save_prefix}_endpoints.png")

    return start_3d, end_3d


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # ── 1. Device ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 \
                else torch.float16
    else:
        dtype = torch.float32
    print(f">>> Running on: {device}")

    # ── 2. Load VGGT ──
    print(">>> Loading VGGT-1B...")
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    vggt_model.eval()

    # ── 3. Load Metric3D ──
    metric3d_model = load_metric3d(device)

    # ── 4. Images ──
    # Replace with your actual filenames.
    image_names = [
        "buttjoint_000115.png",
        "buttjoint_000215.png",
        "buttjoint_000255.png",
    ]
    for img in image_names:
        if not os.path.exists(img):
            print(f"ERROR: Could not find {img}")
            return

    # ── 5. VGGT inference (multi-view 3-D reconstruction) ──
    print(">>> Running VGGT multi-view reconstruction...")
    images_t = load_and_preprocess_images(image_names).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype):
            predictions = vggt_model(images_t)

    depths_vggt = predictions["depth"].cpu().numpy()       # relative depth
    pts3d_vggt  = predictions["world_points"].cpu().numpy()
    colors      = images_t.permute(0, 2, 3, 1).cpu().numpy()

    # ── 6. Save VGGT depth maps ──
    print(">>> Saving VGGT depth maps...")
    for i, name in enumerate(image_names):
        d_map = np.squeeze(depths_vggt[i])
        if d_map.ndim == 3 and d_map.shape[0] == 3:
            d_map = np.transpose(d_map, (1, 2, 0))[:, :, 0]
        plt.figure(figsize=(10, 5))
        plt.imshow(d_map, cmap="magma")
        plt.title(f"VGGT Depth: {name}")
        plt.axis("off")
        plt.savefig(f"depth_result_{i}.png")
        plt.close()

    # ── 7. Export VGGT coloured point cloud ──
    print(">>> Exporting VGGT point cloud → buttjoint_reconstruction.ply")
    flat_pts    = pts3d_vggt.reshape(-1, 3)
    flat_colors = (colors.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    with open("buttjoint_reconstruction.ply", "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(flat_pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(flat_pts, flat_colors):
            if np.all(p == 0):
                continue
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

    # ── 8. Metric3D per-frame: metric depth + weld endpoint detection ──
    print("\n>>> Running Metric3D + weld endpoint detection per image...")
    all_endpoints = []

    for i, img_path in enumerate(image_names):
        print(f"\n  Image {i+1}/{len(image_names)}: {img_path}")

        # Get metric depth (absolute, in metres)
        metric_depth, (fx, fy, cx, cy) = metric3d_depth(
            metric3d_model, img_path, device,
            # Uncomment and set your real camera intrinsics for best accuracy:
            # intrinsics=[fx, fy, cx, cy],
        )
        np.save(f"metric_depth_{i}.npy", metric_depth)
        print(f"    Metric depth range: {metric_depth.min():.3f} – "
              f"{metric_depth.max():.3f} m")

        # Detect weld start / end in 3-D
        start_3d, end_3d = detect_weld_endpoints(
            rgb_image_path=img_path,
            depth_map=metric_depth,
            fx=fx, fy=fy, cx=cx, cy=cy,
            visualise=True,
            save_prefix=f"weld_frame{i}",
        )

        seam_length = np.linalg.norm(end_3d - start_3d)
        all_endpoints.append({
            "image":      img_path,
            "start_3d":   start_3d,
            "end_3d":     end_3d,
            "seam_length_m": seam_length,
        })

        print(f"    START  (x,y,z): ({start_3d[0]:.4f}, "
              f"{start_3d[1]:.4f}, {start_3d[2]:.4f}) m")
        print(f"    END    (x,y,z): ({end_3d[0]:.4f}, "
              f"{end_3d[1]:.4f}, {end_3d[2]:.4f}) m")
        print(f"    Seam length:     {seam_length*1000:.1f} mm")

    # ── 9. Summary across all frames ──
    print("\n" + "="*60)
    print("WELD ENDPOINT SUMMARY")
    print("="*60)
    for ep in all_endpoints:
        print(f"\n  {ep['image']}")
        s, e = ep["start_3d"], ep["end_3d"]
        print(f"    Start : ({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}) m")
        print(f"    End   : ({e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f}) m")
        print(f"    Length: {ep['seam_length_m']*1000:.1f} mm")

    # Average endpoint across frames (simple fusion)
    avg_start = np.mean([ep["start_3d"] for ep in all_endpoints], axis=0)
    avg_end   = np.mean([ep["end_3d"]   for ep in all_endpoints], axis=0)
    avg_len   = np.linalg.norm(avg_end - avg_start) * 1000
    print(f"\n  ── Averaged across {len(image_names)} frames ──")
    print(f"    Start : ({avg_start[0]:.4f}, {avg_start[1]:.4f}, {avg_start[2]:.4f}) m")
    print(f"    End   : ({avg_end[0]:.4f},   {avg_end[1]:.4f},   {avg_end[2]:.4f}) m")
    print(f"    Length: {avg_len:.1f} mm")
    print("\nDONE.")

    # ── 9. Send weld path to robot via ROS 2 ────────────────────────────────
    # Set SEND_TO_ROBOT = True (and ARMED = True for a live weld) once you
    # have verified the endpoint coordinates look correct in the visualisation.
    SEND_TO_ROBOT = False   # ← flip to True when ready
    ARMED         = False   # ← flip to True for a real weld (False = dry-run)
    ROBOT_IP      = "192.168.2.151"  # ← your Fanuc controller IP

    if SEND_TO_ROBOT:
        try:
            from weld_ros_bridge import send_weld_to_robot
            success = send_weld_to_robot(
                start_3d_cam        = avg_start,   # metres, camera frame
                end_3d_cam          = avg_end,     # metres, camera frame
                robot_ip            = ROBOT_IP,
                # camera_T_robot    = np.load("hand_eye.npy"),  # ← your calib
                procedure           = 1,
                schedule            = 1,
                speed_mm_s          = 5.0,
                armed               = ARMED,
                approach_offset_mm  = 20.0,
            )
            print(f"\n[bridge] Robot execution {'SUCCESS' if success else 'FAILED'}")
        except ImportError:
            print("\n[bridge] weld_ros_bridge not found — "
                  "place it in the same folder or on PYTHONPATH.")
        except Exception as exc:
            print(f"\n[bridge] Error sending to robot: {exc}")
    else:
        print("\n[bridge] SEND_TO_ROBOT=False — robot not commanded.")
        print("         Set SEND_TO_ROBOT=True in test.py when ready.")


if __name__ == "__main__":
    main()
