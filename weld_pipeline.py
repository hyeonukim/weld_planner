import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def verify_images(image_names):
    # check all input images exist before doing anything
    print("\n[STEP 1] Verifying images...")
    for img in image_names:
        if not os.path.exists(img):
            print(f"  ERROR: Could not find '{img}'")
            return False
        print(f"  Found: {img}")
    return True


def run_vggt_reconstruction(image_names, device, dtype):
    # run VGGT on the images to get a 3D point cloud and depth maps
    print("\n[STEP 2] Running VGGT 3D reconstruction...")
    print("  Loading model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    images = load_and_preprocess_images(image_names).to(device)
    print(f"  Image tensor shape: {images.shape}")

    print("  Running inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype):
            predictions = model(images)

    depths = predictions['depth'].cpu().numpy()
    pts3d  = predictions['world_points'].cpu().numpy()
    confs  = predictions.get('world_points_conf', None)
    if confs is not None:
        confs = confs.cpu().numpy()

    # permute from (B, C, H, W) to (B, H, W, C) so colors line up with points
    colors = images.permute(0, 2, 3, 1).cpu().numpy()

    print(f"  Point cloud shape: {pts3d.shape}")
    return depths, pts3d, colors, confs


def export_depth_maps(depths, image_names):
    # save a depth map image for each view so we can visually check the reconstruction
    print("\n[STEP 3a] Saving depth maps...")
    for i in range(len(depths)):
        d_map = np.squeeze(depths[i])
        if d_map.ndim == 3 and d_map.shape[0] == 3:
            d_map = np.transpose(d_map, (1, 2, 0))[:, :, 0]

        plt.figure(figsize=(10, 5))
        plt.imshow(d_map, cmap='magma')
        plt.title(f"Depth Map: {image_names[i]}")
        plt.colorbar(label='Depth')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"depth_result_{i}.png", dpi=150)
        plt.close()
        print(f"  Saved: depth_result_{i}.png")


def export_point_cloud(pts3d, colors, confs=None, conf_threshold=1.0, downsample=1):
    # flatten the 3D output into a list of points and filter out bad ones
    print("\n[STEP 3b] Processing point cloud...")

    flat_pts    = pts3d.reshape(-1, 3)
    flat_colors = (colors.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)

    # remove points at the origin since those are invalid
    valid_mask = ~np.all(flat_pts == 0, axis=1)

    # remove low confidence points if VGGT gave us a confidence map
    if confs is not None:
        flat_confs  = confs.reshape(-1)
        valid_mask &= (flat_confs > conf_threshold)
        print(f"  Points after confidence filter: {valid_mask.sum()}")

    flat_pts    = flat_pts[valid_mask]
    flat_colors = flat_colors[valid_mask]

    if downsample > 1:
        flat_pts    = flat_pts[::downsample]
        flat_colors = flat_colors[::downsample]

    print(f"  Total points: {len(flat_pts)}")
    return flat_pts, flat_colors


def calibrate_scale(flat_pts, hole_spacing_m=0.05):
    # figure out the real-world scale by having the user click two adjacent hole centers
    # the holes are 5cm apart so we can use that to convert VGGT units to meters
    print("\n[STEP 4] Scale calibration using fixture holes...")
    print(f"  Real hole spacing: {hole_spacing_m*100:.0f}cm = {hole_spacing_m}m")
    print("\n  A plot will open. Click the centers of TWO adjacent holes, then close the window.")
    print("  Adjacent means directly next to each other horizontally or vertically.")

    # show a top-down XY view so the holes are easy to click
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(flat_pts[:, 0], flat_pts[:, 1], c='gray', s=0.3, alpha=0.5)
    ax.set_title("Click 2 adjacent hole centers for scale calibration, then close")
    ax.set_xlabel("X (VGGT units)")
    ax.set_ylabel("Y (VGGT units)")
    ax.set_aspect('equal')

    clicks = []

    def on_click(event):
        if event.inaxes and event.button == 1:
            clicks.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
            fig.canvas.draw()
            if len(clicks) == 2:
                ax.plot([clicks[0][0], clicks[1][0]],
                        [clicks[0][1], clicks[1][1]], 'r-', linewidth=2)
                fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show(block=True)

    if len(clicks) < 2:
        print("  WARNING: Need 2 clicks for calibration. Defaulting scale to 1.0 (VGGT units).")
        return 1.0

    # measure the VGGT distance between the two clicked hole centers
    p1 = np.array(clicks[0])
    p2 = np.array(clicks[1])
    vggt_dist = np.linalg.norm(p2 - p1)

    # scale factor converts VGGT units to meters
    scale = hole_spacing_m / vggt_dist

    print(f"  VGGT distance between clicks: {vggt_dist:.4f} units")
    print(f"  Scale factor: {scale:.4f} (multiply VGGT coords by this to get meters)")
    return scale


def write_ply(flat_pts, flat_colors, scale=1.0):
    # write the point cloud to a PLY file, applying the scale factor to get real-world coords
    print("\n[STEP 5] Writing PLY file...")

    scaled_pts = flat_pts * scale

    with open("reconstruction.ply", "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(scaled_pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(scaled_pts, flat_colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

    print("  Saved: reconstruction.ply")
    return scaled_pts


def print_scene_stats(scaled_pts, scale):
    # print basic measurements of the reconstructed scene in real-world units
    print("\n[STEP 6] Scene measurements (real-world)...")
    x_range = scaled_pts[:, 0].max() - scaled_pts[:, 0].min()
    y_range = scaled_pts[:, 1].max() - scaled_pts[:, 1].min()
    z_range = scaled_pts[:, 2].max() - scaled_pts[:, 2].min()
    print(f"  Scale factor applied: {scale:.4f}")
    print(f"  Scene width  (X): {x_range*100:.1f} cm")
    print(f"  Scene depth  (Y): {y_range*100:.1f} cm")
    print(f"  Scene height (Z): {z_range*100:.1f} cm")
    print(f"  Total points: {len(scaled_pts)}")

    # save scale info to JSON for reference, casting to float so JSON can serialize
    with open("scene_info.json", "w") as f:
        json.dump({
            "scale_factor":     float(scale),
            "scene_width_cm":   float(round(x_range * 100, 2)),
            "scene_depth_cm":   float(round(y_range * 100, 2)),
            "scene_height_cm":  float(round(z_range * 100, 2)),
            "coordinate_frame": "real_world_meters"
        }, f, indent=2)
    print("  Saved: scene_info.json")


def main():
    image_names  = ["groove1.jpg", "groove2.jpg", "groove3.jpg"]
    CONF_THRESH  = 1.0   # drop points below this VGGT confidence score
    DOWNSAMPLE   = 1     # set to 4 to speed things up and reduce file size
    HOLE_SPACING = 0.05  # distance between adjacent fixture holes in meters (5cm)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) \
             if device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    if not verify_images(image_names):
        return

    depths, pts3d, colors, confs = run_vggt_reconstruction(image_names, device, dtype)

    export_depth_maps(depths, image_names)
    flat_pts, flat_colors = export_point_cloud(pts3d, colors, confs,
                                               conf_threshold=CONF_THRESH,
                                               downsample=DOWNSAMPLE)

    scale = calibrate_scale(flat_pts, hole_spacing_m=HOLE_SPACING)

    scaled_pts = write_ply(flat_pts, flat_colors, scale=scale)

    print_scene_stats(scaled_pts, scale)

    print("\nPipeline complete.")
    print("  Open reconstruction.ply in your 3D viewer.")
    print("  Coords are now in real-world meters.")


if __name__ == "__main__":
    main()