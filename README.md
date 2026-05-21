# Weld Path Planner

3D reconstruction pipeline for welding with a browser-based visualizer.  
Uses **VGGT** (multi-view 3D reconstruction) + **Metric3D** (absolute depth) to detect weld seam start/end coordinates from photos.

---

## Folder structure

```
weld_planner/
├── index.html          ← browser visualizer (served by server.py)
├── server.py           ← Flask backend that runs the pipeline
├── test.py             ← standalone CLI pipeline (no server needed)
├── weld_ros_bridge.py  ← ROS 2 bridge to Fanuc CRx robot (optional)
├── requirements.txt
├── vggt/               ← VGGT model source
│   ├── models/
│   ├── layers/
│   ├── heads/
│   └── utils/
├── groove1.jpg         ← sample weld images
├── groove2.jpg
├── groove3.jpg
└── reconstruction.ply  ← sample output point cloud
```

---

## Quick start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Start the web server

```bash
python server.py
```

Then open **http://localhost:5000** in your browser.

### 3 — Use the visualizer

1. Drop **2–6 weld photos** onto the upload zone (or use the sample `groove*.jpg` files)
2. Click **Generate 3D Reconstruction**
3. Watch the 7-step progress bar while VGGT + Metric3D run
4. The point cloud loads automatically with the detected start/end markers shown
5. Use **Manual Pick** to click any two points yourself, or **Auto Detect** to re-run the seam detection with tunable thresholds
6. Export coordinates as `weld_path.json`

You can also drop a `.ply` file directly to skip reconstruction and just use the visualizer.

---

## Standalone CLI (no browser)

```bash
python test.py
```

Edit the `image_names` list at the top of `main()` to point to your images.  
Outputs: `buttjoint_reconstruction.ply`, `metric_depth_*.npy`, `weld_frame*_endpoints.png`

---

## Robot integration (ROS 2)

After reconstruction, set `SEND_TO_ROBOT = True` in `test.py` and start:

```bash
# Terminal 1 — ROS 2 robot listener
ros2 run robot_gui_listener robot_gui_listener

# Terminal 2 — pipeline
python test.py
```

See `weld_ros_bridge.py` for the full topic/service map.

---

## Notes

- First run downloads VGGT-1B (~4 GB) from HuggingFace and Metric3D from torch.hub — internet required
- GPU strongly recommended (CPU will work but is slow)
- For production accuracy, provide real camera intrinsics `[fx, fy, cx, cy]` to `metric3d_depth()`
