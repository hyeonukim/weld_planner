# Weld Path Planner

Reconstruct a workpiece from photos, identify a candidate welding joint, and inspect its start point, end point, and full 3D path in a browser.

**VGGT provides the shared 3D geometry. Metric3D estimates its real-world scale.** Seam detection uses geometric fitting and local connectivity; neither model is a dedicated weld-joint segmentation model.

The main workflow is visual inspection and path export. **ROS2 is not required.**

## Features

- Reconstruct a point cloud from 2–6 photos of the same stationary workpiece.
- Detect straight intersections between workpiece faces, including upright T-joints.
- Trace curved joints where a standing member meets a base plate (C/S/U-shaped plates, pipe-to-plate loops) by following the member's contact footprint.
- Display an orange path with start/end markers and waypoint-based playback.
- Select endpoints manually or draw an ordered path directly on the point cloud.
- Export the reconstruction, detection result, and selected path.
- Load synthetic straight and curved examples without running the reconstruction models.

## Setup

Run commands from the project directory. The pipeline was validated locally with Python 3.10 and an NVIDIA GPU.

### 1. Create a Python environment

On Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The commands below use `python`. If the virtual environment is not activated, replace it with `.venv\Scripts\python.exe`.

The legacy `install.bat` installs CPU PyTorch and omits SciPy. Use `requirements.txt` for the maintained pipeline.

### 2. Check GPU availability

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

GPU acceleration requires a CUDA-enabled PyTorch installation compatible with the machine. The server selects CUDA when available and otherwise uses the CPU. CPU reconstruction can be slow; it was not the configuration used for the photo validation reported below.

The first reconstruction needs internet access to obtain VGGT weights from Hugging Face and Metric3D through Torch Hub. Models are large, so allow time and sufficient RAM, GPU memory, and disk space. Model instances are cached within the running server process.

Metric3D is optional: if it cannot load, reconstruction can continue. Fixture holes may still provide scale; otherwise coordinates remain in VGGT units.

### 3. Start the viewer

```powershell
python server.py
```

Open [http://localhost:5000](http://localhost:5000).

The existing `start.bat` also launches `server.py` using the Python executable on PATH. Stop the server with `Ctrl+C`.

## Use the visualizer

### Try the path display immediately

Click **Straight demo** or **Curved demo**, then **Play Weld**. These synthetic examples demonstrate graph-traced paths and playback without downloading models. They do not measure reconstruction accuracy on real photos.

### Reconstruct from photos

1. Upload 2–6 JPG or PNG photos showing the same stationary part from overlapping viewpoints.
2. Select **Joint location** before starting:
   - **Between workpiece surfaces (any orientation)** is the default for joints between the parts.
   - **Workpiece to fixture/base — explicit selection** enables the base-contact strategy. Choose it only when that contact is the intended joint.
3. Optionally enter camera intrinsics or horizontal field of view for the Metric3D scale estimate.
4. Click **Generate 3D Reconstruction**.
5. Inspect the candidate path on the reconstructed cloud. Use **Focus Joint** to zoom onto it and **Fit** to return to the scene.
6. Use **Play Weld** to follow the complete path, then export the result or adjust the selection.

The detector labels its output as a **candidate**. If it cannot find a supported seam, the cloud still loads and the viewer asks for manual selection. It does not substitute scene depth extremes or a line through warm-colored points.

### Adjust or select a path

- **Set Start / Set End:** choose two cloud points for a straight path.
- **Draw Path:** click successive points along the joint in traversal order, then click **Finish**. Use **Undo** to remove the last point.
- **Clear:** remove the current selection.
- **Auto Detect → Restore Detected Path:** restore the reconstruction's original candidate after manual edits. This does not rerun inference.

Manual selections follow the points you choose; they are not an additional automatic seam detector. For an imported PLY without a reconstruction result, use manual selection.

### Viewer controls

| Control | Action |
| --- | --- |
| Left drag | Rotate the view |
| Right drag | Pan |
| Mouse wheel | Zoom |
| Focus Joint | Frame the selected path |
| Fit / Reset | Return to the default scene view |
| Axes | Toggle coordinate axes |
| Point Size / Opacity | Adjust cloud visibility |
| Play Weld | Animate travel along the path by arc length |

## Run photos from the command line

`validate_photos.py` runs the same maintained pipeline as the browser backend:

```powershell
python validate_photos.py tu1.jpg tu2.jpg tu3.jpg --output validation_output
```

For an explicitly intended fixture/base contact:

```powershell
python validate_photos.py image1.jpg image2.jpg image3.jpg --joint-mode base --output validation_output
```

The output directory contains `reconstruction.ply` and `weld_result.json`. Reusing the directory replaces those outputs. The CLI currently uses the default camera-scale settings; enter explicit intrinsics through the browser workflow.

### Reopen a saved reconstruction

```powershell
python preview_saved.py validation_output
```

Open [http://127.0.0.1:5001/?job=saved](http://127.0.0.1:5001/?job=saved).

To use another port:

```powershell
python preview_saved.py validation_output --port 5002
```

Then open `http://127.0.0.1:5002/?job=saved`.

The directory must contain both `reconstruction.ply` and `weld_result.json`. This preview does not rerun the models.

## Exports and coordinate frames

| File | Contents |
| --- | --- |
| `reconstruction.ply` | Filtered point cloud in raw VGGT world coordinates |
| `weld_result.json` | Original automatic candidate, scale metadata, and detection method |
| `weld_path.json` | Current viewer selection, including manually edited waypoints |

Use the result card's **JSON** download for the backend result. Use **Export weld_path.json** for the path currently selected in the viewer. Manual edits do not rewrite the backend result.

Important backend fields:

| Field | Meaning |
| --- | --- |
| `detection_status` | `candidate` or `needs_selection` |
| `weld_start`, `weld_end` | Endpoints in the same raw world frame as the PLY; null when detection fails |
| `weld_path` | Ordered 3D waypoints; empty when detection fails |
| `weld_path_m` | Waypoints scaled to metres, when scale is available |
| `seam_length_m`, `seam_length_mm` | Metric arc length; null without metric scale or a detected path |
| `seam_length_vggt_units` | Unscaled path length |
| `scale_factor`, `scale_source` | Conversion from VGGT units to metres and its source |
| `detection_method` | Geometric strategy and supporting counts |

The viewer flips Y for display and reverses that flip for coordinate export. A table alignment is a viewing aid; it does not change the raw cloud coordinates. Coordinates are not calibrated robot coordinates.

Browser jobs are held in memory, with output files in temporary directories. An existing job can be reopened at `/?job=<job_id>` while its server process is alive. Download both reconstruction files to retain a reusable result after restarting the server.

## How detection works

1. **Reconstruct:** VGGT produces per-view world points, confidence values, and depth maps in a shared frame.
2. **Filter:** the backend fits the fixture plane and keeps points more than 4× the table's own surface noise above it (`PART_CUT_SIGMA`), so a thin base plate and the weld root on it survive the crop.
3. **Find candidates**, in order of priority:
   1. **Seam between the part's own plates** (`t-joint:members`). Ignoring the base, two planes are fitted to what stands above it. They are accepted only if they meet at 30° or more and at least 40% of local surface normals line up with one of them (`MEMBER_MIN_ANGLE_DEG`, `MEMBER_FLAT_FRAC`). This rejects pipes and curved walls, whose plane fits are only chords. The seam is the plane∩plane line, clipped to the points lying on both planes.
   2. **Footprint on the base** (`curved-joint` / `straight-joint:contact-trace`). Points in a thin band just above the base that sit on the base and continue upward into a wall are dropped onto the base plane. That footprint is the weld root and follows any curvature. When it is straight, the more precise plane∩plane line is used, clipped to the wall's real extent.
   3. Narrow ridge candidates, or explicitly selected base contacts, remain as fallbacks.
4. **Order the path:** closed footprints are traced as loops first, then open ones by graph connectivity (see the table below). Support checks reject broad surfaces and substantially disconnected candidates. Local smoothing reduces small fluctuations.
5. **Estimate scale:** Metric3D depths are aligned to VGGT's view dimensions and image cropping before depth ratios are compared. Detected fixture holes can anchor the scale using the configured 50 mm spacing.
6. **Preview:** the browser renders the full path, endpoints, and arc-length playback.

Metric3D changes scale, not the seam's geometric shape. Improving its scale estimate cannot recover a joint that VGGT did not reconstruct.

### Straight, curved, and closed seams

No curve model is fitted: the path follows the measured points, so the same steps handle every shape.

| Seam shape | Example | Tracer | How it is ordered | Output |
| --- | --- | --- | --- | --- |
| Straight between plates | Upright T, corner | `_member_seam` | Plane∩plane line, clipped to points on both planes | 2 waypoints |
| Straight on the base | Plate standing on a plate | `trace_path`, then line fit | Footprint within 2.5× tolerance (or 2% of length) of a line → precise plane∩plane line | 2 waypoints |
| Open curve | C/S/U-shaped wall, 200° arc | `trace_path` (`seam_path.py`) | Voxel-thin the footprint, link near neighbours into a graph, take the shortest path between its two farthest-apart points, smooth | Up to 200 waypoints, resampled by arc length |
| Closed loop | Pipe on a plate | `trace_loop` (`seam_path.py`), tried first | Needs ≥ 330° angular coverage around the centre with no gap > 35°; one median point per angle slice, then circular smoothing | Closed polyline (last point = first) |

Straight vs curved is decided from the traced path itself: if it never strays more than 0.8% of its length (or 2.5× the fit tolerance) from a straight line it is reported as straight; otherwise it is kept as a curve.

## Changes addressing poor detection

- Removed the assumption that valid joints must run parallel to the fixture table.
- Made fixture/base contact an explicit selection instead of an automatic substitute for the intended joint.
- Bounded plane-fit tolerance by workpiece size so distant background geometry does not dominate it.
- Allowed unequal-sized workpiece faces to support a T-joint.
- Replaced single-axis curved-path ordering with local graph tracing.
- Fixed VGGT depth dimension handling and Metric3D crop alignment.
- Removed fabricated endpoint fallbacks and the browser's color-only straight-line replacement.
- Made the displayed path more visible and added a joint-focused view.

### Curved-seam and seam-priority changes

| Change | Where | Problem it fixes | Effect |
| --- | --- | --- | --- |
| Table cut from measured surface noise | `_fit_table` (`part_cut`, `PART_CUT_SIGMA`), `run_pipeline` | The old `3 × thresh` cut grew with the background and sat 2–3 cm above the table, deleting a 12 mm base plate and the weld root on it | Base plates survive the crop, so seams on them can be traced |
| Closed-loop tracer runs first | `_contact_seam` | The open tracer accepted a pipe's ring footprint but kept only 70–90% of it | Pipe-to-plate seams come out as full closed loops |
| Seam between the part's own plates wins | `_member_seam`, `MEMBER_MIN_ANGLE_DEG`, `MEMBER_FLAT_FRAC` | With the base kept, an upright T's footprint on the base out-ranked the seam between its plates | Upright T-joints give the vertical inter-plate seam again; pipes and curved walls fall through to the footprint trace |
| Three regression tests | `test_seam_path.py` | — | Unevenly seen pipe → full loop; thin plate survives the crop; upright T on a plate → member seam |
| Gazebo test scenes | `sim/` | No curved-part photos to test with | Rendered photos of round parts with known ground-truth seams |

Results before and after these changes:

| Input | Before | After |
| --- | --- | --- |
| Sim: pipe on plate (r 57 mm) | No seam | Closed loop, 181 waypoints |
| Sim: arc wall (200°) | No seam | Open curve, 106 waypoints |
| Sim: small pipe (r 36.5 mm) | Wrong straight line | Closed loop, 180 waypoints |
| Real `tu1–3` (upright T) | Straight inter-plate seam | Same seam, now reaching the table |
| Real `t-joint_1–3` | Straight seam | Same line (plane∩plane) |

## Validation

Run the regression suite:

```powershell
python -m unittest test_seam_path -v
```

Nineteen tests cover noisy straight, U-shaped, and 3D S-shaped candidates; endpoint and arc-length error; broad surfaces; disconnected fragments; VGGT view dimensions; crop/padding alignment; an upright T-joint, alone and standing on a plate; C- and S-shaped walls on a plate; pipe-on-plate closed loops, including one seen unevenly; thin base plates surviving the table crop; thick plates with end faces; a straight T-joint clipped to the wall's ends; and a lone flat plate that must not yield a seam.

The local `tu1.jpg`–`tu3.jpg` photo run produces a straight vertical inter-plate candidate (`t-joint:members`). Its length is about **151 mm** with that run's scale, against 135.7 mm before the table-cut change, because the seam now reaches the table. These are pipeline results, not comparisons against a measured reference length.

### Simulated round parts

`sim/` renders test scenes in Gazebo Harmonic (run from WSL) with known ground-truth seams. Because hole-based scale is unreliable on the textured sim table, `sim/analyze_lengths.py` scales each reconstruction from the base plate's known 300 mm side:

| Scene | Detected (plate-scaled) | Ground truth | Error |
| --- | --- | --- | --- |
| Pipe on plate | 356.2 mm, r 56.4 mm | 358.1 mm, r 57.0 mm | −0.5% |
| Small pipe | 226.1 mm, r 35.9 mm | 229.3 mm, r 36.5 mm | −1.4% |
| Arc wall | 334.0 mm | 363.0 mm outer face / ≈ 335 mm inner face | −8.0% / ≈ 0% |
| S-shaped wall | 294.1 mm, 91 waypoints | 307.3 mm (centre line) | −4.3% |

The arc wall trace follows the wall's inner fillet; both faces have one.

`sim/residuals.py` plots how far each waypoint sits from the true seam, in the plate's plane. Paths are placed using the base plate only, so position error is not fitted away. Wall residuals are measured from the plate's centre line, with its faces at ±4 mm:

| Scene | RMS | Max | Gap at seam ends |
| --- | --- | --- | --- |
| Pipe on plate | 0.6 mm | 1.2 mm | ≤ 1.7 mm |
| Small pipe | 1.0 mm | 1.8 mm | ≤ 1.3 mm |
| Arc wall | 4.7 mm | 6.9 mm | ≤ 2.5 mm |
| S-shaped wall | 4.7 mm | 7.2 mm | ≤ 0.3 mm |

On thin walls the trace can sit up to about 3 mm beyond a plate face, and on the S wall it crosses from one face to the other and back. That crossing, not missing seam ends, is why the S path is short.

| Script | Purpose |
| --- | --- |
| `sim/make_scene.py` | Build the scenes, textures, robot, cameras, and `ground_truth.json` (run in WSL) |
| `sim/render.sh` | Render four camera photos per scene, headless |
| `sim/gui.sh` | Open a scene in the Gazebo window: `wsl bash sim/gui.sh arc_wall` |
| `sim/redetect.py` | Re-run seam detection on a saved `geometry.npz` |
| `sim/analyze_lengths.py` | Compare detected seam lengths with ground truth |
| `sim/residuals.py` | Write `sim/build/residuals.png` and `residuals.csv`: waypoint distance from the true seam |
| `sim/seam_compare.py` | Write `sim/build/seam_compare.png`: true and detected seams on the sim photos and in plan view |
| `sim/residual_report.py` | Write `sim/build/residual_report.pdf`: three-page report on residuals, lengths, and the seam comparison |
| `sim/photos_pdf.py` | Write `sim/build/photos_used.pdf`: input photos with each detected seam |

Real curved-joint photo accuracy still needs validation.

## Limitations and troubleshooting

| Symptom | Explanation / next step |
| --- | --- |
| No supported seam | Inspect the cloud and use Draw Path. The visible joint may be missing, occluded, or too broad/noisy for the geometric detector. |
| Wrong joint selected | Check Joint location and inspect the candidate. Geometry alone does not establish which contact the user intends to weld. |
| Length looks wrong | Check camera intrinsics, field of view, and fixture spacing. The hole detector assumes 50 mm centre spacing. |
| Coordinates show `u` | No metric scale is available; these are VGGT units. |
| First run stalls or fails loading weights | Check model-host connectivity and available memory/disk space. Inspect the server terminal for the underlying error. |
| `No module named scipy` | Run `python -m pip install -r requirements.txt` in the same environment used to launch the server. |
| Saved job disappears after restart | Reopen downloaded files using `preview_saved.py`; in-memory job IDs are not persistent. |

Automatic detection remains a geometric heuristic. Reflections, thin grooves lost at reconstruction resolution, occlusion, and noisy surfaces can affect the result. Curved-joint tracing needs a reasonably flat base face; a curved member on a curved base (e.g. pipe-to-pipe) is not handled. When both faces of a thick standing plate are reconstructed the traced path lies near the plate's mid-thickness, roughly half a plate thickness from each fillet root. Closed loops must be roughly star-shaped about their centre; branches and several competing joints are not reliably resolved (the longest contact footprint is chosen).

Metric3D can reuse a learned correction in `camera_calib.json`. That correction is not a substitute for camera-specific calibration, and its validity should be reconsidered when the camera or capture setup changes.

## Project files

```text
weld_planner/
├── server.py             # Flask backend and reconstruction pipeline
├── seam_path.py          # Graph tracing and depth alignment helpers
├── index.html            # Browser point-cloud and path viewer
├── validate_photos.py    # Run the maintained pipeline on local photos
├── preview_saved.py      # Serve an existing reconstruction/result pair
├── test_seam_path.py     # Regression tests
├── requirements.txt      # Python dependencies
├── examples/             # Synthetic straight/curved preview data
├── sim/                  # Gazebo round-part scenes, rendering, and length checks
├── vggt/                 # VGGT model source
├── test.py               # Legacy standalone workflow
├── weld_pipeline.py      # Legacy reconstruction/calibration workflow
└── weld_ros_bridge.py    # Optional legacy ROS2 integration
```

The legacy scripts and ROS2-related files are separate from the maintained visual-preview workflow described here. Robot-send controls are hidden in the viewer, and no ROS2 connection is needed for reconstruction, inspection, or export.
