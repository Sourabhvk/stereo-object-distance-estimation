# Stereo Object Distance Estimation

Stereo vision project for estimating scene depth (in meters) from left/right camera images, then using that depth map for point-click and object-level distance inspection.

---

## 1) Project Goal

This repository demonstrates an end-to-end stereo depth pipeline:

1. Load a rectified stereo image pair (left + right).
2. Compute a disparity map using OpenCV StereoSGBM.
3. Convert disparity to metric depth with camera calibration.
4. Interactively inspect depth:
   - click a pixel region for local depth
   - click detected objects (YOLO) for robust object distance

The focus is learning practical computer vision concepts, not building a production-grade depth system.

---

## 2) Core Concepts

### 2.1 Stereo Vision

A stereo setup uses two horizontally separated cameras viewing the same scene.

- **Baseline (B)**: physical distance between camera centers (meters).
- **Focal length (f)**: camera focal length in pixel units.
- **Disparity (d)**: horizontal shift (in pixels) of the same 3D point between left and right images.

Nearby objects have larger disparity; far objects have smaller disparity.

### 2.2 Depth from Disparity

Depth is estimated by:

`Z = (f * B) / d`

Where:

- `Z` = depth in meters
- `f` = focal length in pixels
- `B` = baseline in meters
- `d` = disparity in pixels

In this project:

- `focal_length_px` and `baseline_m` are loaded from a simple calibration file.
- depth is computed only for valid disparity pixels (`d > 0`).

### 2.3 Why StereoSGBM

The project uses OpenCV **StereoSGBM** (`cv2.StereoSGBM_create`) because it is:

- classical and interpretable,
- easy to run locally,
- suitable for learning the stereo pipeline.

It estimates disparity by matching local image structure while regularizing across directions.

### 2.4 Why Depth Is Noisy

Stereo matching struggles when correspondence is ambiguous:

- low texture regions (road/sky/walls),
- reflections or shiny surfaces,
- occlusions (visible in one camera, not the other),
- illumination differences,
- very distant objects (tiny disparity).

So the depth map is approximate and should be interpreted as a robust estimate, not exact measurement.

---

## 3) Repository Structure

```text
src/
  core/
    calibration.py      # calibration parsing (focal length + baseline)
    stereo_depth.py     # stereo loading, disparity, depth conversion, visualization normalization
  interactive/
    click_depth.py      # local depth extraction around clicked pixel
    object_depth.py     # robust object-depth estimation from detection box

scripts/
  run_depth_demo.py         # builds disparity/depth outputs
  click_depth_ui.py         # click any point to inspect local depth
  click_object_depth_ui.py  # YOLO detections + click object for distance
  test_estimate_box_depth.py
  test_object_depth.py

data/
  sample/               # sample left/right/calibration placeholders
  KITTI_RAW DATA/       # current repo folder name (includes a space)

outputs/
  depth.npy             # metric depth map (generated)
  depth.png             # depth visualization (generated)
  disparity.png         # disparity visualization (generated)
```

---

## 4) Pipeline Walkthrough

### Step A: Load Stereo Pair

`load_stereo_pair()` reads left/right images as grayscale and validates shape equality.

Why grayscale: Stereo matching depends on intensity structure, not color semantics.

### Step B: Compute Disparity

`compute_disparity()` runs StereoSGBM with configured parameters (`numDisparities`, `blockSize`, penalties).

OpenCV returns disparity scaled by 16, so values are divided by 16.0.

### Step C: Convert to Depth

`disparity_to_depth()` applies the stereo formula on valid pixels (`disparity > 0`).

Invalid/unknown pixels remain 0 in depth map.

### Step D: Visualize

`normalize_for_visualization()` maps valid values to 8-bit range for PNG output while preserving invalid zeros.

### Step E: Interactive Inspection

### Pixel-level click depth (`click_depth_ui.py`)

- Loads left image + `outputs/depth.npy`.
- On click, samples a local window.
- Returns median of valid depths in that patch for stability.

### Object-level click depth (`click_object_depth_ui.py`)

- Runs YOLO (`yolov8n.pt`) on the left image.
- Draws detections and lets user click an object.
- Picks the smallest box containing click when overlaps exist.
- Estimates robust object depth from box depths.

---

## 5) Object Depth Estimation Logic

`estimate_box_depth()` is designed to reduce background noise inside detection boxes:

1. Start from a center crop of the box (`center_ratio`), since object core is usually central.
2. Expand crop progressively (`expand_factor`) until enough valid pixels (`min_valid_pixels`) or full box.
3. Remove outliers:
   - primary: MAD-based filtering (`mad_multiplier`)
   - fallback: IQR filtering when MAD is degenerate
4. Spatial preference toward box center:
   - `select`: keep nearest pixels (`max_pixels`) and take median
   - `weighted`: weighted median by inverse distance to center

This makes object distance less sensitive to boundary/background contamination.

---

## 6) Data and Calibration Notes

- The repo structure is prepared for KITTI-style stereo data (`image_2`, `image_3`, calibration folders).
- Current demo calibration parser expects a simple `key=value` file with:
  - `focal_length_px`
  - `baseline_m`
- Matching left/right images must be rectified and aligned for meaningful disparity.

---

## 7) Setup

From project root:

```bash
python -m pip install -r requirements.txt
```

Dependencies:

- opencv-python
- numpy
- matplotlib
- ultralytics

---

## 8) How to Run

### 8.1 Generate disparity and depth outputs

```bash
python scripts/run_depth_demo.py
```

Creates:

- `outputs/disparity.png`
- `outputs/depth.png`
- `outputs/depth.npy`

### 8.2 Pixel click UI

```bash
python scripts/click_depth_ui.py
```

Click anywhere to print/display local depth estimate.

### 8.3 Object click UI (YOLO + depth)

```bash
python scripts/click_object_depth_ui.py
```

Controls:

- Left click: estimate depth for clicked detection
- `m`: toggle spatial method (`select` / `weighted`)
- `q` or `Esc`: quit

---

## 9) Validation Scripts

```bash
python scripts/test_estimate_box_depth.py
python scripts/test_object_depth.py
```

Notes:

- `test_object_depth.py` expects `outputs/depth.npy` (run `run_depth_demo.py` first).
- Sample image placeholders may need real stereo files placed in the expected paths.

---

## 10) Limitations and Failure Cases

This project intentionally keeps the pipeline simple. Current limitations include:

- sensitivity to calibration/rectification quality,
- weak performance in textureless/reflective/occluded regions,
- unstable depth for far objects with tiny disparity,
- approximate object distance due to box-level aggregation,
- no confidence map or uncertainty estimation,
- no quantitative benchmarking against ground truth in current scripts.

---

## 11) Future Improvements

- Parse official KITTI calibration files directly.
- Add CLI arguments for custom image/calibration paths.
- Add confidence-based filtering and invalid-pixel handling.
- Benchmark against KITTI disparity/depth ground truth.
- Add temporal smoothing for video sequences.
- Improve interactive UI with richer overlays and diagnostics.
- Explore deep stereo/depth baselines for comparison.

---

## 12) Summary

The project teaches the practical bridge from stereo geometry to interactive distance estimation:

**stereo pair → disparity → metric depth → robust point/object distance inspection**.

It is a strong foundation for further work in autonomous driving perception, robotics, and 3D scene understanding.
