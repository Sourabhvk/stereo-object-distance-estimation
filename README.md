# Stereo Object Distance Estimation

Clear, robust estimation of object distance using stereo depth and object detections.

Summary
- **Goal:** Estimate metric distance (meters) to detected objects in an image using a precomputed stereo depth map and object detection.
- **Key ideas:** combine YOLO object detection with robust depth sampling inside boxes; remove outliers with MAD/IQR; prefer center pixels via spatial selection or weighted median.

Quick demo
- Interactive UI: `scripts/click_object_depth_ui.py` — click a detection to display estimated object distance; press `m` to toggle the spatial mode (`select` vs `weighted`).

Why this is recruiter-friendly
- Short, demoable feature that ties together model usage (YOLO), stereo vision basics (depth from disparity), robust statistics, and an interactive visualization.
- Shows practical engineering: reproducible scripts, small tests, and a toggleable UI for qualitative evaluation.

Quick start
1. Create and activate a Python virtual environment (PowerShell):
```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
2. Ensure sample data exists:
- `data/sample/left/000000_10.png`
- `outputs/depth.npy`
3. Run the interactive UI:
```bash
python scripts/click_object_depth_ui.py
```
4. Run quick unit checks:
```bash
python scripts/test_estimate_box_depth.py
python scripts/test_object_depth.py
```

Important files
- `src/interactive/object_depth.py` — core estimator: `estimate_box_depth()` (supports `spatial_method='select'|'weighted'`).
- `scripts/click_object_depth_ui.py` — demo + UI toggle.
- `scripts/test_estimate_box_depth.py` and `scripts/test_object_depth.py` — small tests.

More details
- See `docs/LEARN_IT.md` for an in-depth explanation of the computer-vision concepts used here and an interview-style Q&A.

License
- See top-level `LICENSE` file.

Contact
- Want a live demo or a short walk-through? Run the UI and try clicking detections.
