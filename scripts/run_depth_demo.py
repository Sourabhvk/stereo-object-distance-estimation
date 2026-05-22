from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.core.calibration import load_calibration
from src.core.stereo_depth import (
    compute_disparity,
    disparity_to_depth,
    load_stereo_pair,
    normalize_for_visualization,
)


def main():
    left_path = PROJECT_ROOT / "data" / "sample" / "left" / "000000_10.png"
    right_path = PROJECT_ROOT / "data" / "sample" / "right" / "000000_10.png"
    calib_path = PROJECT_ROOT / "data" / "sample" / "calib" / "calib.txt"
    output_dir = PROJECT_ROOT / "outputs"

    output_dir.mkdir(exist_ok=True)

    # 1. Load sample stereo images.
    left_img, right_img = load_stereo_pair(left_path, right_path)

    # 2. Load KITTI-style calibration values.
    focal_length_px, baseline_m = load_calibration(calib_path)

    # 3. Compute disparity from the left/right image pair.
    disparity = compute_disparity(left_img, right_img)

    # 4. Convert disparity into approximate depth in meters.
    depth = disparity_to_depth(disparity, focal_length_px, baseline_m)

    # 5. Save visualizations that are easy to inspect.
    disparity_vis = normalize_for_visualization(disparity)
    depth_vis = normalize_for_visualization(depth)

    cv2.imwrite(str(output_dir / "disparity.png"), disparity_vis)
    cv2.imwrite(str(output_dir / "depth.png"), depth_vis)
    np.save(output_dir / "depth.npy", depth)

    print(f"Saved disparity visualization to: {output_dir / 'disparity.png'}")
    print(f"Saved depth visualization to: {output_dir / 'depth.png'}")
    print(f"Saved metric depth array to: {output_dir / 'depth.npy'}")


if __name__ == "__main__":
    main()
