from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.interactive.object_depth import estimate_box_depth

DEPTH_PATH = PROJECT_ROOT / "outputs" / "depth.npy"

def main():
    depth = np.load(DEPTH_PATH)
    # sample box roughly centered in KITTI sample image
    box = (150, 120, 470, 380)
    depth_m = estimate_box_depth(depth, box)
    print(f"Estimated depth for box {box}: {depth_m}")

if __name__ == '__main__':
    main()
