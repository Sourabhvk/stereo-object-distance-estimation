import os
import sys
import numpy as np

# Ensure the project root is on sys.path so `src` can be imported when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.interactive.object_depth import estimate_box_depth


def run():
    # Case 1: uniform patch
    depth = np.zeros((100, 100), dtype=float)
    depth[30:70, 30:70] = 5.0
    box = (30, 30, 70, 70)
    print("Uniform ->", estimate_box_depth(depth, box))

    # Case 2: same patch with a far outlier
    depth2 = np.zeros((100, 100), dtype=float)
    depth2[30:70, 30:70] = 5.0
    depth2[50, 50] = 50.0
    print("With outlier (select) ->", estimate_box_depth(depth2, box, spatial_method="select"))
    print("With outlier (weighted) ->", estimate_box_depth(depth2, box, spatial_method="weighted"))

    # Case 3: sparse valid pixels (requires small min_valid_pixels)
    depth3 = np.zeros((100, 100), dtype=float)
    depth3[49, 49] = 4.5
    box2 = (40, 40, 60, 60)
    print("Sparse single pixel ->", estimate_box_depth(depth3, box2, min_valid_pixels=1))


if __name__ == "__main__":
    run()
