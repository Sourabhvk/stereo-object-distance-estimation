import numpy as np


def get_local_depth(depth, x, y, window_size=9):
    """Estimate depth near a click using nearby valid pixels."""
    half = window_size // 2
    y1 = max(y - half, 0)
    y2 = min(y + half + 1, depth.shape[0])
    x1 = max(x - half, 0)
    x2 = min(x + half + 1, depth.shape[1])

    patch = depth[y1:y2, x1:x2]
    valid_depths = patch[patch > 0]

    if valid_depths.size == 0:
        return None

    # Median is more stable than a single pixel because disparity can be noisy.
    return float(np.median(valid_depths))
