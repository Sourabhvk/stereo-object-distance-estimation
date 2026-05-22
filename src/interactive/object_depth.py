import cv2
import numpy as np


def estimate_box_depth(
    depth,
    box,
    center_ratio=0.5,
    min_valid_pixels=200,
    expand_factor=1.5,
    mad_multiplier=3.0,
    max_pixels=500,
    spatial_method="select",
    spatial_eps=1e-6,
):
    """Estimate object distance robustly from depth within a detection box.

    Parameters
    - depth: 2D numpy array of metric depth values (meters) or 0 for invalid.
    - box: tuple (x1, y1, x2, y2) pixel coordinates in image space.
    - center_ratio: initial fraction of the box to use around the center (0-1).
    - min_valid_pixels: target minimum number of valid depth pixels to gather.
    - expand_factor: multiplicative factor to grow the center crop when not enough
        valid pixels are found. Repeatedly applied until the full box is used.
    - mad_multiplier: multiplier for Median Absolute Deviation (MAD) to discard
        outliers. Larger values are more permissive.
    - max_pixels: cap on how many closest valid pixels to consider for the final median.
        - spatial_method: 'select' (choose closest pixels) or 'weighted' (weighted median by
            inverse pixel distance to box center).
        - spatial_eps: small epsilon to avoid division by zero when computing weights.

    Algorithm overview / rationale
    1. YOLO bounding boxes can include background; the object usually lies near the
       bounding-box center. We therefore start with a smaller central crop to reduce
       background contamination.
    2. If too few valid depth pixels exist in the central crop (e.g., due to
       occlusions, missing disparity), we progressively expand the crop toward the
       full box until we reach `min_valid_pixels` or use the full box.
    3. Collect valid (positive) depth pixels and their image coordinates.
    4. Compute a robust central value (median) and remove outliers using MAD.
       If MAD is zero (degenerate), fall back to an IQR-based filter.
    5. Prefer depth pixels nearer the bounding-box center (spatial weighting):
       sort by image distance to the center and keep up to `max_pixels` closest
       values to compute the final median. This reduces influence from background
       regions that survived the outlier filter.

    Returns
    - median depth (float) in meters, or None if no valid depth pixels were found.
    """

    x1, y1, x2, y2 = box

    # Ensure valid integer box dimensions (avoid zero-width/height)
    box_width = max(int(x2 - x1), 1)
    box_height = max(int(y2 - y1), 1)

    # Box center in image coordinates
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    cur_ratio = float(center_ratio)

    # We'll collect valid pixel coordinates and depths here
    valid_coords = None
    valid_depths = None

    # Expand center crop until we have enough valid pixels or we've covered the full box
    while True:
        center_w = int(box_width * cur_ratio)
        center_h = int(box_height * cur_ratio)

        # Compute crop coordinates, clamped to the box boundaries
        cx1 = max(center_x - center_w // 2, x1)
        cx2 = min(center_x + center_w // 2, x2)
        cy1 = max(center_y - center_h // 2, y1)
        cy2 = min(center_y + center_h // 2, y2)

        # Extract the depth patch for the current crop
        patch = depth[cy1:cy2, cx1:cx2]

        # If the patch is empty (unexpected), bail out
        if patch.size == 0:
            return None

        # Find coordinates of valid depth pixels (depth > 0)
        ys, xs = np.nonzero(patch > 0)
        if ys.size > 0:
            # Depths from the patch, convert to float for numerical stability
            ds = patch[ys, xs].astype(float)
            # Convert local patch coordinates back to full-image coordinates
            img_xs = xs + cx1
            img_ys = ys + cy1
            valid_coords = np.column_stack((img_xs, img_ys))
            valid_depths = ds

        # Stop expanding if we gathered enough valid pixels
        if valid_depths is not None and valid_depths.size >= min_valid_pixels:
            break

        # If we've already expanded to the full box, stop
        if cur_ratio >= 1.0:
            break

        # Grow the crop and try again
        cur_ratio = min(1.0, cur_ratio * expand_factor)

    # No valid depth pixels found in the box at all
    if valid_depths is None or valid_depths.size == 0:
        return None

    # Use the median as a robust starting estimate
    median_depth = float(np.median(valid_depths))

    # Robust outlier detection: Median Absolute Deviation (MAD)
    diffs = np.abs(valid_depths - median_depth)
    mad = np.median(diffs)

    if mad > 0:
        # Keep depths within mad_multiplier * MAD of the median
        mask = diffs <= (mad_multiplier * mad)
    else:
        # Degenerate case: MAD is zero -> fall back to Interquartile Range (IQR)
        q1, q3 = np.percentile(valid_depths, [25.0, 75.0])
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (valid_depths >= lower) & (valid_depths <= upper)
        else:
            # All values are identical; keep them
            mask = np.ones_like(valid_depths, dtype=bool)

    filtered_depths = valid_depths[mask]
    filtered_coords = valid_coords[mask]

    # If every depth was removed as an outlier, return the coarse median
    if filtered_depths.size == 0:
        return median_depth

    # Spatial preference: prefer pixels closer to the box center
    center_point = np.array([center_x, center_y])
    dists = np.linalg.norm(filtered_coords - center_point, axis=1)
    order = np.argsort(dists)

    # Two spatial strategies:
    # - 'select': pick up to `max_pixels` closest pixels and return their median (default)
    # - 'weighted': compute a weighted median using weights = 1 / (distance + eps)
    if spatial_method == "select":
        sel = order[: max(1, min(filtered_depths.size, max_pixels))]
        final_depths = filtered_depths[sel]
        return float(np.median(final_depths))

    if spatial_method == "weighted":
        # Use all filtered pixels but weight them by inverse distance to center
        # Avoid division by zero by adding a small epsilon
        weights = 1.0 / (dists + float(spatial_eps))

        # Weighted median implementation: sort by depth and find cumulative weight >= 50%
        idx = np.argsort(filtered_depths)
        vals = filtered_depths[idx]
        w = weights[idx].astype(float)
        cumsum = np.cumsum(w)
        cutoff = 0.5 * cumsum[-1]
        im = np.searchsorted(cumsum, cutoff, side="left")
        im = int(min(im, vals.size - 1))
        return float(vals[im])

    # Unknown spatial method -> fallback to simple median
    sel = order[: max(1, min(filtered_depths.size, max_pixels))]
    final_depths = filtered_depths[sel]
    return float(np.median(final_depths))


def find_clicked_detection(detections, x, y):
    """Return the smallest detected box that contains the clicked point."""
    # We may have multiple detections containing the click; choose the smallest
    # (by area) since smaller boxes are more likely to tightly enclose the object
    matches = []

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]

        # Check whether the click falls inside the detection box (inclusive)
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            matches.append((area, detection))

    if not matches:
        return None

    # Return detection with smallest area
    return min(matches, key=lambda item: item[0])[1]


def draw_detections(image, detections):
    """Draw all detected COCO objects before the user clicks."""
    display = image.copy()
    # Draw each detection with a label showing class name and confidence
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = f"{detection['label']} {detection['confidence']:.2f}"

        # Green boxes for detections
        cv2.rectangle(display, (x1, y1), (x2, y2), (80, 220, 80), 2)

        # Position text slightly above the top-left corner of the box; clamp to image
        cv2.putText(
            display,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 220, 80),
            2,
            cv2.LINE_AA,
        )

    return display
