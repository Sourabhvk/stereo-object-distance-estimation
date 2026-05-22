import cv2
import numpy as np


def estimate_box_depth(depth, box, center_ratio=0.5):
    """Estimate object distance using valid depth pixels near the box center."""
    x1, y1, x2, y2 = box

    box_width = x2 - x1
    box_height = y2 - y1

    # YOLO boxes often include background around the object.
    # The center region is more likely to belong to the clicked object itself.
    center_width = int(box_width * center_ratio)
    center_height = int(box_height * center_ratio)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    cx1 = max(center_x - center_width // 2, x1)
    cx2 = min(center_x + center_width // 2, x2)
    cy1 = max(center_y - center_height // 2, y1)
    cy2 = min(center_y + center_height // 2, y2)

    patch = depth[cy1:cy2, cx1:cx2]
    valid_depths = patch[patch > 0]

    if valid_depths.size == 0:
        return None

    # Median reduces the effect of noisy stereo pixels.
    return float(np.median(valid_depths))


def find_clicked_detection(detections, x, y):
    """Return the smallest detected box that contains the clicked point."""
    matches = []

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]

        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            matches.append((area, detection))

    if not matches:
        return None

    return min(matches, key=lambda item: item[0])[1]


def draw_detections(image, detections):
    """Draw all detected COCO objects before the user clicks."""
    display = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = f"{detection['label']} {detection['confidence']:.2f}"

        cv2.rectangle(display, (x1, y1), (x2, y2), (80, 220, 80), 2)
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
