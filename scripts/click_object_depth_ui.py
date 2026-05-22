from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

LEFT_IMAGE_PATH = PROJECT_ROOT / "data" / "sample" / "left" / "000000_10.png"
DEPTH_PATH = PROJECT_ROOT / "outputs" / "depth.npy"

from src.interactive.object_depth import (
    draw_detections,
    estimate_box_depth,
    find_clicked_detection,
)


def main():
    image = cv2.imread(str(LEFT_IMAGE_PATH))
    depth = np.load(DEPTH_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {LEFT_IMAGE_PATH}")

    if image.shape[:2] != depth.shape:
        raise ValueError("Image and depth map must have the same height and width.")

    # yolov8n is a small COCO-pretrained model. First run may download weights.
    model = YOLO("yolov8n.pt")
    result = model(image, verbose=False)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        detections.append(
            {
                "box": (x1, y1, x2, y2),
                "label": model.names[class_id],
                "confidence": confidence,
            }
        )

    display = draw_detections(image, detections)
    window_name = "Click Object Depth UI"

    def on_mouse(event, x, y, flags, param):
        nonlocal display

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        display = draw_detections(image, detections)
        detection = find_clicked_detection(detections, x, y)

        if detection is None:
            print(f"x={x}, y={y}, object=none")
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            return

        depth_m = estimate_box_depth(depth, detection["box"])
        x1, y1, x2, y2 = detection["box"]

        if depth_m is None:
            label = f"{detection['label']} depth: invalid"
            print(f"object={detection['label']}, depth=invalid")
        else:
            label = f"{detection['label']} {depth_m:.2f} m"
            print(
                f"object={detection['label']}, "
                f"confidence={detection['confidence']:.2f}, "
                f"depth={depth_m:.2f} m"
            )

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(display, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(
            display,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
