from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

LEFT_IMAGE_PATH = PROJECT_ROOT / "data" / "sample" / "left" / "000000_10.png"
DEPTH_PATH = PROJECT_ROOT / "outputs" / "depth.npy"

from src.interactive.click_depth import get_local_depth


def main():
    # The UI displays the original left image, but reads depth from depth.npy.
    image = cv2.imread(str(LEFT_IMAGE_PATH))
    depth = np.load(DEPTH_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {LEFT_IMAGE_PATH}")

    if image.shape[:2] != depth.shape:
        raise ValueError("Image and depth map must have the same height and width.")

    display = image.copy()
    window_name = "Click Depth UI"

    def on_mouse(event, x, y, flags, param):
        nonlocal display

        # Only estimate depth when the user left-clicks on the image.
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        display = image.copy()
        depth_m = get_local_depth(depth, x, y)

        if depth_m is None:
            label = "No valid depth"
            print(f"x={x}, y={y}, depth=invalid")
        else:
            label = f"{depth_m:.2f} m"
            print(f"x={x}, y={y}, depth={depth_m:.2f} m")

        # Draw the selected point and depth label on the displayed image.
        cv2.circle(display, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(
            display,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    # Keep the window open until the user presses q or Esc.
    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
