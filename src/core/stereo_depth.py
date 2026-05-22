import cv2
import numpy as np


def load_stereo_pair(left_path, right_path):
    """Load a left/right stereo pair as grayscale images."""
    # Stereo matching works on intensity values, so grayscale is enough here.
    left_img = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

    if left_img is None:
        raise FileNotFoundError(f"Could not load left image: {left_path}")

    if right_img is None:
        raise FileNotFoundError(f"Could not load right image: {right_path}")

    if left_img.shape != right_img.shape:
        raise ValueError("Left and right images must have the same shape.")

    return left_img, right_img


def compute_disparity(left_img, right_img):
    """Compute a disparity map using OpenCV StereoSGBM."""
    # numDisparities controls the search range and must be divisible by 16.
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=96,
        blockSize=5,
        P1=8 * 1 * 5**2,
        P2=32 * 1 * 5**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparity = stereo.compute(left_img, right_img).astype(np.float32)

    # OpenCV stores StereoSGBM disparity values scaled by 16.
    disparity = disparity / 16.0
    return disparity


def disparity_to_depth(disparity, focal_length, baseline):
    """Convert disparity in pixels to approximate depth in meters."""
    depth = np.zeros_like(disparity, dtype=np.float32)

    # Disparity must be positive to avoid invalid depth values.
    valid_pixels = disparity > 0
    depth[valid_pixels] = (focal_length * baseline) / disparity[valid_pixels]

    return depth


def normalize_for_visualization(image):
    """Normalize an image array to 8-bit for saving or display."""
    # Ignore zero values because they represent invalid or unknown pixels.
    valid_pixels = image > 0

    if not np.any(valid_pixels):
        return np.zeros_like(image, dtype=np.uint8)

    normalized = np.zeros_like(image, dtype=np.uint8)
    normalized[valid_pixels] = cv2.normalize(
        image[valid_pixels],
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    ).astype(np.uint8).reshape(-1)

    return normalized
