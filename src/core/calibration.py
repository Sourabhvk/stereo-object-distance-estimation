from pathlib import Path


def load_calibration(calib_path):
    """Load simple key=value stereo calibration values."""
    calib_path = Path(calib_path)
    values = {}

    with calib_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # Allow calib.txt to include human-readable notes.
            if not line or line.startswith("#"):
                continue

            # Expected format: focal_length_px=721.5377
            key, value = line.split("=", maxsplit=1)
            values[key.strip()] = float(value.strip())

    # Return only the values needed by the depth formula.
    return values["focal_length_px"], values["baseline_m"]
