import numpy as np
from PIL import Image
from typing import Tuple


def generate_heightmap(
    image_path: str,
    target_width_mm: float,
    target_height_mm: float,
    min_thickness_mm: float,
    max_thickness_mm: float,
    pixel_size_mm: float = 0.1,
    invert: bool = True,
) -> Tuple[np.ndarray, int, int]:
    """
    Loads an image, converts it to grayscale, resizes to target physical dimensions,
    and returns a 2D numpy array of Z heights (in mm) along with grid dimensions.

    - pixel_size_mm: Spatial resolution (0.1 mm = 10 px/mm, optimal for SLA/MSLA resin).
    - invert: True for lithophanes (darker pixels = thicker material to block light).
    """
    img = Image.open(image_path).convert("L")

    # Compute target resolution in pixels
    cols = max(2, int(round(target_width_mm / pixel_size_mm)))
    rows = max(2, int(round(target_height_mm / pixel_size_mm)))

    # High-quality Lanczos resampling to match physical aspect ratio
    resized_img = img.resize((cols, rows), Image.Resampling.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)

    # Normalize pixel luminance to 0.0 - 1.0 range
    normalized = img_array / 255.0

    if invert:
        # 0 (black) becomes 1.0 (max thickness), 255 (white) becomes 0.0 (min thickness)
        normalized = 1.0 - normalized

    # Map normalized values to physical thickness range (mm)
    thickness_delta = max_thickness_mm - min_thickness_mm
    heightmap = min_thickness_mm + (normalized * thickness_delta)

    return heightmap, rows, cols