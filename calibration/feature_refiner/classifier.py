from calibration.benchmark.benchmark import BenchmarkResult
import numpy as np
from PIL import ImageOps
from calibration.data.babelcalib.entry import Entry
from calibration.feature_detector.checkerboard import detect_corners


def get_corner_responses(r: BenchmarkResult):
    assert isinstance(r.input, Entry)
    assert r.input.image is not None
    assert r.features is not None
    cornerness = detect_corners(np.array(ImageOps.grayscale(r.input.image)))

    corners = r.features.corners.astype(int)

    # Preparing output array
    outputs = []

    # Padding image to handle edge cases
    pad_image = np.pad(cornerness, ((5, 5), (5, 5)), mode="constant")

    # Shifting corners because of padding
    corners += 5

    # Iterate over corners
    for corner in corners:
        x, y = corner
        # Get 11x11 window around corner
        window = pad_image[y - 5 : y + 6, x - 5 : x + 6].flatten()
        # Get distances to the corner
        yy, xx = np.mgrid[-5:6, -5:6]
        distances = np.hypot(xx, yy).flatten()
        # Get corner_id
        # ids = np.full(distances.shape, idx)
        # Stack arrays
        output = np.column_stack((window, distances))
        outputs.append(output)

    return np.concatenate(outputs)
