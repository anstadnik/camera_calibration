import numpy as np
from PIL import ImageOps
from PIL.Image import Image
from calibration.feature_detector.checkerboard import detect_corners


def prune_corners(
        corners: np.ndarray, mask: np.ndarray, image: Image, thr:float
) -> tuple[np.ndarray, np.ndarray]:
    cornerness = detect_corners(np.array(ImageOps.grayscale(image)))
    corners = corners.astype(int)[mask]
    responses = cornerness[corners[:, 1], corners[:, 0]]
    # Now mask is an array of 0 and 1
    mask = mask.astype(int)
    # Now, if responces > thr, mask is 2, otherwise 1. 0s are unchanged
    mask[mask==1] = (responses > thr).astype(int) + 1
    return responses, mask
