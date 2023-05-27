import numpy as np
from PIL import ImageOps
from PIL.Image import Image
from calibration.feature_detector.checkerboard import detect_corners


def prune_corners(
    corners: np.ndarray, mask: np.ndarray, image: Image, thr: float
) -> tuple[np.ndarray, np.ndarray]:
    cornerness = detect_corners(np.array(ImageOps.grayscale(image)))

    corners = corners.astype(int)[mask]
    # 0: original, 1: new
    mask = mask.astype(int)

    out_of_img = ((corners < 0) | (corners >= image.size)).any(axis=1).astype(int)
    # Now mask is an array of 0, 1 and 3
    mask[mask == 1] = 1 + out_of_img * 2
    corners = corners[out_of_img == 0]

    responses = cornerness[corners[:, 1], corners[:, 0]]

    # Now mask is an array of 0 and 1
    # Now, if responces > thr, mask is 2, otherwise 1. 0s are unchanged
    mask[mask == 1] = (responses > thr).astype(int) + 1

    # Mask:
    # 0 - unchanged
    # 1 - filtered out
    # 2 - new corner
    # 3 - out of image
    return responses, mask
