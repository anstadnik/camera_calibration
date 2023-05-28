import numpy as np
from PIL import ImageOps
from PIL.Image import Image
from cbdetect_py import hessian_response
from tqdm.auto import tqdm
from calibration.benchmark.benchmark_result import BenchmarkResult
from calibration.data.babelcalib.entry import Entry
from calibration.feature_detector.checkerboard import detect_corners
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import scipy.ndimage as ndi


def find_modes_meanshift(hist):
    # samples = np.array([[i] * int(v) for i, v in enumerate(hist)]).reshape(-1, 1)

    hist = hist.astype(int)
    samples = np.repeat(np.arange(len(hist)), hist).reshape(-1, 1)
    # bandwidth = estimate_bandwidth(samples, quantile=0.2, n_samples=500)
    ms = MeanShift(bin_seeding=True).fit(samples)

    # Retrieve unique labels and cluster centers
    labels_unique = np.unique(ms.labels_)
    cluster_centers = ms.cluster_centers_

    modes = sorted(
        [
            (cluster_centers[k][0], np.mean(samples[ms.labels_ == k]))
            for k in labels_unique
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    return modes


def compute_orientation(img, positions, window_size):
    def sobel_filters(img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndi.filters.convolve(img, Kx)
        Iy = ndi.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)

        return (G, theta)

    # G, theta = sobel_filters(img)
    Ix = ndi.sobel(img.astype(int), axis=0)
    Iy = ndi.sobel(img.astype(int), axis=1)
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    # import plotly.express as px
    # px.imshow(G).show()
    # px.imshow(theta).show()

    # histogram_modes = []
    passes = []
    i = 0
    for position in tqdm(positions):
        i += 1
        x, y = position
        if (
            x - window_size // 2 >= 0
            and x + window_size // 2 < img.shape[1]
            and y - window_size // 2 >= 0
            and y + window_size // 2 < img.shape[0]
        ):
            window = theta[
                y - window_size // 2 : y + window_size // 2 + 1,
                x - window_size // 2 : x + window_size // 2 + 1,
            ]
            weights = G[
                y - window_size // 2 : y + window_size // 2 + 1,
                x - window_size // 2 : x + window_size // 2 + 1,
            ]

            histogram, bin_edges = np.histogram(
                window, bins=32, weights=weights, range=(-np.pi, np.pi)
            )

            if np.allclose(histogram, 0):
                passes.append(False)
                continue
            # ms = MeanShift(bin_seeding=True)
            # # import plotly.express as px
            # # px.bar(y=histogram, x=bin_edges[1:]).show()
            # ms.fit(histogram.reshape(-1, 1))
            # histogram_modes = ms.cluster_centers_[:, 0]

            histogram_modes = np.array(find_modes_meanshift(histogram))
            passes.append((2 * histogram_modes > histogram_modes.max()).sum() == 2)
            # histogram_modes.append(sorted(ms.cluster_centers_[:, 0]))
            # if i > 10:
            #     break
        else:
            passes.append(False)

    return passes


def prune_corners(
    corners: np.ndarray, mask: np.ndarray, image: Image, thr: float
) -> tuple[np.ndarray, np.ndarray]:
    # cornerness = detect_corners(np.array(ImageOps.grayscale(image)))
    cornerness = hessian_response(np.array(ImageOps.grayscale(image)))
    if cornerness.shape[0] == image.size[1] // 2:
        assert cornerness.shape[1] == image.size[0] // 2
        cornerness = cornerness.repeat(2, axis=0).repeat(2, axis=1)
    # assert cornerness.shape == ImageOps.grayscale(image).size

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


def get_corner_responses(r: BenchmarkResult):
    assert isinstance(r.input, Entry)
    assert r.input.image is not None
    assert r.features is not None
    cornerness = detect_corners(np.array(ImageOps.grayscale(r.input.image)))
    # cornerness = hessian_response(np.array(ImageOps.grayscale(r.input.image)))
    if cornerness.shape[0] == r.input.image.size[1] // 2:
        assert cornerness.shape[1] == r.input.image.size[0] // 2
        cornerness = cornerness.repeat(2, axis=0).repeat(2, axis=1)

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
