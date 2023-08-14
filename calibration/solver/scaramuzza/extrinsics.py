import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd


def solve_extrinsic(
    x: NDArray[np.float64], X: NDArray[np.float64]
) -> list[NDArray[np.float64]]:
    """
    Computes the extrinsic parameters of a camera, given the 2D image points and
    the 3D world points.

    Args:
        x (NDArray[np.float64]): Points in image space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].
        X (NDArray[np.float64]): Points in the board space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].
        image_center (tuple[float, float]): A tuple representing the
            image center coordinates (x, y).

    Returns:
        NDArray[np.float64]: A 3x3 array representing the extrinsic parameters of the camera.
    """
    M = np.vstack(
        [[-v * X_, -v * Y_, u * X_, u * Y_, -v, u] for (u, v), (X_, Y_) in zip(x, X)]
    )

    V: NDArray[np.float64]
    _, _, V = svd(M, full_matrices=False)
    assert isinstance(V, np.ndarray)
    # Assert that it's np array of float
    H = V.T[:, -1]
    r_11, r_12, r_21, r_22, t_1, t_2 = H

    AA = (r_11 * r_12 + r_21 * r_22) ** 2
    BB = r_11**2 + r_21**2
    CC = r_12**2 + r_22**2

    # TODO: Use resolution
    r_32_2 = np.roots([1, CC - BB, -AA])
    r_32_2 = r_32_2[(r_32_2 >= 0) & (r_32_2 <= 1000)]

    # r_32_2_closed_form = [
    #     r for r in orthonormality_closed_form(r_11, r_12, r_21, r_22)
    #     if 0 <= r <= 1000
    # ]
    #
    # assert np.allclose(r_32_2, r_32_2_closed_form)
    assert len(r_32_2) != 0

    r_31, r_32 = [], []
    for r_32_2_ in r_32_2:
        for sg in [-1, 1]:
            r_32_ = sg * np.sqrt(r_32_2_)
            r_32.append(r_32_)
            if np.isclose(r_32_, 0):
                # r_31 += [np.sqrt(CC - BB), -np.sqrt(CC - BB)]
                r_31 += [np.sqrt(abs(CC - BB)), -np.sqrt(abs(CC - BB))]
                r_32.append(r_32_)
            else:
                r_31.append(-(r_11 * r_12 + r_21 * r_22) / r_32_)

    RR = np.zeros((3, 3, len(r_32) * 2))
    count = 0
    for i1 in range(len(r_32)):
        for sg in [-1, 1]:
            count += 1
            Lb = 1 / np.linalg.norm([r_11, r_21, r_31[i1]])
            RR[:, :, count - 1] = (
                sg
                * Lb
                * np.array(
                    [[r_11, r_12, t_1], [r_21, r_22, t_2], [r_31[i1], r_32[i1], 0.0]]
                )
            )

    # for i1 in range(RR.shape[2]):
    #     print(f"RR[:, :, {i1}] = \n{RR[:, :, i1]}")

    np.testing.assert_array_equal(X[0], [0, 0])

    # This part is designed to find Hs which correspond to the correct side from
    # the camera's point of view (i.e. it should be in front of the camera and
    #                               not behind it)
    RR_t1_dists = np.linalg.norm(RR[:2, 2, :] - x[0][:, np.newaxis], axis=0)
    min_, max_ = np.min(RR_t1_dists), np.max(RR_t1_dists)
    min_ids = np.nonzero(RR_t1_dists < min_ + ((max_ - min_) / 100))[0]
    return [RR[:, :, i] for i in min_ids]
