import numpy as np


def solve_extrinsic(x: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Computes the extrinsic parameters of a camera, given the 2D image points and
    the 3D world points.

    Args:
        x (np.ndarray): Points in image space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].
        X (np.ndarray): Points in the board space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].
        image_center (tuple[float, float]): A tuple representing the
            image center coordinates (x, y).

    Returns:
        np.ndarray: A 3x3 array representing the extrinsic parameters of the camera."""
    M = np.vstack(
        [[-v * X_, -v * Y_, u * X_, u * Y_, -v, u] for (u, v), (X_, Y_) in zip(x, X)]
    )

    _, _, V = np.linalg.svd(M, full_matrices=False)
    assert isinstance(V, np.ndarray)
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

    # for i in range(RR.shape[2]):
    #     ic(RR[:, :, i])

    minRR = np.inf
    minRR_ind = -1
    for min_count in range(RR.shape[2]):
        if np.linalg.norm(RR[:2, 2, min_count] - x[0]) < minRR:
            minRR = np.linalg.norm(RR[:2, 2, min_count] - x[0])
            minRR_ind = min_count

    H = RR[:, :, minRR_ind]

    return H
