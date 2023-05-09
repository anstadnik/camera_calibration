import numpy as np

from .extrinsics import solve_extrinsic
from .intrinsics import solve_intrinsic


def solve(
    x: np.ndarray, X: np.ndarray, intrinsic_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find intrinsic and extrinsic parameters for a set of points

    Args:
        x: [x, y] mx2 array of image points
        X: [x, y] mx2 array of board points
        intrinsic_matrix: Intrinsic matrix

    Returns:
        lambdas: [2x1] array of camera parameters
        R: [3x3] rotation matrix
        t: [1x1] translation
    """
    x = np.c_[x, np.ones(x.shape[0])]
    x = np.linalg.pinv(intrinsic_matrix) @ x.T
    x = x.T[:, :2]

    # x -= intrinsic_matrix[:2, 2]
    # assert np.linalg.norm(x, axis=0).shape == (2,)
    # biggest_r_i = np.argmax(np.linalg.norm(x, axis=1))
    # biggest_r = np.max(np.linalg.norm(x, axis=1))
    # x /= biggest_r
    # x_max = x.max(axis=0)
    # print(x_max.shape)
    # x /= x_max
    p = solve_extrinsic(x, X)
    lambdas, t_3 = solve_intrinsic(x, X, p)
    p[2, 2] = t_3
    t = p[:, 2]
    # t *= biggest_r
    R = np.c_[p[:, :2], np.cross(p[:, 0], p[:, 1])]
    return lambdas, R, t
