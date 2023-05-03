import numpy as np

from .extrinsics import solve_extrinsic
from .intrinsics import solve_intrinsic


def solve(
    x: np.ndarray, X: np.ndarray, intrinsic_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # x = np.c_[x, np.ones(x.shape[0])]
    # x = np.linalg.pinv(intrinsic_matrix) @ x.T
    # x = x.T[:, :2]
    x -= intrinsic_matrix[:2, 2]
    assert np.linalg.norm(x, axis=0).shape == (2,)
    # biggest_r_i = np.argmax(np.linalg.norm(x, axis=1))
    biggest_r = np.max(np.linalg.norm(x, axis=1))
    x /= biggest_r
    # x_max = x.max(axis=0)
    # print(x_max.shape)
    # x /= x_max
    p = solve_extrinsic(x, X)
    lambdas, t_3 = solve_intrinsic(x, X, p)
    p[2, 2] = t_3
    t = p[:, 2]
    t *= biggest_r
    R = np.c_[p[:, :2], np.cross(p[:, 0], p[:, 1])]
    return lambdas, R, t
