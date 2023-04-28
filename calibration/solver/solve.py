import numpy as np

from .extrinsics import solve_extrinsic
from .intrinsics import solve_intrinsic


def solve(
    x: np.ndarray, X: np.ndarray, image_center: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x -= image_center
    p = solve_extrinsic(x, X)
    lambdas, t_3 = solve_intrinsic(x, X, p)
    p[2, 2] = t_3
    t = p[:, 2]
    R = np.c_[p[:, :2], np.cross(p[:, 0], p[:, 1])]
    return lambdas, R, t
