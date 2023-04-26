import numpy as np

from .extrinsics import solve_extrinsic
from .intrinsics import solve_intrinsic


def solve(
    x: np.ndarray, X: np.ndarray, image_center: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p = solve_extrinsic(x, X, image_center)
    lambdas, t_3 = solve_intrinsic(x, X, p)
    p[2, 2] = t_3
    return lambdas, p
