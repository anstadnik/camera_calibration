import numpy as np

from calibration.projector.camera import Camera
from calibration.projector.projector import Projector

from .extrinsics import solve_extrinsic
from .intrinsics import solve_intrinsic


def solve(x: np.ndarray, X: np.ndarray, camera: Camera) -> Projector:
    """Find intrinsic and extrinsic parameters for a set of points

    Args:
        x: [x, y] mx2 array of image points
        X: [x, y] mx2 array of board points
        camera: Camera object

    Returns:
        lambdas: [2x1] array of camera parameters
        R: [3x3] rotation matrix
        t: [1x1] translation
    """
    x = np.c_[x, np.ones(x.shape[0])]
    x = (np.linalg.inv(camera.intrinsic_matrix) @ x.T).T
    x /= x[:, 2][:, None]  # type: ignore
    x = x[:, :2]

    p = solve_extrinsic(x, X)
    lambdas, t_3 = solve_intrinsic(x, X, p)
    p[2, 2] = t_3
    t = p[:, 2]
    R = np.c_[p[:, :2], np.cross(p[:, 0], p[:, 1])]
    return Projector(R=R, t=t, lambdas=lambdas, camera=camera)
