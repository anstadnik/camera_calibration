import numpy as np

from src.projector.camera import Camera
from src.projector.projector import Projector

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
    x_ = np.c_[x, np.ones(x.shape[0])]
    x_ = (np.linalg.inv(camera.intrinsic_matrix) @ x_.T).T
    x_ /= x_[:, 2][:, None]
    x_ = x_[:, :2]

    ps = solve_extrinsic(x_, X)
    assert ps
    projs = []
    for p in ps:
        lambdas, t_3 = solve_intrinsic(x_, X, p)
        t = np.r_[p[:2, 2], t_3]
        R = np.c_[p[:, :2], np.cross(p[:, 0], p[:, 1])]
        projs.append(Projector(R=R, t=t, lambdas=lambdas, camera=camera))

    reproj_errors = [
        np.linalg.norm(proj_.backproject(x) - X, axis=1).mean() for proj_ in projs
    ]

    return projs[np.argmin(reproj_errors)]
