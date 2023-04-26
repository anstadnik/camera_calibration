from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.linalg import qr
from scipy.optimize import root_scalar

from .camera import Camera


@dataclass
class SimulParams:
    """
    A data class to store the simulation parameters for the projection of 3D points
    onto a 2D image plane.

    Attributes:
        R (np.ndarray):
            A 3x3 orthogonal matrix representing the rotation matrix (default: random).
        t (np.ndarray):
            A 3D vector representing the translation vector (default: random).
        lambdas (np.ndarray):
            A 1D array of radial distortion coefficients (default: random).
        camera (Camera):
            A Camera object representing the camera intrinsics (default: Camera()).
        distortion_center (np.ndarray | None):
            A 3D vector representing the distortion center
            (default: None, calculated as the mean of input points X).
    """

    R: np.ndarray = field(default_factory=lambda: qr(np.random.randn(3, 3))[0])
    t: np.ndarray = field(
        default_factory=lambda: np.random.uniform([-5.0, -5, 5.0], [5.0, 5, 15.0])
    )
    lambdas: np.ndarray = field(
        default_factory=lambda: np.random.uniform([-1.9, -1.9], [-0.1, -0.1])
    )
    camera: Camera = field(default_factory=Camera)
    distortion_center: np.ndarray | None = None


class SimulOut(NamedTuple):
    """
    A named tuple to store the output of the simul_projection function.

    Attributes:
        X_board (np.ndarray): The points in the board space (n, 2)
            where n is the number of points and each point is represented as [y, x].
        X_image (np.ndarray): The points in the image space (n, 2)
            where n is the number of points and each point is represented as [y, x].
        lambdas (np.ndarray): The radial distortion coefficients.
        R (np.ndarray): The rotation matrix.
        t (np.ndarray): The translation vector.
    """

    X_board: np.ndarray
    X_image: np.ndarray
    lambdas: np.ndarray
    R: np.ndarray
    t: np.ndarray


EPS = np.finfo(np.float64).eps


def simul_projection(X: np.ndarray, p: SimulParams | None = None) -> SimulOut:
    """
    Simulates the projection of board points (X) onto a image plane,
    given the simulation parameters (p).

    Args:

    X (np.ndarray):
        An array of point in the board space, with shape (n, 2),
        where n is the number of points and each point is represented as [x, y].
    p (SimulParams | None):
        A SimulParams object containing the simulation parameters. If None,
        default values will be used (default: None).

    Returns:
        SimulOut: A SimulOut object containing the projected points
            and other relevant information.
    """

    if p is None:
        p = SimulParams()

    n = X.shape[0]
    X_h = np.c_[X, np.ones(n)]

    P = np.c_[p.R[:, :2], p.t]

    x = (P @ X_h.T).T
    x /= x[:, 2][:, None] + EPS

    if p.distortion_center is None:
        p.distortion_center = x[:, :2].mean(axis=0)

    x[:, :2] -= p.distortion_center

    r = np.linalg.norm(x[:, :2], axis=1) + EPS

    def psi(r):
        assert p is not None
        return 1 + sum(ld * r ** (2 * i) for i, ld in enumerate(p.lambdas, 1))

    def f(r, x):
        return np.linalg.norm(x[:2]) * psi(r) - x[2] * r

    r_hat = np.array([root_scalar(f, args=(xi,), bracket=(0, 1000)).root for xi in x])

    x *= (r_hat / r)[:, np.newaxis]

    x[:, 2] = 1
    x = (p.camera.intrinsic_matrix @ x.T).T
    x /= x[:, 2][:, None]
    x = x[:, :2]
    x += p.distortion_center

    return SimulOut(X, x, p.lambdas, p.R, p.t)
