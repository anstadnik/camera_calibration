from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.linalg import qr

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
        distortion_center (np.ndarray):
            A 3D vector representing the distortion center
            (default: center of the image).
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
