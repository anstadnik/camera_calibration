from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import qr
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from .camera import Camera


def _gen_lambdas() -> NDArray[np.float64]:
    l1 = np.random.uniform(-5.0, 1.0)
    l2 = np.random.uniform(
        -2.61752136752137 * l1 - 6.85141810943093,
        -2.61752136752137 * l1 - 4.39190876941320,
    )
    return np.array([l1, l2])


@dataclass
class Projector:
    """
    A data class to store the simulation parameters for the projection of 3D points
    onto a 2D image plane.

    Attributes:
        R (NDArray[np.float64]):
            A 3x3 orthogonal matrix representing the projection rotation
            matrix (default: random).
        t (NDArray[np.float64]):
            A 3D vector representing the projection translation vector
            (default: random).
        lambdas (NDArray[np.float64]):
            A 1D array of radial distortion coefficients (default: random).
        camera (Camera):
            A Camera object representing the camera intrinsics (default: Camera()).
        distortion_center (NDArray[np.float64]):
            A 3D vector representing the distortion center
            (default: center of the image).
    """

    R: NDArray[np.float64] = field(default_factory=lambda: qr(np.random.randn(3, 3))[0])
    t: NDArray[np.float64] = field(
        default_factory=lambda: np.random.uniform([-1.0, -0.7, 2.5], [0.0, -0.3, 4.0])
    )
    lambdas: NDArray[np.float64] = field(default_factory=_gen_lambdas)
    camera: Camera = field(default_factory=Camera)

    @property
    def P(self) -> NDArray[np.float64]:
        return np.c_[self.R[:, :2], self.t]

    def psi(self, r):
        l1, l2 = self.lambdas
        return 1 + l1 * r**2 + l2 * r**4

    def project(
        self, X: NDArray[np.float64], max_r: float | None = None
    ) -> NDArray[np.float64]:
        """
        Simulates the projection of board points (X) onto a image plane,
        given the simulation parameters (p).

        Args:

        X (NDArray[np.float64]):
            An array of point in the board space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].

        Returns:
            x (NDArray[np.float64]):
                An array of point in the image space, with shape (n, 2),
                where n is the number of points and each point is represented as [x, y].
        """
        # Extrinsics
        X_h = np.c_[X, np.ones(X.shape[0])]
        # x = (np.linalg.inv(self.P) @ X_h.T).T
        x = (self.P @ X_h.T).T
        x /= x[:, 2][:, None]

        # Distortion
        idx = np.linalg.norm(x[:, :2], axis=1) > np.finfo(float).eps

        def f(r, x):
            # x[2] == 1
            return self.psi(r) * np.linalg.norm(x[:2]) - r

        if max_r is None:
            max_point_img_space = np.r_[self.camera.resolution, 1]
            max_point = (
                np.linalg.inv(self.camera.intrinsic_matrix) @ max_point_img_space
            )
            max_r = float(np.linalg.norm(max_point[:2])) * 1.1
        rs = np.array(
            [root_scalar(f, args=(xi,), bracket=(0, max_r)).root for xi in x[idx]]
        )

        x[idx] *= (rs / np.linalg.norm(x[idx, :2], axis=1))[:, np.newaxis]
        np.testing.assert_almost_equal(
            self.psi(np.linalg.norm(x[idx, :2], axis=1)), x[idx, 2], decimal=5
        )
        x[:, 2] = 1

        # Intrinsics
        x = (self.camera.intrinsic_matrix @ x.T).T
        # Pyright bug
        x /= x[:, 2][:, None]
        return x[:, :2]

    def backproject(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Simulates the backprojection of image points (x) onto a board plane,
        given the simulation parameters (p).

        Args:

        x (NDArray[np.float64]):
            An array of point in the image space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].

        Returns:
            X (NDArray[np.float64]):
                An array of point in the board space, with shape (n, 2),
                where n is the number of points and each point is represented as [x, y].
        """
        # Intrinsics
        x = np.c_[x, np.ones(x.shape[0])]
        x = (np.linalg.inv(self.camera.intrinsic_matrix) @ x.T).T
        # Pyright bug
        x /= x[:, 2][:, None]

        # Distortion
        x[:, 2] = self.psi(np.linalg.norm(x[:, :2], axis=1))
        x /= x[:, 2][:, None]

        # Extrinsics
        x = (np.linalg.inv(self.P) @ x.T).T
        x /= x[:, 2][:, None]
        return x[:, :2]
