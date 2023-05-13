from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import qr
from scipy.optimize import root_scalar

from .camera import Camera


def _gen_lambdas() -> np.ndarray:
    l1 = np.random.uniform(-1.5, 1.5)
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
        R (np.ndarray):
            A 3x3 orthogonal matrix representing the backprojection rotation
            matrix (default: random).
        t (np.ndarray):
            A 3D vector representing the backprojection translation vector
            (default: random).
        # R_inv (np.ndarray):
        #     A 3x3 orthogonal matrix representing the projection rotation matrix
        #     (inv(R)).
        # t_inv (np.ndarray):
        #     A 3D vector representing the projection translation vector (-t).
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
        # default_factory=lambda: np.random.uniform([2.0, 2.0, -100.0], [6.0, 4.0, -20.0])
        default_factory=lambda: np.random.uniform(
            # [-6.0, -4.0, 20.0], [-2.0, -2.0, 100.0]
            # [-6.0, -4.0, 2.0], [-2.0, -2.0, 10.0]
            [-0.3, -0.3, 1.5],
            [0.3, 0.3, 3.0],
        )
    )
    lambdas: np.ndarray = field(default_factory=_gen_lambdas)
    camera: Camera = field(default_factory=Camera)

    def project(self, X: np.ndarray) -> np.ndarray:
        """
        Simulates the projection of board points (X) onto a image plane,
        given the simulation parameters (p).

        Args:

        X (np.ndarray):
            An array of point in the board space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].

        Returns:
            SimulOut: A SimulOut object containing the projected points
                and other relevant information.
        """
        # Extrinsics
        X_h = np.c_[X, np.ones(X.shape[0])]
        P = np.c_[self.R[:, :2], self.t]
        x = (P @ X_h.T).T
        x /= x[:, 2][:, None]
        x.copy()

        # Distortion
        # distortion_center = (
        #     np.linalg.inv(self.camera.intrinsic_matrix)
        #     @ np.r_[self.camera.resolution, 1]
        # ) / 2
        # x[:, :2] -= distortion_center[:2]

        idx = np.linalg.norm(x[:, :2], axis=1) > 0

        def f(lbd, x):
            # x[2] == 1
            return self.psi(np.linalg.norm(x[:2]) / lbd) - x[2] / lbd

        max_lambda = np.linalg.norm(self.camera.resolution) / 2
        lambdas = np.array(
            [
                root_scalar(
                    f, args=(xi,), bracket=(0 + np.finfo(float).eps, max_lambda)
                ).root
                for xi in x[idx]
            ]
        )

        assert np.linalg.norm(x[:, :2], axis=1).shape == (63,)
        assert lambdas.shape == (63,)
        x[idx] /= lambdas[:, np.newaxis]
        # np.testing.assert_array_almost_equal(x[:, 1] / x[:, 0], x1[:, 1] / x1[:, 0])
        # print(np.c_[x[:, 1] / x[:, 0], x1[:, 1] / x1[:, 0]][:20])
        # np.testing.assert_almost_equal(
        #     self.psi(np.linalg.norm(x[:, :2], axis=1)), x[:, 2], decimal=5
        # )
        x[:, 2] = 1
        # x[:, :2] += distortion_center[:2]
        x.copy()
        # np.testing.assert_array_almost_equal(x2[:, 1] / x2[:, 0], x1[:, 1] / x1[:, 0])

        # Intrinsics
        x = (self.camera.intrinsic_matrix @ x.T).T
        # Pyright bug
        x /= x[:, 2][:, None]  # type: ignore
        return x[:, :2]
        # return x1[:, :2], x2[:, :2], x[:, :2]

    def backproject(self, x: np.ndarray) -> np.ndarray:
        """
        Simulates the backprojection of image points (x) onto a board plane,
        given the simulation parameters (p).

        Args:

        x (np.ndarray):
            An array of point in the image space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].

        Returns:
            SimulOut: A SimulOut object containing the projected points
                and other relevant information.
        """
        # Intrinsics
        x = np.c_[x, np.ones(x.shape[0])]
        x = (np.linalg.inv(self.camera.intrinsic_matrix) @ x.T).T
        # Pyright bug
        x /= x[:, 2][:, None]  # type: ignore

        # Distortion
        x[:, 2] = self.psi(np.linalg.norm(x[:, :2], axis=1))
        x /= x[:, 2][:, None]  # type: ignore

        # Extrinsics
        P = np.c_[self.R[:, :2], self.t]
        P_inv = np.linalg.inv(P)
        x = (P_inv @ x.T).T
        x /= x[:, 2][:, None]
        return x[:, :2]

    def psi(self, r):
        l1, l2 = self.lambdas
        return 1 + l1 * r**2 + l2 * r**4
