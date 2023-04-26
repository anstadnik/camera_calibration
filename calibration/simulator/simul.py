from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import qr
from scipy.optimize import root_scalar


@dataclass
class SimulParams:
    R: np.ndarray = field(default_factory=lambda: qr(np.random.randn(3, 3))[0])
    t: np.ndarray = field(
        default_factory=lambda: np.random.uniform([-5.0, -5, 5.0], [5.0, 5, 15.0])
    )
    lambdas: np.ndarray = field(
        default_factory=lambda: np.random.uniform([-1.9, -1.9], [-0.1, -0.1])
    )
    K: np.ndarray = field(
        default_factory=lambda: np.array(
            [[1166.67, 0.0, 600.0], [0.0, 1166.67, 400.0], [0.0, 0.0, 1.0]]
        )
    )
    distortion_center: np.ndarray | None = None


SimulOut = namedtuple("SimulOut", ["X_board", "X_image", "lambdas", "R", "t"])
EPS = np.finfo(np.float64).eps


def simul_projection(X: np.ndarray, p: SimulParams | None = None) -> SimulOut:
    if p is None:
        p = SimulParams()
    if p.distortion_center is None:
        p.distortion_center = X.mean(axis=0)

    n = X.shape[0]
    X_h = np.c_[X - p.distortion_center, np.ones(n)]

    P = np.c_[p.R[:, :2], p.t]

    # x = np.matmul(P, X_h.T).T
    x = (P @ X_h.T).T
    x /= x[:, 2][:, None] + EPS

    r = np.linalg.norm(x[:, :2], axis=1) + EPS

    def psi(r):
        assert p is not None
        return 1 + sum(ld * r ** (2 * i) for i, ld in enumerate(p.lambdas, 1))

    def f(r, x):
        return np.linalg.norm(x[:2]) * psi(r) - x[2] * r

    r_hat = np.array([root_scalar(f, args=(xi,), bracket=(0, 1000)).root for xi in x])

    x *= (r_hat / r)[:, np.newaxis]

    x[:, 2] = 1
    # x = np.matmul(K, x.T).T
    x = (p.K @ x.T).T
    # x = np.array([vgtk.project(xi) for xi in x])
    x /= x[:, 2][:, None]
    x = x[:, :2]

    return SimulOut(X, x, p.lambdas, p.R, p.t)
