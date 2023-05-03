import numpy as np
from scipy.optimize import root_scalar

from .types import SimulParams


def apply_extrinsics(X: np.ndarray, p: SimulParams) -> np.ndarray:
    X_h = np.c_[X, np.ones(X.shape[0])]
    P = np.c_[p.R[:, :2], p.t]
    x = (P @ X_h.T).T
    x /= x[:, 2][:, None]
    return x


def apply_distortion(x: np.ndarray, p: SimulParams) -> np.ndarray:
    # x_norm = x.max(axis=0)
    # x /= x_norm

    r = np.linalg.norm(x[:, :2], axis=1)
    idx = r > 0

    l1, l2 = p.lambdas

    def psi(r):
        # return 1 + sum(ld * r ** (2 * i) for i, ld in enumerate(p.lambdas, 1))
        return 1 + l1 * r**2 + l2 * r**4

    def f(r, x):
        return np.linalg.norm(x[:2]) * psi(r) - x[2] * r

    max_r = np.linalg.norm(p.camera.resolution / 2)
    r_hat = np.array(
        [root_scalar(f, args=(xi,), bracket=(0, max_r)).root for xi in x[idx]]
    )

    x[idx] *= (r_hat / r[idx])[:, np.newaxis]

    # x *= x_norm

    x[:, 2] = 1
    return x


def apply_intrinsics(x: np.ndarray, p: SimulParams) -> np.ndarray:
    x = (p.camera.intrinsic_matrix @ x.T).T
    # Pyright bug
    x /= x[:, 2][:, None]  # type: ignore
    return x[:, :2]
