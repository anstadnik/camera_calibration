import numpy as np

from .helpers import apply_distortion, apply_extrinsics, apply_intrinsics
from .types import SimulOut, SimulParams


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

    p = p if p is not None else SimulParams()

    x = apply_extrinsics(X, p)
    x = apply_distortion(x, p)
    x = apply_intrinsics(x, p)

    return SimulOut(X, x, p.lambdas, p.R, p.t)
