from .extrinsics import solve_extrinsic
from .intrinsics import solve_intrinsic
import numpy as np

def solve(x: np.ndarray, X: np.ndarray) -> np.ndarray:
    H = solve_extrinsic(x, X)
    return solve_intrinsic(x, X, H)
