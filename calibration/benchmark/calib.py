from functools import partial
from typing import Callable
import numpy as np
from tqdm.contrib.concurrent import process_map

from calibration.projector.camera import Camera
from calibration.projector.projector import Projector

from .features import Features


_SOLVER = Callable[[np.ndarray, np.ndarray, Camera], Projector]


def _calibrate_helper(
    arg: tuple[Features | None, Camera], solvers: list[tuple[str, _SOLVER]]
) -> dict[str, Projector]:
    features, camera = arg
    if features is None:
        return {}
    return features and {
        solver_name: solver(features.corners, features.board, camera)
        for solver_name, solver in solvers
    }


def calibrate(
    solvers: list[tuple[str, _SOLVER]],
    feature_and_camera: list[tuple[Features, Camera]],
) -> list[dict[str, Projector]]:
    return process_map(
        partial(_calibrate_helper, solvers=solvers),
        feature_and_camera,
        # chunksize=1000,
        chunksize=10,
        leave=False,
        desc="Calibrating",
    )
