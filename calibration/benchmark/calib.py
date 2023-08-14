from functools import partial
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from tqdm.contrib.concurrent import process_map

from calibration.projector.camera import Camera
from calibration.projector.projector import Projector

from .features import Features


_SOLVER = Callable[[NDArray[np.float64], NDArray[np.float64], Camera], Projector | None]


def _calibrate_helper(
    arg: tuple[Features, Camera] | None, solvers: list[tuple[str, _SOLVER]]
) -> dict[str, Projector]:
    if arg is None:
        return {}
    features, camera = arg
    ret = {
        solver_name: solver(features.corners, features.board, camera)
        for solver_name, solver in solvers
    }
    return {k: v for k, v in ret.items() if v is not None}


def calibrate(
    solvers: list[tuple[str, _SOLVER]],
    feature_and_camera: list[tuple[Features, Camera] | None],
) -> list[dict[str, Projector]]:
    return process_map(
        partial(_calibrate_helper, solvers=solvers),
        feature_and_camera,
        chunksize=10,
        max_workers=8,
        leave=False,
        desc="Calibrating",
    )
