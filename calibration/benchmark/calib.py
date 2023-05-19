from tqdm.contrib.concurrent import process_map

from calibration.projector.camera import Camera
from calibration.projector.projector import Projector
from calibration.solver.solve import solve

from .features import Features


def _calibrate(arg: tuple[Features | None, Camera]) -> Projector | None:
    features, camera = arg
    return None if features is None else solve(features.corners, features.board, camera)


def calibrate(
    feature_and_camera: list[tuple[Features | None, Camera]]
) -> list[Projector | None]:
    return process_map(
        _calibrate, feature_and_camera, chunksize=100, leave=False, desc="Calibrating"
    )
