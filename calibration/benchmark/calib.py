from tqdm.contrib.concurrent import process_map
from calibration.projector.camera import Camera
from calibration.projector.projector import Projector
from calibration.solver.solve import solve
from .features import Features


def _calibrate(arg: tuple[Features, Camera]) -> Projector:
    features, camera = arg
    return solve(features.corners, features.board, camera.intrinsic_matrix)


def calibrate(feature_and_camera: list[tuple[Features, Camera]]) -> list[Projector]:
    return process_map(
        _calibrate, feature_and_camera, chunksize=100, leave=False, desc="Calibrating"
    )
