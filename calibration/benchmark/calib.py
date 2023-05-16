from calibration.projector.projector import Projector
from calibration.solver.solve import solve
from .features import Features


def calibrate(features: Features) -> Projector:
    camera = Projector().camera
    return solve(features.corners, features.board, camera.intrinsic_matrix)
