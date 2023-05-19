from dataclasses import dataclass, field
from tqdm.auto import tqdm
import numpy as np
from tqdm.contrib.concurrent import process_map

from calibration.benchmark.calib import calibrate

from calibration.benchmark.features import Features, babelcalib_features, simul_features
from calibration.data.babelcalib.babelcalib import Dataset, load_babelcalib
from calibration.data.babelcalib.entry import Entry
from calibration.projector.board import gen_checkerboard_grid
from calibration.projector.camera import Camera
from calibration.projector.projector import Projector


@dataclass
class BenchmarkResult:
    input: Entry | Projector
    features: Features | None
    prediction: Projector | None
    error: float | None = field(init=False)

    def __post_init__(self):
        self.error = self._calc_error()

    def _calc_error(self) -> float | None:
        if self.features is None or self.prediction is None:
            return None
        try:
            max_point_img_space = np.r_[self.prediction.camera.resolution, 1]
            max_point = (
                np.linalg.inv(self.prediction.camera.intrinsic_matrix)
                @ max_point_img_space
            )
            max_r = float(np.linalg.norm(max_point[:2]))
            corners_ = self.prediction.project(self.features.board, max_r * 10)
        except ValueError:
            return -1.0
        return np.sqrt(((corners_ - self.features.corners) ** 2).mean())


SIMUL_INP = tuple[Features | None, Projector]
BABELCALIB_INP = tuple[Features | None, Entry]


def _eval(arg: tuple[SIMUL_INP | BABELCALIB_INP, Projector | None]) -> BenchmarkResult:
    (feats, inp), proj_ = arg
    return BenchmarkResult(inp, feats, proj_)


def evaluate(
    inp: list[SIMUL_INP] | list[BABELCALIB_INP], projs: list[Projector | None]
) -> list[BenchmarkResult]:
    kwargs = dict(total=len(inp), leave=False, desc="Calibrating")
    return process_map(_eval, map(tuple, zip(inp, projs)), **kwargs)


def benchmark_simul(
    n=int(1e6), board=gen_checkerboard_grid(7, 9)
) -> list[BenchmarkResult]:
    feats_and_projs = simul_features(n, board)
    projs_ = calibrate([(f, p.camera) for f, p in feats_and_projs])
    return evaluate(feats_and_projs, projs_)


def get_camera_from_entry(entry: Entry) -> Camera:
    assert entry.image is not None
    resolution = np.array(entry.image.size)
    sensor_size = np.array([36, 36.0 * resolution[1] / resolution[0]])
    focal_length = 35
    return Camera(focal_length, resolution, sensor_size, skew=0)


def benchmark_babelcalib(dataset: list[Dataset] | None = None) -> list[BenchmarkResult]:
    if dataset is None:
        dataset = load_babelcalib()
    feats_and_ents = babelcalib_features(dataset)
    projs_ = calibrate([(f, get_camera_from_entry(e)) for f, e in feats_and_ents])
    return evaluate(feats_and_ents, projs_)
