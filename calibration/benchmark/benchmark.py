from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from calibration.benchmark.calib import calibrate

from calibration.benchmark.features import (
    BABELCALIB_INP,
    SIMUL_INP,
    Features,
    babelcalib_features,
    simul_features,
)
from calibration.data.babelcalib.babelcalib import Dataset, load_babelcalib
from calibration.data.babelcalib.entry import Entry
from calibration.projector.board import gen_checkerboard_grid
from calibration.projector.camera import Camera
from calibration.projector.projector import Projector
from calibration.solver.optimization.solve import solve as solve_optimization
from calibration.solver.scaramuzza.solve import solve as solve_scaramuzza


@dataclass
class BenchmarkResult:
    input: Entry | Projector
    features: Features | None
    predictions: dict[str, Projector]
    errors: dict[str, float] = field(init=False)

    def __post_init__(self):
        self.errors = {k: self._calc_error(v) for k, v in self.predictions.items()}

    def _calc_error(self, proj: Projector) -> float:
        assert self.features is not None
        try:
            max_point_img_space = np.r_[proj.camera.resolution, 1]
            max_point = (
                np.linalg.inv(proj.camera.intrinsic_matrix) @ max_point_img_space
            )
            max_r = float(np.linalg.norm(max_point[:2]))
            corners_ = proj.project(self.features.board, max_r)
        except ValueError:
            return -1.0
        return np.sqrt(((corners_ - self.features.corners) ** 2).mean())


def results_into_df(results: list[BenchmarkResult]) -> pd.DataFrame:
    return pd.json_normalize(
        [{k: v for k, v in asdict(r).items() if v is not None} for r in results]
    )


def evaluate(
    inp: list[SIMUL_INP] | list[BABELCALIB_INP], projs: list[dict[str, Projector]]
) -> list[BenchmarkResult]:
    return process_map(
        BenchmarkResult,
        *zip(*inp),
        projs,
        total=len(inp),
        chunksize=100,
        leave=False,
        desc="Evaluating",
    )


_solvers = [("Scaramuzza", solve_scaramuzza), ("Optimization", solve_optimization)]


def benchmark_simul(
    n=int(1e5), board=gen_checkerboard_grid(7, 9), kwargs: dict | None = None
) -> list[BenchmarkResult]:
    projs_and_feats = simul_features(n, board, kwargs or {})
    solvers_args = [(f, p.camera) if f else None for p, f in projs_and_feats]
    projs = calibrate(_solvers, solvers_args)
    assert len(projs) == len(projs_and_feats)
    return evaluate(projs_and_feats, projs)


def get_camera_from_entry(entry: Entry) -> Camera:
    assert entry.image is not None
    resolution = np.array(entry.image.size)
    sensor_size = np.array([36, 36.0 * resolution[1] / resolution[0]])
    focal_length = 35.0
    return Camera(
        focal_length=focal_length, resolution=resolution, sensor_size=sensor_size
    )


def benchmark_babelcalib(dataset: list[Dataset] | None = None) -> list[BenchmarkResult]:
    if dataset is None:
        dataset = load_babelcalib()
        # Skip aprilgrid
        aprilgrid_datasets = ["UZH", "Kalibr"]
        dataset = [
            ds for ds in dataset if not any(map(ds.name.startswith, aprilgrid_datasets))
        ]
    ents_and_feats = babelcalib_features(dataset)
    solvers_args = [
        (f, get_camera_from_entry(e)) if f else None for e, f in ents_and_feats
    ]
    projs = calibrate(_solvers, solvers_args)
    assert len(projs) == len(ents_and_feats)
    return evaluate(ents_and_feats, projs)
