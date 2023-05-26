from dataclasses import dataclass, field
from functools import partial
import numpy as np
from tqdm.contrib.concurrent import process_map
from calibration.benchmark.benchmark_result import BenchmarkResult, calc_error

from calibration.benchmark.features import Features
from calibration.data.babelcalib.entry import Entry
from calibration.projector.projector import Projector


@dataclass
class RefinedResult:
    input: Entry
    features: Features
    refined_features: Features
    new_board_mask: np.ndarray
    prediction: Projector
    error: float = field(init=False)

    def __post_init__(self):
        self.error = calc_error(self.prediction, self.refined_features)


def refine_features_single(
    r: BenchmarkResult, solver_name: str, pan_size: int = 1
) -> RefinedResult:
    assert isinstance(r.input, Entry)
    board = r.features.board.astype(int)
    x_min, x_max, y_min, y_max = (
        board[:, 0].min() - pan_size,
        board[:, 0].max() + pan_size,
        board[:, 1].min() - pan_size,
        board[:, 1].max() + pan_size,
    )
    new_board = np.array(
        [[x, y] for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]
    )

    # Crazy, innit?
    mask = np.isin(
        new_board.view([("", new_board.dtype)] * new_board.shape[1]),
        board.view([("", board.dtype)] * board.shape[1]),
    )

    proj = r.predictions[solver_name]
    new_corners = proj.project(new_board)
    return RefinedResult(
        r.input, r.features, Features(new_board, new_corners), mask, proj
    )


def refine_features(
    results: list[BenchmarkResult], solver_name: str, pan_size: int = 1
) -> list[RefinedResult]:
    return process_map(
        partial(refine_features_single, solver_name=solver_name, pan_size=pan_size),
        results,
        chunksize=100,
        leave=False,
        desc="Refining features",
    )
