from dataclasses import dataclass, field
from functools import partial
import numpy as np
from tqdm.contrib.concurrent import process_map
from calibration.benchmark.benchmark_result import BenchmarkResult, calc_error

from calibration.benchmark.features import Features
from calibration.data.babelcalib.entry import Entry
from calibration.feature_refiner.classifier import prune_corners
from calibration.projector.projector import Projector


@dataclass
class RefinedResult:
    input: Entry
    features: Features
    refined_features: Features
    responses: np.ndarray
    # Mask:
    # 0 - unchanged
    # 1 - filtered out
    # 2 - new corner
    # 3 - out of image
    new_board_mask: np.ndarray
    prediction: Projector
    error: float


# https://stackoverflow.com/a/45313353/ @Divakar
def view1D(a, b):  # a, b are arrays
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    return a.view(void_dt).ravel(), b.view(void_dt).ravel()


def refine_features_single(
    r: BenchmarkResult, solver_name: str="Optimization", pan_size: int = 1,
    thr=0.0019
) -> RefinedResult | None:
    assert isinstance(r.input, Entry) and r.input.image is not None
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

    A, B = view1D(new_board, board)

    mask = np.isin(A, B, invert=True)

    proj = r.predictions[solver_name]
    try:
        new_corners = proj.project(new_board)
        # new_corners = proj.project(r.features.board)
    except ValueError:
        return None
    # import plotly.express as px
    # px.scatter(x=new_corners[:, 0],
    #                y=new_corners[:, 1]).show()
    # px.scatter(x=r.features.corners[:, 0],
    #            y=r.features.corners[:, 1]).show()
    # print(new_corners.shape)
    # print(board.shape)
    # print(new_board.shape)
    responses, new_mask = prune_corners(new_corners, mask, r.input.image, thr)
    # print(new_corners.shape)
    return RefinedResult(
        r.input,
        r.features,
        Features(new_board, new_corners),
        responses,
        new_mask,
        proj,
        r.errors[solver_name],
    )


def refine_features(
    results: list[BenchmarkResult], solver_name: str = "Optimization", pan_size: int = 1
) -> list[RefinedResult | None]:
    return process_map(
        partial(refine_features_single, solver_name=solver_name, pan_size=pan_size),
        results,
        chunksize=10,
        leave=False,
        desc="Refining features",
    )
