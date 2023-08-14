import contextlib
from dataclasses import dataclass

import numpy as np
from cbdetect_py import CornerType, Params, boards_from_corners, find_corners
from numpy.typing import NDArray
from tqdm.contrib.concurrent import process_map

from calibration.data.babelcalib.babelcalib import Dataset
from calibration.data.babelcalib.entry import Entry
from calibration.data.babelcalib.target import BoardType
from calibration.projector.projector import Projector


@dataclass
class Features:
    board: NDArray[np.float64]
    corners: NDArray[np.float64]


SIMUL_INP = tuple[Projector, Features | None]
BABELCALIB_INP = tuple[Entry, Features | None]


# TODO: Add noise
def _simul_features(args: tuple[dict, NDArray[np.float64]]) -> SIMUL_INP:
    kwargs, board = args
    p = Projector(**kwargs)
    with contextlib.suppress(ValueError):
        corners = p.project(board)
        out_of_img = ((corners < 0) | (corners >= p.camera.resolution)).any(axis=1)
        board_ = board[~out_of_img]
        corners = corners[~out_of_img]

        if corners.size != 0 and not np.isinf(corners).any():
            return (p, Features(board_, corners))

    return (p, None)


def simul_features(n: int, board: NDArray[np.float64], kwargs: dict) -> list[SIMUL_INP]:
    # TODO: try partial
    args = ((kwargs, board) for _ in range(n))
    return process_map(
        _simul_features, args, chunksize=100, leave=False, total=n, desc="Simulating"
    )


def _process_ds(ds: Dataset) -> list[Features | None]:
    assert ds.targets[0].type == BoardType.RECTANGULAR
    features: list[Features | None] = []
    for entry in (*ds.train, *ds.test):
        img = np.array(entry.image)
        params = Params()
        params.show_processing = False
        params.corner_type = (
            CornerType.SaddlePoint
            # if target.type == BoardType.RECTANGULAR
            # else CornerType.MonkeySaddlePoint
        )

        corners = find_corners(img, params)
        boards = boards_from_corners(img, corners, params)

        if not boards:
            features.append(None)
            continue
        best_board_i = np.array([(np.array(b.idx) > 0).sum() for b in boards]).argmax()
        best_board = np.array(boards[best_board_i].idx)
        board = np.transpose(np.nonzero(best_board >= 0)).astype(np.float64)
        # Swap y and x
        board = board[:, [1, 0]]
        board -= board[0]
        corners = np.array(corners.p)[best_board[best_board >= 0]]

        features.append(None if np.isinf(corners).any() else Features(board, corners))
    return features


# TODO: Add ds, subds and image index name
def babelcalib_features(datasets: list[Dataset]) -> list[BABELCALIB_INP]:
    results = process_map(
        _process_ds, datasets, leave=False, desc="Find features in dataset"
    )
    results = [r for res in results for r in res]
    entries = (e for ds in datasets for subds in (ds.train, ds.test) for e in subds)
    return list(zip(entries, results))
