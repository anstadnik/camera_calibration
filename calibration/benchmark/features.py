import contextlib
from dataclasses import dataclass

import numpy as np
from cbdetect_py import CornerType, Params, boards_from_corners, find_corners
from tqdm.contrib.concurrent import process_map

from calibration.data.babelcalib.babelcalib import Dataset
from calibration.data.babelcalib.entry import Entry
from calibration.data.babelcalib.target import BoardType
from calibration.projector.projector import Projector


@dataclass
class Features:
    board: np.ndarray
    corners: np.ndarray


# TODO: Add noise
def _process_projector(
    projector_board: tuple[Projector, np.ndarray]
) -> tuple[Features | None, Projector]:
    p, board = projector_board
    with contextlib.suppress(ValueError):
        corners = p.project(board)
        out_of_img = ((corners < 0) | (corners > p.camera.resolution)).any(axis=1)
        board_ = board[~out_of_img]
        corners = corners[~out_of_img]

        if corners.size != 0 and not np.isinf(corners).any():
            return Features(board_, corners), p

    return None, p


def _get_Projector(kwargs: dict) -> Projector:
    return Projector(**kwargs)


def simul_features(
    n: int, board: np.ndarray, kwargs: dict
) -> list[tuple[Features | None, Projector]]:
    projectors = process_map(
        _get_Projector,
        (kwargs for _ in range(n)),
        chunksize=100,
        leave=False,
        total=n,
        desc="Generating projectors",
    )
    return process_map(
        _process_projector,
        [(p, board) for p in projectors],
        chunksize=100,
        leave=False,
        desc="Simulating",
    )


def _process_ds(ds: Dataset) -> list[Features | None]:
    assert ds.targets[0].type == BoardType.RECTANGULAR
    features = []
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
        # if (board.max(axis=0) == 0).any():
        #     breakpoint()
        # board /= board.max(axis=0)
        corners = np.array(corners.p)[best_board[best_board >= 0]]

        features.append(None if np.isinf(corners).any() else Features(board, corners))
    return features


# TODO: Add ds, subds and image index name
def babelcalib_features(datasets: list[Dataset]) -> list[tuple[Features | None, Entry]]:
    results = process_map(_process_ds, datasets, leave=False, desc="Process dataset")
    results = [r for res in results for r in res]
    # res = map(_process_entry, (*ds.train, *ds.test))
    # res = process_map(
    #     _process_entry,
    #     iter((*ds.train, *ds.test)),
    #     chunksize=10,
    #     leave=False,
    #     desc="Searching corners",
    # )
    # results.extend(res)

    entries = (e for ds in datasets for subds in (ds.train, ds.test) for e in subds)
    return list(zip(results, entries))

    # Skip entries with no board
    # return [
    #     (res, entry) for res, entry in zip(results, entries) if res.board is not None
    # ]
