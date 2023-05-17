import contextlib
from dataclasses import dataclass

import numpy as np
from cbdetect_py import CornerType, Params, boards_from_corners, find_corners
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from calibration.data.babelcalib.babelcalib import Dataset
from calibration.data.babelcalib.entry import Entry
from calibration.data.babelcalib.target import BoardType
from calibration.projector.projector import Projector


@dataclass
class Features:
    board: np.ndarray
    corners: np.ndarray


# # TODO: Add noise
# def gen_simul_features(n: int, board: np.ndarray) -> list[tuple[Projector, Features]]:
#     projectors = [Projector() for _ in range(n)]
#     ret = []
#     for p in tqdm(projectors):
#         with contextlib.suppress(ValueError):
#             corners = p.project(board)
#             out_of_img = ((corners < 0) | (corners > p.camera.resolution)).any(axis=1)
#             board_ = board[~out_of_img]
#             corners = corners[~out_of_img]
#             ret.append((p, Features(board_, corners)))
#     return ret


def _process_projector(
    projector_board: tuple[Projector, np.ndarray]
) -> tuple[Features, Projector] | None:
    p, board = projector_board
    with contextlib.suppress(ValueError):
        corners = p.project(board)
        out_of_img = ((corners < 0) | (corners > p.camera.resolution)).any(axis=1)
        board_ = board[~out_of_img]
        corners = corners[~out_of_img]
        return Features(board_, corners), p
    return None


def _get_Projector(_) -> Projector:
    return Projector()


def simul_features(n: int, board: np.ndarray) -> list[tuple[Features, Projector]]:
    projectors = process_map(
        _get_Projector,
        range(n),
        chunksize=100,
        leave=False,
        desc="Generating projectors",
    )
    ret = process_map(
        _process_projector,
        [(p, board) for p in projectors],
        chunksize=100,
        leave=False,
        desc="Simulating",
    )
    return [item for item in ret if item]


def _process_entry(entry: Entry) -> Features:
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

    best_board_i = np.array([(np.array(b.idx) > 0).sum() for b in boards]).argmax()
    best_board = np.array(boards[best_board_i].idx)
    board = np.transpose(np.nonzero(best_board >= 0))
    # Swap y and x
    board = board[:, [1, 0]]
    corners = np.array(corners.p)[tuple(best_board[best_board >= 0])]

    return Features(board, corners)


# TODO: Add ds, subds and image index name
def babelcalib_features(datasets: list[Dataset]) -> list[tuple[Features, Entry]]:
    results = []
    for ds in tqdm(datasets, leave=False, desc="Process dataset"):
        for subds in tqdm([ds.train, ds.test], leave=False):
            assert ds.targets[0].type == BoardType.RECTANGULAR
            res = process_map(
                _process_entry,
                subds,
                chunksize=100,
                leave=False,
                desc="Searching corners",
            )
            results.extend(res)

    entries = (e for ds in datasets for subds in (ds.train, ds.test) for e in subds)
    return [(res, entry) for res, entry in zip(results, entries) if res.board]
