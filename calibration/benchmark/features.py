from dataclasses import dataclass

import numpy as np
from cbdetect_py import CornerType, Params, boards_from_corners, find_corners
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from calibration.data.babelcalib.babelcalib import load_babelcalib
from calibration.data.babelcalib.entry import Entry
from calibration.data.babelcalib.target import BoardType
from calibration.projector.board import gen_checkerboard_grid
from calibration.projector.projector import Projector


@dataclass
class Features:
    board: np.ndarray
    corners: np.ndarray


# TODO: Add noise
def gen_simul_features(
    n=int(1e6), board=gen_checkerboard_grid(7, 9)
) -> tuple[list[Projector], list[Features]]:
    projectors = [Projector() for _ in range(n)]
    features = [Features(board, p.project(board)) for p in projectors]
    return projectors, features


def process_entry(entry: Entry) -> Features:
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
def gen_babelcalib_features(
    datasets=load_babelcalib(),
) -> tuple[list[Entry], list[Features]]:
    results = []
    for ds in tqdm(datasets):
        for _, subds in zip(tqdm(["train", "test"], leave=False), [ds.train, ds.test]):
            assert ds.targets[0].type == BoardType.RECTANGULAR
            res = process_map(process_entry, subds, leave=False)
            results.extend(res)

    entries = (e for ds in datasets for subds in (ds.train, ds.test) for e in subds)
    entries = [entry for entry, res in zip(entries, results) if res.board]
    results = [res for res in results if res.board]

    return entries, results
