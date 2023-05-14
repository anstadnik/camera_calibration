import os
from icecream import ic
from dataclasses import dataclass

import numpy as np


@dataclass
class Board:
    ofs: int  # Offset of the board in the list of boards
    id: int  # Board ID
    rows: int  # Number of rows in the board
    cols: int  # Number of columns in the board
    type: int  # Board type: 1 for triangular, 0 for rectangular
    tsize: float  # Feature size in mm
    tfam: str  # Board tag family
    tborder: float  # Tag border size
    tag_locations: list[int]  # list of tag locations on the board
    tags: list[int]  # list of tag IDs on the board
    nodes: tuple[list[int], list[int]]  # Positions on the board (row, col)
    pts: list[
        tuple[float, float, float]
    ]  # Designed positions of points in mm (x, y, z)
    Rt: np.ndarray | None = None  # Rotation and translation matrix for the board


# class TargetPlane:
#
#     def __init__(self):
#         self.target_id_ = ""


def load_from_dsc_file_tp_file(dsc_file, tp_file) -> list[Board]:
    boards = []
    target_id_ = os.path.splitext(os.path.basename(dsc_file))[0]

    boards = []
    ofs = 0

    with open(dsc_file) as fid:
        while True:
            h1 = fid.readline().strip()
            if len(h1) > 0:
                h1 = list(map(float, h1.split(",")))
                h2 = fid.readline().split(",")

                board = Board(
                    ofs=ofs,
                    id=int(h1[0]),
                    rows=int(h1[1]) - 1,
                    cols=int(h1[2]) - 1 if h1[2] == 0 else int(h1[2]) - 1,
                    type=1 if h1[2] == 0 else 0,
                    tsize=h1[3],
                    tfam=h2[0],
                    tborder=float(h2[1]),
                    tag_locations=[],
                    tags=[],
                    nodes=([], []),
                    pts=[],
                )

                nodes = (
                    board.rows * (board.cols - 1) // 2
                    if board.type == 1
                    else board.rows * board.cols
                )

                t = []
                while nodes:
                    line = fid.readline().strip()
                    if line:
                        nodes -= 1
                        t.append(list(map(float, line.strip().split(","))))

                board.tag_locations = [i for i, x in enumerate(t) if x[0] >= 0]
                board.tags = [x[0] for x in t if x[0] >= 0]
                board.nodes = ([x[1] for x in t], [x[2] for x in t])
                board.pts = [(x[3] * 10.0, x[4] * 10.0, x[5] * 10.0) for x in t]

                boards.append(board)
                ofs += nodes
            else:
                break

    with open(tp_file) as fid2:
        # fid2.readline()
        for _ in range(len(boards)):
            bid = int(fid2.readline())

            if not bid:
                break

            Rt = np.array([list(map(float, fid2.readline().split())) for _ in range(3)])

            for b in range(len(boards)):
                if boards[b].id == bid:
                    boards[b].Rt = Rt
                    break
    return boards
