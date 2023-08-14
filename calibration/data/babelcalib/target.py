import enum
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


class BoardType(enum.Enum):
    RECTANGULAR = 0
    TRIANGULAR = 1


@dataclass
class Target:
    ofs: int  # Offset of the board in the list of boards
    id: int  # Board ID
    rows: int  # Number of rows in the board
    cols: int  # Number of columns in the board
    type: BoardType  # Board type
    tsize: float  # Feature size in mm
    tfam: str  # Board tag family
    tborder: float  # Tag border size
    tag_locations: list[int]  # list of tag locations on the board
    tags: list[int]  # list of tag IDs on the board
    nodes: tuple[list[int], list[int]]  # Positions on the board (row, col)
    pts: list[
        tuple[float, float, float]
    ]  # Designed positions of points in mm (x, y, z)
    Rt: NDArray[
        np.float64
    ] | None = None  # Rotation and translation matrix for the board


def _load_tp_file(tp_file: str, targets: list[Target]) -> list[Target]:
    with open(tp_file) as fid2:
        for _ in targets:
            line = fid2.readline()
            if line.startswith("#"):
                continue
            bid = int(line)

            if not bid:
                break

            Rt = np.array([list(map(float, fid2.readline().split())) for _ in range(3)])

            for target__ in targets:
                if target__.id == bid:
                    target__.Rt = Rt
                    break

    return targets


def _load_ds_file(lines: Iterator[str]) -> list[Target]:
    targets = []
    ofs = 0

    while True:
        try:
            h1 = next(lines).strip()
        except StopIteration:
            break
        if h1 == "":
            continue
        h1 = list(map(float, h1.split(",")))
        h2 = next(lines).split(",")

        target = Target(
            ofs=ofs,
            id=int(h1[0]),
            rows=int(h1[1]) - 1,
            cols=int(h1[2]) - 1 if h1[2] != 0 else int(h1[1]) - 1,
            type=BoardType(int(h1[2] == 0)),
            tsize=h1[3],
            tfam=h2[0],
            tborder=float(h2[1]),
            tag_locations=[],
            tags=[],
            nodes=([], []),
            pts=[],
        )

        nodes = (
            target.rows * (target.cols - 1) // 2
            if target.type == BoardType.TRIANGULAR
            else target.rows * target.cols
        )

        t = []
        while nodes:
            if line := next(lines).strip():
                nodes -= 1
                t.append(list(map(float, line.strip().split(","))))

        target.tag_locations = [i for i, x in enumerate(t) if x[0] >= 0]
        target.tags = [x[0] for x in t if x[0] >= 0]
        target.nodes = ([x[1] for x in t], [x[2] for x in t])
        target.pts = [(x[3] * 10.0, x[4] * 10.0, x[5] * 10.0) for x in t]

        targets.append(target)
        ofs += nodes

    return targets


def load_from_dsc_file_tp_file(dsc_file, tp_file) -> list[Target]:
    # target_id_ = os.path.splitext(os.path.basename(dsc_file))[0]

    with open(dsc_file) as fid:
        lines = iter(fid.readlines())

    targets = _load_ds_file(lines)
    targets = _load_tp_file(tp_file, targets)

    return targets
