import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from cbdetect_py import Corner, Board


def show_corners(img: np.ndarray, corner: Corner) -> go.Figure:
    fig1 = px.imshow(img, binary_string=True)
    fig2 = px.scatter(
        x=[p[0] for p in corner.p], y=[p[1] for p in corner.p], color=corner.score
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=[p[0] for p in corner.p],
    #         y=[p[1] for p in corner.p],
    #         marker={"color": corner.score},
    #     )
    # )
    assert isinstance(fig1.data, tuple) and isinstance(fig2.data, tuple)
    assert len(fig1.data) == 1 and len(fig2.data) == 1
    return go.Figure(data=[fig1.data[0], fig2.data[0]])


def show_boards(img: np.ndarray, corner: Corner, boards: list[Board]) -> go.Figure:
    fig1 = px.imshow(img, binary_string=True)

    points = []
    for board_i, board in enumerate(boards):
        points.extend(
            {
                "Board": board_i,
                "i": i,
                "j": j,
                "x": corner.p[board.idx[i][j]][0],
                "y": corner.p[board.idx[i][j]][1],
            }
            for i, j in itertools.product(
                range(len(board.idx)), range(len(board.idx[0]))
            )
        )
    df = pd.DataFrame(points)
    fig2 = px.scatter(df, x="x", y="y", color="Board", hover_data=["i", "j"])
    return go.Figure(data=[fig1.data[0], fig2.data[0]])
