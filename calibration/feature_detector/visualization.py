import numpy as np
from numpy.typing import NDArray
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def show_corners(
    img: NDArray[np.float64],
    corners: NDArray[np.float64],
    scores: NDArray[np.float64] | None = None,
) -> go.Figure:
    fig1 = px.imshow(img, binary_string=True)
    fig2 = px.scatter(x=corners[:, 0], y=corners[:, 1], color=scores)
    assert isinstance(fig1.data, tuple) and isinstance(fig2.data, tuple)
    assert len(fig1.data) == 1 and len(fig2.data) == 1
    return go.Figure(data=[fig1.data[0], fig2.data[0]])


# TODO: Pass lists of NDArray[np.float64] instead of Corner and Board
def show_boards(
    img: NDArray[np.float64], corners: NDArray[np.float64], board: NDArray[np.float64]
) -> go.Figure:
    fig1 = px.imshow(img, binary_string=True)

    points = [
        {"n": i, "i": feature[0], "j": feature[1], "x": corner[0], "y": corner[1]}
        for i, (corner, feature) in enumerate(zip(corners, board))
    ]
    df = pd.DataFrame(points)
    fig2 = px.scatter(df, x="x", y="y", color="n", hover_data=["i", "j"])
    return go.Figure(data=[fig1.data[0], fig2.data[0]])
