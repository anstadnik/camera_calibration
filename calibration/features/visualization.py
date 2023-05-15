import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from cbdetect_py import Corner


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
    fig = go.Figure(data=[fig1.data[0] , fig2.data[0]])
    return fig
