import pandas as pd
import plotly.express as px

from .babelcalib import Entry
from plotly.graph_objs import Figure


def visualize(calibration_data: Entry) -> Figure:
    df = pd.DataFrame(
        [[corner.x, corner.y] for corner in calibration_data.corners],
        columns=["x", "y"],
    )
    fig = px.scatter(
        df, x="x", y="y", width=calibration_data.width, height=calibration_data.height
    )
    fig.update_traces(
        marker=dict(size=10, color="red", line=dict(width=2, color="DarkSlateGrey"))
    )
    fig.update_layout(
        images=[
            dict(
                source=calibration_data.image,
                xref="x",
                yref="y",
                x=0,
                y=calibration_data.height,
                sizex=calibration_data.width,
                sizey=calibration_data.height,
                sizing="stretch",
                opacity=1,
                layer="below",
            )
        ]
    )
    return fig
