import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def show_refined_corners(
    img: np.ndarray,
    corners: np.ndarray,
    board: np.ndarray,
    mask: np.ndarray,
    responses: np.ndarray,
) -> go.Figure:
    fig1 = px.imshow(img, binary_string=True)

    mask_mapping = {
        0: "unchanged",
        1: "filtered out",
        2: "new corner",
        3: "out of image",
    }

    # mask_mapping = {
    #     0: "Original feature",
    #     1: "Possible new feature",
    #     2: "Possible new feature",
    #     3: "Possible new feature",
    # }

    responses_all = np.full_like(mask, -1)
    responses_all[(mask > 0) & (mask < 3)] = responses
    points = [
        {
            "n": i,
            "i": feature[0],
            "j": feature[1],
            "x": corner[0],
            "y": corner[1],
            "mask": mask_mapping[mask_v],
            "resp": resp,
        }
        for i, (corner, feature, mask_v, resp) in enumerate(
            zip(corners, board, mask, responses_all)
        )
    ]
    df = pd.DataFrame(points)
    colors = [
        "#1f77b4",  ## muted blue
        "#ff7f0e",  ## safety orange
        "#2ca02c",  ## cooked asparagus green
        "#d62728",  ## brick red
        "#9467bd",  ## muted purple
        "#8c564b",  ## chestnut brown
        "#e377c2",  ## raspberry yogurt pink
        "#7f7f7f",  ## middle gray
        "#bcbd22",  ## curry yellow-green
        "#17becf",  ## blue-teal
    ]

    fig2 = px.scatter(
        # df, x="x", y="y", color="resp", symbol="mask", hover_data=["i", "j"]
        df,
        x="x",
        y="y",
        color="mask",
        hover_data=["i", "j", "resp"],
        # df, x="x", y="y", hover_data=["i", "j", "resp"]
        color_discrete_sequence=colors,
        category_orders={"mask": list(mask_mapping.values())},
    ).update_traces(marker=dict(size=10))
    assert isinstance(fig1.data, tuple) and isinstance(fig2.data, tuple)
    return go.Figure(data=fig1.data + fig2.data)
