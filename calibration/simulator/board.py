import numpy as np
import plotly.express as px


def gen_checkerboard_grid(h: int, w: int) -> np.ndarray:
    """Generate a checkerboard pattern

    Args:
        h: height
        w: width

    Returns:
        np.ndarray: Array of pairs of [x, y]
    """
    return np.array([[x, y] for y in range(h) for x in range(w)])


def gen_charuco_grid(h: int, w: int, s1: float, s2: float | None = None) -> np.ndarray:
    """Generate a charruco pattern

    Args:
        h: height
        w: width
        s1: step size for x
        s2: step size for y (default: s1)

    Returns:
        np.ndarray: Array of pairs of [x, y]
    """
    s2 = s2 if s2 is not None else s1
    x = [i / 2 if i % 2 == 0 else (i - 1) / 2 + s1 for i in range(w)]
    y = [j / 2 if j % 2 == 0 else (j - 1) / 2 + s2 for j in range(h)]
    return np.array([[x[j], y[i]] for i in range(h) for j in range(w)])


def draw_board(
    board: np.ndarray, title: str | None = None, max_xy: np.ndarray | None = None
):
    range_x = None if max_xy is None else (0, max_xy[0])
    range_y = None if max_xy is None else (0, max_xy[1])
    return px.scatter(
        x=board[:, 0], y=board[:, 1], title=title, range_x=range_x, range_y=range_y
    )
