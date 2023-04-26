import numpy as np
import plotly.express as px


def gen_checkerboard_grid(h: int, w: int) -> np.ndarray:
    """Generate a checkerboard pattern

    Args:
        h: height
        w: width

    Returns:
        np.ndarray: Array of pairs of [y, x]
    """
    return np.array([[r, c] for r in range(h) for c in range(w)])


def gen_charuco_grid(h: int, w: int, s1: float, s2: float | None = None) -> np.ndarray:
    """Generate a charruco pattern

    Args:
        h: height
        w: width
        s1: step size for y
        s2: step size for x (default: s1)

    Returns:
        np.ndarray: Array of pairs of [y, x]
    """
    s2 = s2 if s2 is not None else s1
    y = [j / 2 if j % 2 == 0 else (j - 1) / 2 + s1 for j in range(h)]
    x = [i / 2 if i % 2 == 0 else (i - 1) / 2 + s2 for i in range(w)]
    return np.array([[y[i], x[j]] for i in range(h) for j in range(w)])


def draw_board(board: np.ndarray):
    px.scatter(y=board[:, 0], x=board[:, 1]).show()


# print(gen_checkerboard_grid(3, 5))
