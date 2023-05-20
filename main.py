import pickle as pkl

import numpy as np
import plotly.express as px
from pandas.compat import os

from calibration.benchmark.benchmark import benchmark_babelcalib, benchmark_simul
from calibration.solver.solve import solve


def is_orthogonal(A, rtol=1e-05, atol=1e-08):
    A = np.array(A)  # Convert input to a numpy array, if it isn't already
    return np.allclose(A.dot(A.T), np.eye(A.shape[0]), rtol, atol) and np.allclose(
        A.T.dot(A), np.eye(A.shape[0]), rtol, atol
    )


def test_solver():
    with open("WTF.pkl", "rb") as f:
        proj, board = pkl.load(f)

    x = proj.project(board)
    # px.scatter(
    #     x=x[:, 0],
    #     y=x[:, 1],
    #     # color=range(x.shape[0]),
    #     range_x=[0, proj.camera.resolution[0]],
    #     range_y=[0, proj.camera.resolution[1]],
    # ).show()
    # px.scatter(x=board[:, 0], y=board[:, 1], color = range(board.shape[0])).show()

    proj_ = solve(x, board, proj.camera)
    # print(proj.R)
    # print(proj_.R)
    # print(abs(proj.R) - abs(proj_.R))
    # print(proj.t, proj_.t)
    # print(proj_.lambdas, proj.lambdas)
    # print(proj_.t - proj.t)
    # print(proj_.lambdas - proj.lambdas)
    board_ = proj_.backproject(x)
    np.testing.assert_allclose(board, board_, atol=1e-10, rtol=1e-6)

    return
    # x_ = proj_.project(board, 1000)
    x_ = proj_.project(board)

    px.scatter(x=x_[:, 0], y=x_[:, 1], color=range(x_.shape[0])).show()

    assert is_orthogonal(proj.R)
    assert is_orthogonal(proj_.R)

    np.testing.assert_allclose(proj.t, proj_.t, atol=1e-10, rtol=0.1)
    np.testing.assert_allclose(proj.R, proj_.R, atol=1e-10, rtol=0.1)
    np.testing.assert_allclose(proj.lambdas, proj_.lambdas, atol=1e-10, rtol=0.1)

    np.testing.assert_allclose(x_, x, atol=1e-10, rtol=1e-6)


if __name__ == "__main__":
    test_solver()
    # if not os.path.isfile("babelcalib_results.pkl"):
    #     babelcalib_results = benchmark_babelcalib()
    #     with open("babelcalib_results.pkl", "wb") as f:
    #         pkl.dump(babelcalib_results, f)
    # if not os.path.isfile("simul_results.pkl"):
    #     simul_results = benchmark_simul(int(1e5))
    #     with open("simul_results.pkl", "wb") as f:
    #         pkl.dump(simul_results, f)
