import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from calibration.simulator.board import gen_checkerboard_grid
from calibration.simulator.simul import simul_projection
from calibration.simulator.types import SimulParams
from calibration.solver.solve import solve


def gen_sample(_):
    params = SimulParams()
    board = gen_checkerboard_grid(7, 9)
    try:
        X, x, lambdas, R, t = simul_projection(board, params)
    except ValueError as e:
        assert str(e) == "f(a) and f(b) must have different signs"
        ret = [
            params.lambdas,
            np.full_like(params.lambdas, np.nan),
            params.R,
            np.full_like(params.R, np.nan),
            params.t,
            np.full_like(params.t, np.nan),
            np.nan,
            True,
            False,
        ]
        assert ret[-1] is not None
        return ret
    out_of_img = ((x < 0) | (x > params.camera.resolution)).any(axis=1)
    X = X[~out_of_img]
    x = x[~out_of_img]
    if len(X) < 6:
        ret = [
            params.lambdas,
            np.full_like(params.lambdas, np.nan),
            params.R,
            np.full_like(params.R, np.nan),
            params.t,
            np.full_like(params.t, np.nan),
            out_of_img.sum(),
            False,
            True,
        ]
        assert ret[-1] is not None
        return ret
    lambdas_, R_, t_ = solve(x, X, params.camera.intrinsic_matrix)
    # Should be the case
    # assert lambdas_[0] == 1
    # lambdas_ = lambdas[1:]

    assert lambdas.shape == lambdas_.shape == (2,)
    assert R.shape == R_.shape == (3, 3)
    assert t.shape == t_.shape == (3,)
    ret = [lambdas, lambdas_, R, R_, t, t_, out_of_img.sum(), False, False]
    assert ret[-1] is not None
    return ret


def gen_data(n=int(1e6)) -> pd.DataFrame:
    n = 8000
    results = process_map(gen_sample, range(n), chunksize=1000)
    # n = 10
    # results = [gen_sample(2) for _ in range(n)]

    n_lambdas = len(results[0][0])
    results = [
        list(lambdas)
        + list(lambdas_)
        + list(R.flatten())
        + list(R_.flatten())
        + list(t)
        + list(t_)
        + [out_of_img, had_ve, not_enough_points]
        for lambdas, lambdas_, R, R_, t, t_, out_of_img, had_ve, not_enough_points in results
    ]
    if any(r[-1] is None for r in results):
        __import__("ipdb").set_trace()
    df = pd.DataFrame(
        results,
        columns=[f"lambda_gt_{i}" for i in range(n_lambdas)]
        + [f"lambda_{i}" for i in range(n_lambdas)]
        + [f"R_gt_{i}{j}" for i in range(3) for j in range(3)]
        + [f"R_{i}{j}" for i in range(3) for j in range(3)]
        + [f"t_gt_{i}" for i in range(3)]
        + [f"t_{i}" for i in range(3)]
        + ["out_of_img", "had_ve", "not_enough_points"],
    )
    if df["not_enough_points"].isna().sum() != 0:
        __import__("ipdb").set_trace()
    return df
