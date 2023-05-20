#!/usr/bin/env python
import numpy as np
import plotly.express as px


kwargs = {
    "R": np.eye(3),
    # "R": np.array(
    #     [
    #         [0.96984631, -0.14131448, 0.19856573],
    #         [0.17101007, 0.97508244, -0.14131448],
    #         [-0.17364818, 0.17101007, 0.96984631],
    #     ]
    # ),
    # "lambdas": np.array([-1.5, -1.92513606]),
    "lambdas": np.array(
[  5.        , -17.93902495]),
    # "lambdas": np.array([0.0, 0.0]),
    # "t": np.array([1.01, 1.01, 2.5]),
    "t": np.array([1.51, 1.51, 4.0]),
}


def test():
    try:
        from calibration.projector.projector import Projector
        from calibration.projector.board import gen_checkerboard_grid
        from calibration.solver.solve import solve

        proj = Projector(**kwargs)
        X = gen_checkerboard_grid(7, 9).astype(np.float64)

        X += X.min(axis=0)
        X /= X.max(axis=0)
        x = proj.project(X)
        px.scatter(x, x=0, y=1).show()

        try:
            proj_ = solve(x, X, proj.camera)
        except TypeError:
            proj_ = solve(x, X, proj.camera.intrinsic_matrix)

        if isinstance(proj_, tuple):
            lambdas_, R_, t_ = proj_
        else:
            lambdas_, R_, t_ = proj_.lambdas, proj_.R, proj_.t

        R = proj.R
        t = proj.t
        lambdas = proj.lambdas
    except ModuleNotFoundError:
        from calibration.benchmark.benchmark import gen_data
        from calibration.simulator.board import gen_checkerboard_grid
        from calibration.simulator.simul import SimulParams, simul_projection
        from calibration.solver.solve import solve

        kwargs["t"] *= -1
        kwargs["R"] = np.linalg.inv(kwargs["R"])
        params = SimulParams(**kwargs)

        X = gen_checkerboard_grid(7, 9).astype(np.float64)
        X += X.min(axis=0)
        X /= X.max(axis=0)
        X, x, lambdas, R, t = simul_projection(X, params)
        lambdas_, R_, t_ = solve(x, X, params.camera.intrinsic_matrix)

    np.testing.assert_array_equal(X[0], [0, 0])

    np.testing.assert_array_almost_equal(abs(R_), abs(R), decimal=5)
    np.testing.assert_array_almost_equal(abs(t_), abs(t), decimal=5)
    np.testing.assert_array_almost_equal(abs(lambdas_), abs(lambdas), decimal=5)

    np.testing.assert_array_almost_equal(R_, R, decimal=5)
    np.testing.assert_array_almost_equal(t_, t, decimal=5)
    np.testing.assert_array_almost_equal(lambdas_, lambdas, decimal=5)


def run_proj():
    from calibration.projector.projector import Projector
    from calibration.projector.board import gen_checkerboard_grid
    from calibration.solver.solve import solve

    proj = Projector(**kwargs)
    X = gen_checkerboard_grid(7, 9).astype(np.float64)

    # X += X.min(axis=0)
    # X /= X.max(axis=0)
    # px.scatter(X, x=0, y=1).show()
    x = proj.project(X)
    px.scatter(x, x=0, y=1).show()
    # px.scatter(proj.backproject(x), x=0, y=1).show()

    proj_ = solve(x, X, proj.camera)

    R = proj.R
    t = proj.t
    lambdas = proj.lambdas
    lambdas_, R_, t_ = proj_.lambdas, proj_.R, proj_.t

    # np.testing.assert_array_almost_equal(abs(R_), abs(R), decimal=5)
    # np.testing.assert_array_almost_equal(abs(t_), abs(t), decimal=5)
    # np.testing.assert_array_almost_equal(abs(lambdas_), abs(lambdas), decimal=5)

    np.testing.assert_array_almost_equal(R_, R, decimal=5)
    np.testing.assert_array_almost_equal(t_, t, decimal=5)
    np.testing.assert_array_almost_equal(lambdas_, lambdas, decimal=5)


if __name__ == "__main__":
    run_proj()
    # test()
