import numpy as np
from icecream import ic

from calibration.simulator.board import draw_board, gen_checkerboard_grid
from calibration.simulator.simul import SimulParams, simul_projection
from calibration.solver.solve import solve


def test_solver():
    # params = SimulParams(
    #     R=np.eye(3), t=np.array([0, 0, 10]), lambdas=np.array([0.0, 0.0])
    # )
    params = SimulParams(R=np.eye(3), t=np.array([0, 0, 15]))
    # params = SimulParams()

    # X, x, lambdas, R, t = simul_projection(gen_charuco_grid(7, 9, 0.4, 0.2), params)
    X, x, lambdas, R, t = simul_projection(gen_checkerboard_grid(7, 9), params)

    # draw_board(X)
    draw_board(x, params.camera.resolution)

    assert (x > 0).all()
    assert (x < params.camera.resolution).all()
    lambdas_, rt = solve(x, X, params.camera.image_center)
    ic(R)
    ic(t)
    ic(rt)
    ic(lambdas, lambdas_)


if __name__ == "__main__":
    test_solver()
