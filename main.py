import numpy as np
from icecream import ic

from calibration.simulator.board import gen_charuco_grid
from calibration.simulator.simul import SimulParams, simul_projection
from calibration.solver.extrinsics import solve_extrinsic


def test_solver():
    # params = SimulParams(R=np.eye(3), t=np.array([0, 0, 3]))
    params = SimulParams(
        R=np.eye(3), t=np.array([0, 0, 1]), lambdas=np.array([0.0, 0.0])
    )
    # draw_board(simul_projection(gen_charuco_grid(5, 7, 0.4, 0.2))[1])
    X, x, lambdas, R, t = simul_projection(gen_charuco_grid(7, 9, 0.4, 0.2), params)

    # X += 1
    H = solve_extrinsic(X, x, params.camera.image_center)
    ic(R)
    ic(t)
    ic(H)
    # # solve(X, x)
    # # draw_board(simul_projection(gen_checkerboard_grid(5, 7))[1])


if __name__ == "__main__":
    test_solver()
