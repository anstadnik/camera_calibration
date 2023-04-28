import numpy as np
from icecream import ic

from calibration.simulator.board import gen_checkerboard_grid
from calibration.simulator.simul import SimulParams, simul_projection
from calibration.solver.solve import solve


def test_solver():
    # params = SimulParams(
    #     R=np.eye(3), t=np.array([-4, -3, 15]), lambdas=np.array([0.0, 0.0])
    # )
    params = SimulParams(R=np.eye(3), t=np.array([-4, -3, 15]))
    # params.t[:2] = params.camera.principal_point
    # params = SimulParams()

    # X, x, lambdas, R, t = simul_projection(gen_charuco_grid(7, 9, 0.4, 0.2), params)
    X, x, lambdas, R, t = simul_projection(gen_checkerboard_grid(7, 9), params)

    # print(x)
    # draw_board(x, max_xy=params.camera.resolution)

    assert (x > 0).all()
    assert (x < params.camera.resolution).all()
    lambdas_, R_, t_ = solve(x, X, params.camera.image_center)
    ic(R.round(3))
    ic(R_.round(3))
    ic(t.round(3))
    ic(t_.round(3))
    ic(lambdas.round(3), lambdas_.round(3))


if __name__ == "__main__":
    np.random.seed(44)
    test_solver()
    # df = gen_data()
    # # df.to_pickle("/tmp/data.pkl")
    # # df = pd.read_pickle("/tmp/data.pkl")
