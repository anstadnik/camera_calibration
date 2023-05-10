import numpy as np
from icecream import ic
from calibration.benchmark.benchmark import gen_data

from calibration.projector.board import draw_board, gen_checkerboard_grid
from calibration.projector.projector import Projector
from calibration.solver.solve import solve


def test_solver():
    # projector = Projector(
    #     R=np.eye(3), t=np.array([-4, -3, 15]), lambdas=np.array([0.0, 0.0])
    # )
    projector = Projector(R=np.eye(3))
    # projector.t[:2] = projector.camera.principal_point
    # projector = Projector()

    # X, x, lambdas, R, t = simul_projection(gen_charuco_grid(7, 9, 0.4, 0.2), projector)
    X = gen_checkerboard_grid(7, 9)
    x = projector.project(X)
    # X, x, lambdas, R_inv, t_inv =

    # print(x)
    # draw_board(x, max_xy=projector.camera.resolution).show()
    draw_board(X).show()
    draw_board(projector.backproject(X)).show()

    assert (x > 0).all()
    assert (x < projector.camera.resolution).all()
    lambdas_, R_inv_, t_inv_ = solve(x, X, projector.camera.intrinsic_matrix)
    ic(projector.R.round(3))
    ic(R_inv_.round(3))
    ic(projector.t.round(3))
    ic(t_inv_.round(3))
    ic(projector.lambdas.round(3), lambdas_.round(3))


if __name__ == "__main__":
    # np.random.seed(44)
    # test_solver()
    df = gen_data()
    df.to_pickle("/tmp/data.pkl")
    # df = pd.read_pickle("/tmp/data.pkl")
    # datasets = load_babelcalib()
    # for ds in datasets:
    #     pass
    # print(ds.name)
    # print(len(ds.test))
    # print(len(ds.train))
    # px.imshow(ds.test[0].image).show()
    # break
