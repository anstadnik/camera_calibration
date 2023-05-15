import numpy as np
from tqdm.auto import tqdm
import os
from icecream import ic
from calibration.benchmark.benchmark import gen_data
from calibration.data.babelcalib.babelcalib import load_babelcalib
from cbdetect_py import CornerType, Params, boards_from_corners, find_corners
from calibration.features.visualization import show_corners

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

    # print(x[:20])
    # draw_board(x, max_xy=projector.camera.resolution).show()
    draw_board(X).show()
    # draw_board(projector.backproject(X)).show()
    return

    assert (x > 0).all()
    assert (x < projector.camera.resolution).all()
    lambdas_, R_inv_, t_inv_ = solve(x, X, projector.camera.intrinsic_matrix)
    ic(projector.R.round(3))
    ic(R_inv_.round(3))
    ic(projector.t.round(3))
    ic(t_inv_.round(3))
    ic(projector.lambdas.round(3), lambdas_.round(3))


def hm():
    rez = []

    for t1 in np.arange(-0.3, 0.31, 0.3):
        for t2 in np.arange(-0.3, 0.31, 0.3):
            for t3 in np.arange(-3, -1.5, 0.3):
                for l1 in np.arange(-1.5, 1.51, 0.3):
                    for l2 in np.arange(
                        -2.61752136752137 * l1 - 6.85141810943093,
                        -2.61752136752137 * l1 - 4.39190876941320,
                        0.1,
                    ):
                        t = np.array([t1, t2, t3])
                        lambdas = np.array([l1, l2])

                        key = [*t, *lambdas]
                        proj = Projector(R=np.eye(3), t=t, lambdas=lambdas)
                        X = gen_checkerboard_grid(7, 9)
                        try:
                            x = proj.project(X)

                            if all(x > 0) and all(x < proj.camera.resolution):
                                rez.append(key + [0])
                            else:
                                rez.append(key + [1])

                        except ValueError as e:
                            if str(e) != "f(a) and f(b) must have different signs":
                                raise
                            # self.fail(f"Value error for {t=}, {lambdas=}")
                            rez.append(key + [2])


def gen_features():
    datasets = load_babelcalib()
    for ds in tqdm(datasets):
        for t in ds.targets:
            print(f"Ds: {ds.name}, {t.type} board {t.rows}x{t.cols}")
            if not all(p[2] == 0 for p in t.pts):
                print(" Has weird points")
                # for i, t in enumerate(ds.targets):
                #     for j, p in enumerate(t.pts):
                #         if p[2] != 0:
                #             print(f"{i=}, {j=}, {p=}")
                # break
        continue
        for subds in tqdm([ds.train, ds.test]):
            for entry in subds:
                img = np.array(entry.image)
                # corners = Corner()
                # boards = []
                params = Params()
                params.show_processing = False
                params.corner_type = (
                    CornerType.SaddlePoint
                    if ds.targets[0].type == 0
                    else CornerType.MonkeySaddlePoint
                )

                # img = cv2.imread(image_path, cv2.IMREAD_COLOR)

                corners = find_corners(img, params)
                # plot_corners(img, corners)
                boards = boards_from_corners(img, corners, params)
                # plot_boards(img, corners, boards, params)

                if not corners.p:
                    print("No corners found")
                if not boards:
                    show_corners(img, corners).show()
                    return
                assert corners.p
                assert boards


if __name__ == "__main__":
    gen_features()
