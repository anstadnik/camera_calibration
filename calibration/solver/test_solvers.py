import unittest
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

from calibration.projector.board import gen_charuco_grid, gen_checkerboard_grid
from calibration.projector.camera import Camera
from calibration.projector.projector import Projector
from .optimization import solve as solve_optimization
from .scaramuzza import solve as solve_scaramuzza


class TestProjectorAndSolverOptimization(unittest.TestCase):
    def test_proj_equal_backproj(self):
        Rs = [
            np.eye(3),
            Rotation.from_euler("xyz", [5, 5, 5], degrees=True).as_matrix(),
            Rotation.from_euler("xyz", [-5, 3, 5], degrees=True).as_matrix(),
            Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix(),
            # Rotation.from_euler("xyz", [0, 0, 180], degrees=True).as_matrix(),
        ]
        lambdass = [
            np.array([l1, l2])
            for l1 in np.arange(-5.0, 1.1, 2)
            for l2 in np.arange(
                -2.61752136752137 * l1 - 6.85141810943093,
                -2.61752136752137 * l1 - 4.39190876941320,
                1.0,
            )
        ]
        cameras = [
            Camera(),
            Camera(135.0, np.array([40, 30]), np.array([1920, 1080]), 1.0),
        ]
        ts_for_cameras = [
            list(map(np.array, iter(product([-1.0, 0.0], [-0.7, -0.3], [3.0, 4.0])))),
            list(map(np.array, iter(product([-1.7, -0.8], [-1.2, -0.8], [20.0])))),
        ]
        boards = [gen_checkerboard_grid(7, 9), gen_charuco_grid(7, 9, 0.4, 0.2)]

        def f(R, t, lambdas, camera, board, solve, solve_name):
            with self.subTest(
                R=R, t=t, lambdas=lambdas, camera=camera, solve_name=solve_name
            ):
                self.assertEqual(board.dtype, np.float64)
                proj = Projector(R=R, t=t, lambdas=lambdas, camera=camera)
                x = proj.project(board)

                self.assertTrue((x > 0).all())
                self.assertTrue((x < proj.camera.resolution).all())

                X_ = proj.backproject(x)
                np.testing.assert_allclose(X_, board, atol=1e-10, rtol=1e-6)

                proj_ = solve(x, board, camera)
                atol = 1e-3 if camera.focal_length == 135 else 1e-5
                np.testing.assert_allclose(proj.t, proj_.t, atol=atol)
                np.testing.assert_allclose(proj.R, proj_.R, atol=atol)
                np.testing.assert_allclose(proj.lambdas, proj_.lambdas, atol=atol)

                x_ = proj_.project(board)
                np.testing.assert_allclose(x_, x, atol=1e-10, rtol=1e-5)

        # sourcery skip: no-loop-in-tests
        for solve_name, solve in [
            ("optimization", solve_optimization),
            ("scaramuzza", solve_scaramuzza),
        ]:
            for camera, ts in zip(
                tqdm(cameras, leave=False, desc="Testing projector and solver"),
                ts_for_cameras,
            ):
                for R, lambdas, t, board in tqdm(
                    product(Rs, lambdass, ts, boards),
                    leave=False,
                    total=len(Rs) * len(lambdass) * len(ts) * len(boards),
                ):
                    f(R, t, lambdas, camera, board, solve, solve_name)

            d = boards[0].max(axis=0)
            f(
                Rs[0],
                ts_for_cameras[0][0] + np.r_[d, 0],
                lambdass[0],
                cameras[0],
                boards[0][::-1] - d,
                solve,
                solve_name,
            )
