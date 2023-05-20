import unittest
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

from calibration.projector.projector import Projector

from .board import gen_charuco_grid, gen_checkerboard_grid
from .camera import Camera


class TestBoard(unittest.TestCase):
    def test_gen_checkerboard_grid(self):
        b = [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [3, 1],
            [4, 1],
            [0, 2],
            [1, 2],
            [2, 2],
            [3, 2],
            [4, 2],
        ]
        b /= np.array([4, 2])

        np.testing.assert_array_equal(b.min(axis=0), [0.0, 0.0])
        np.testing.assert_array_equal(b.max(axis=0), [1.0, 1.0])
        np.testing.assert_array_equal(gen_checkerboard_grid(3, 5), b)

    def test_gen_charuco_grid(self):
        b = np.array(
            [
                [0, 0],
                [0.4, 0],
                [1.0, 0],
                [1.4, 0],
                [2.0, 0],
                [0, 0.2],
                [0.4, 0.2],
                [1.0, 0.2],
                [1.4, 0.2],
                [2.0, 0.2],
                [0, 1.0],
                [0.4, 1.0],
                [1.0, 1.0],
                [1.4, 1.0],
                [2.0, 1.0],
            ]
        )
        b /= np.array([2.0, 1.0])

        np.testing.assert_array_equal(b.min(axis=0), [0.0, 0.0])
        np.testing.assert_array_equal(b.max(axis=0), [1.0, 1.0])
        np.testing.assert_array_equal(gen_charuco_grid(3, 5, 0.4, 0.2), b)


class TestCamera(unittest.TestCase):
    def test_default_values(self):
        expected_matrix = np.array(
            [[1166.66667, 0, 600], [0, 1166.66667, 400], [0, 0, 1]]
        )
        result_matrix = Camera().intrinsic_matrix
        np.testing.assert_allclose(result_matrix, expected_matrix, atol=1e-10)

    def test_custom_values(self):
        f = 135.0
        sensor_size = np.array([40, 30])
        resolution = np.array([1920, 1080])
        skew = 1.0

        expected_matrix = np.array(
            [[6480.0, 1.0, 960.0], [0.0, 4860.0, 540.0], [0.0, 0.0, 1.0]]
        )
        result_matrix = Camera(
            focal_length=f, sensor_size=sensor_size, resolution=resolution, skew=skew
        ).intrinsic_matrix
        np.testing.assert_allclose(result_matrix, expected_matrix, atol=1e-10)


# class TestProjector(unittest.TestCase):
#     def test_proj_equal_backproj(self):
#         Rs = [
#             np.eye(3),
#             Rotation.from_euler("z", 10, degrees=True).as_matrix(),
#             Rotation.from_euler("xyz", [10, 10, 10], degrees=True).as_matrix(),
#         ]
#         lambdass = [
#             np.array([l1, l2])
#             for l1 in np.arange(-5.0, 1.1, 2)
#             for l2 in np.arange(
#                 -2.61752136752137 * l1 - 6.85141810943093,
#                 -2.61752136752137 * l1 - 4.39190876941320,
#                 1.0,
#             )
#         ]
#         cameras = [
#             Camera(),
#             Camera(135.0, np.array([40, 30]), np.array([1920, 1080]), 1.0),
#         ]
#         ts_for_cameras = [
#             list(map(np.array, product([-1.0, 0.0], [-0.7, -0.3], [3.0, 4.0]))),
#             list(map(np.array, product([-1.7, -0.8], [-1.2, -0.8], [13.0, 20.0]))),
#         ]
#         boards = [gen_checkerboard_grid(7, 9), gen_charuco_grid(7, 9, 0.4, 0.2)]
#
#         def f(R, t, lambdas, camera, board):
#             with self.subTest(R=R, t=t, lambdas=lambdas, camera=camera):
#                 self.assertEqual(board.dtype, np.float64)
#                 proj = Projector(R=R, t=t, lambdas=lambdas, camera=camera)
#                 try:
#                     x = proj.project(board)
#                 except ValueError:
#                     self.fail("ValueError in project")
#
#                 self.assertTrue((x > 0).all())
#                 self.assertTrue((x < proj.camera.resolution).all())
#                 X_ = proj.backproject(x)
#                 np.testing.assert_allclose(X_, board, atol=1e-10)
#
#         for camera, ts in zip(
#             tqdm(cameras, leave=False, desc="Testing projector"), ts_for_cameras
#         ):
#             for R, lambdas, t, board in tqdm(
#                 product(Rs, lambdass, ts, boards),
#                 leave=False,
#                 total=len(Rs) * len(lambdass) * len(ts) * len(boards),
#             ):
#                 f(R, t, lambdas, camera, board)
#
#         d = boards[0].max(axis=0)
#         f(
#             Rs[0],
#             ts_for_cameras[0][0] + np.r_[d, 0],
#             lambdass[0],
#             cameras[0],
#             boards[0][::-1] - d,
#         )
