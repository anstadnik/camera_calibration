import unittest

import numpy as np

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
        b -= 0.5

        np.testing.assert_array_equal(b.min(axis=0), [-0.5, -0.5])
        np.testing.assert_array_equal(b.max(axis=0), [0.5, 0.5])
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
        b -= 0.5

        np.testing.assert_array_equal(b.min(axis=0), [-0.5, -0.5])
        np.testing.assert_array_equal(b.max(axis=0), [0.5, 0.5])
        np.testing.assert_array_equal(gen_charuco_grid(3, 5, 0.4, 0.2), b)


class TestCamera(unittest.TestCase):
    def test_default_values(self):
        expected_matrix = np.array(
            [[1166.66667, 0, 600], [0, 1166.66667, 400], [0, 0, 1]]
        )
        result_matrix = Camera().intrinsic_matrix
        np.testing.assert_array_almost_equal(result_matrix, expected_matrix, decimal=5)

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
        np.testing.assert_array_almost_equal(result_matrix, expected_matrix, decimal=5)


class TestProjector(unittest.TestCase):
    def test_proj_grid(self):
        for t1 in [-0.1, 0.1]:
            for t2 in [-0.1, 0.1]:
                for t3 in [-0.4, -0.01]:
                    for l1 in np.arange(-0.5, 0.01, 0.03):
                        for l2 in np.arange(
                            -2.61752136752137 * l1 - 6.85141810943093,
                            -2.61752136752137 * l1 - 4.39190876941320,
                            0.1,
                        ):
                            t = np.array([t1, t2, t3])
                            lambdas = np.array([l1, l2])
                            proj = Projector(R=np.eye(3), t=t, lambdas=lambdas)
                            X = gen_checkerboard_grid(7, 9)
                            try:
                                x = proj.project(X)
                            except ValueError:
                                self.fail(f"Value error for {t=}, {lambdas=}")

                            assert (x > 0).all()
                            assert (x < proj.camera.resolution).all()

    def test_proj_equal_backproj(self):
        R_ = np.eye(3)
        t_ = np.array([-0.1, -0.1, -0.4])
        lambdas_ = np.array([0.0, 0.0])

        sensor_size_ = np.array([40, 30])
        res = np.array([1920, 1080])

        boards = [gen_checkerboard_grid(7, 9), gen_charuco_grid(7, 9, 0.4, 0.2)]
        cameras = [Camera(), Camera(135.0, sensor_size_, res, 1.0)]
        for b_i, X in enumerate(boards):
            assert X.dtype == np.float64
            for c_i, camera in enumerate(cameras):
                projectors = [
                    Projector(R=R_, t=t_, lambdas=lambdas_, camera=camera),
                    Projector(R=R_, t=t_, camera=camera),
                    # Projector(camera=camera),
                ]
                for p_i, projector in enumerate(projectors):
                    with self.subTest(board_i=b_i, camera_i=c_i, projector_i=p_i):
                        try:
                            x = projector.project(X)
                        except ValueError:
                            self.fail(f"ValueError in project for {projector.lambdas=}")
                        X_ = projector.backproject(x)
                        np.testing.assert_array_almost_equal(X_, X, decimal=5)
