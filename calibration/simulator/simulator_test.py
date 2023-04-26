import unittest

import numpy as np

from .board import gen_charuco_grid, gen_checkerboard_grid
from .camera import generate_intrinsic_matrix


class TestBoard(unittest.TestCase):
    def test_gen_checkerboard_grid(self):
        b = [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
        ]

        np.testing.assert_array_equal(gen_checkerboard_grid(3, 5), np.array(b))

    def test_gen_charuco_grid(self):
        np.array(
            [
                [0, 0],
                [0, 0.4],
                [0, 0.8],
                [0, 1.2],
                [0, 1.6],
                [0.2, 0],
                [0.2, 0.4],
                [0.2, 0.8],
                [0.2, 1.2],
                [0.2, 1.6],
                [0.4, 0],
                [0.4, 0.4],
                [0.4, 0.8],
                [0.4, 1.2],
                [0.4, 1.6],
            ]
        )
        gen_charuco_grid(3, 5, 0.4, 0.2)


class TestIntrinsicMatrix(unittest.TestCase):
    def test_default_values(self):
        expected_matrix = np.array(
            [[1166.66667, 0, 600], [0, 1166.66667, 400], [0, 0, 1]]
        )
        result_matrix = generate_intrinsic_matrix()
        print(result_matrix)
        np.testing.assert_array_almost_equal(result_matrix, expected_matrix, decimal=5)

    def test_custom_values(self):
        f = 135.0
        sensor_size = (40, 30)
        resolution = (1920, 1080)
        skew = 1.0

        expected_matrix = np.array(
            [[6480.0, 1.0, 960.0], [0.0, 4860.0, 540.0], [0.0, 0.0, 1.0]]
        )
        result_matrix = generate_intrinsic_matrix(f, sensor_size, resolution, skew)
        np.testing.assert_array_almost_equal(result_matrix, expected_matrix, decimal=5)
