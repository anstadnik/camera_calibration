import unittest

import numpy as np

from .board import gen_charuco_grid, gen_checkerboard_grid


class TestSimul(unittest.TestCase):
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
