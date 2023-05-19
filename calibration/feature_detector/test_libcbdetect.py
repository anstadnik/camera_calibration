# pyright: reportUnusedImport=false
import unittest

import numpy as np
from PIL import Image


class TestFeatures(unittest.TestCase):
    def test_feature_detection(self):
        try:
            from cbdetect_py import (
                CornerType,
                Params,
                boards_from_corners,
                find_corners,
            )
        except ImportError:
            self.fail("ImportError: One or more imports failed")

        img_path = "./data/BabelCalib/OV/cube/ov01/train/0006.pgm"
        with Image.open(img_path) as img:
            img.load()
        img = np.array(img)
        params = Params()
        params.show_processing = False
        params.corner_type = (
            CornerType.SaddlePoint
            # if target.type == BoardType.RECTANGULAR
            # else CornerType.MonkeySaddlePoint
        )

        # img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        corners = find_corners(img, params)
        # plot_corners(img, corners)
        boards = boards_from_corners(img, corners, params)

        self.assertTrue(corners.p)
        self.assertTrue(boards)
