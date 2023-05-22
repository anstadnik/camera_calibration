import unittest
from .babelcalib import load_babelcalib


class TestBabelCalib(unittest.TestCase):
    def test_load_babelcalib(self):
        datasets = load_babelcalib()
        self.assertEqual(len(datasets), 40)
