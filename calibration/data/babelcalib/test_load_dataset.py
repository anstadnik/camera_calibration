import unittest
from .babelcalib import load_babelcalib

class TestBabelCalib(unittest.TestCase):
    def test_load_babelcalib(self):
        datasets = load_babelcalib()
        assert len(datasets) == 40
