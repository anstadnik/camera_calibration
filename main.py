import pickle as pkl
import numpy as np

from pandas.compat import os

from calibration.benchmark.benchmark import (
    BenchmarkResult,
    benchmark_babelcalib,
    benchmark_simul,
)
from calibration.data.babelcalib.entry import Entry
from calibration.feature_detector.visualization import show_boards


def run_benchmark():
    if not os.path.isfile("simul_results.pkl"):
        simul_results = benchmark_simul(int(1e5))
        with open("simul_results.pkl", "wb") as f:
            pkl.dump(simul_results, f)
    if not os.path.isfile("babelcalib_results.pkl"):
        babelcalib_results = benchmark_babelcalib()
        with open("babelcalib_results.pkl", "wb") as f:
            pkl.dump(babelcalib_results, f)


def run_corner_refinement():
    key = "OV/cube/ov00", "train", "ov00/0031.pgm"
    with open("babelcalib_results.pkl", "rb") as f:
        r: BenchmarkResult = next(
            r
            for r in pkl.load(f)
            if (r.input.ds_name, r.input.subds_name, r.input.name) == key
        )
        assert isinstance(r.input, Entry)
        assert r.input.image is not None
        assert r.features is not None
    show_boards(np.array(r.input.image), r.features.corners,
                r.features.board).show()


if __name__ == "__main__":
    run_benchmark()
    run_corner_refinement()
