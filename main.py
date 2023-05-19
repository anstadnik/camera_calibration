import os

import pickle as pkl
from calibration.benchmark.benchmark import benchmark_babelcalib, benchmark_simul
from test import test

if __name__ == "__main__":
    test()
    # if not os.path.isfile("babelcalib_results.pkl"):
    #     babelcalib_results = benchmark_babelcalib()
    #     with open("babelcalib_results.pkl", "wb") as f:
    #         pkl.dump(babelcalib_results, f)
    # if not os.path.isfile("simul_results.pkl"):
    #     simul_results = benchmark_simul(int(1e3))
    #     with open("simul_results.pkl", "wb") as f:
    #         pkl.dump(simul_results, f)
