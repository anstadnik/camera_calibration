import os


from calibration.benchmark.benchmark import (benchmark_babelcalib,
                                             benchmark_simul)

if __name__ == "__main__":
    if not os.path.isfile("babelcalib_results.pkl"):
        babelcalib_results = benchmark_babelcalib()
        babelcalib_results.to_pickle("babelcalib_results.pkl")
    if not os.path.isfile("simul_results.pkl"):
        simul_results = benchmark_simul(1000)
        simul_results.to_pickle("simul_results.pkl")
