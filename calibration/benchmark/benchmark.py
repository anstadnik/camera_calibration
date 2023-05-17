import numpy as np
import pandas as pd

from calibration.benchmark.calib import calibrate
from calibration.benchmark.eval import BenchmarkResult, eval_simul
from calibration.benchmark.features import babelcalib_features, simul_features
from calibration.data.babelcalib.babelcalib import Dataset, load_babelcalib
from calibration.data.babelcalib.entry import Entry
from calibration.projector.board import gen_checkerboard_grid
from calibration.projector.camera import Camera


def benchmark_simul(n=int(1e6), board=gen_checkerboard_grid(7, 9)) -> pd.DataFrame:
    features_and_proj = simul_features(n, board)
    projs_ = calibrate([(f, p.camera) for f, p in features_and_proj])
    benchmark_results = [
        BenchmarkResult(p, f, p_) for (f, p), p_ in zip(features_and_proj, projs_)
    ]
    return eval_simul(benchmark_results)


def get_camera_from_entry(entry: Entry) -> Camera:
    assert entry.image is not None
    resolution = np.array(entry.image.size)
    sensor_size = np.array([36, 36.0 * resolution[1] / resolution[0]])
    focal_length = 35
    return Camera(focal_length, resolution, sensor_size, skew=0)


def benchmark_babelcalib(dataset: list[Dataset] | None = None) -> pd.DataFrame:
    if dataset is None:
        dataset = load_babelcalib()
    features_and_entries = babelcalib_features(dataset)
    cameras = [get_camera_from_entry(e) for _, e in features_and_entries]
    projs_ = calibrate([(f, c) for (f, _), c in zip(features_and_entries, cameras)])
    benchmark_results = [
        BenchmarkResult(e, f, p_) for (f, e), p_ in zip(features_and_entries, projs_)
    ]
    return eval_simul(benchmark_results)
