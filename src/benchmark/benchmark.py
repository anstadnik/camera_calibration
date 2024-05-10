import numpy as np
from tqdm.contrib.concurrent import process_map
from src.augmentations.augmentations import overlay_img, prune_corners
from src.benchmark.benchmark_result import BenchmarkResult

from src.benchmark.calib import calibrate

from src.benchmark.features import (
    BABELCALIB_INP,
    SIMUL_INP,
    Features,
    babelcalib_features,
    simul_features,
)
from src.data.babelcalib.babelcalib import Dataset, load_babelcalib
from src.data.babelcalib.entry import Entry
from src.projector.board import gen_checkerboard_grid
from src.projector.camera import Camera
from src.projector.projector import Projector
from src.solver.optimization.solve import solve as solve_optimization
from src.solver.scaramuzza.solve import solve as solve_scaramuzza


def evaluate(
    inp: list[SIMUL_INP] | list[BABELCALIB_INP], projs: list[dict[str, Projector]]
) -> list[BenchmarkResult]:
    return process_map(
        BenchmarkResult,
        [i for i, _ in inp],
        [f for _, f in inp],
        projs,
        chunksize=100,
        leave=False,
        desc="Evaluating",
    )


_solvers = [("Scaramuzza", solve_scaramuzza), ("Optimization", solve_optimization)]


def benchmark_simul(
    n=int(1e5), board=gen_checkerboard_grid(7, 9), kwargs: dict | None = None
) -> list[BenchmarkResult]:
    projs_and_feats = simul_features(n, board, kwargs or {})
    solvers_args = [(f, p.camera) if f else None for p, f in projs_and_feats]
    projs = calibrate(_solvers, solvers_args)
    assert len(projs) == len(projs_and_feats)
    return evaluate(projs_and_feats, projs)


def get_camera_from_entry(entry: Entry) -> Camera:
    assert entry.image is not None
    resolution = np.array(entry.image.size)
    sensor_size = np.array([36, 36.0 * resolution[1] / resolution[0]])
    focal_length = 35.0
    return Camera(
        focal_length=focal_length, resolution=resolution, sensor_size=sensor_size
    )


def benchmark_babelcalib(
    dataset: list[Dataset] | None = None, aug: str | None = None
) -> list[BenchmarkResult]:
    if dataset is None:
        dataset = load_babelcalib()
        # Skip aprilgrid
        aprilgrid_datasets = ["UZH", "Kalibr"]
        dataset = [
            ds for ds in dataset if not any(map(ds.name.startswith, aprilgrid_datasets))
        ]
    if aug == "overlay":
        for ds in dataset:
            for subds in [ds.train, ds.test]:
                for e in subds:
                    if e.image is not None:
                        e.image = overlay_img(e.image)
    ents_and_feats = babelcalib_features(dataset)
    if aug == "prune_corners":
        ents_and_feats = [
            (e, prune_corners(f) if f is not None else None) for e, f in ents_and_feats
        ]
    solvers_args = [
        (f, get_camera_from_entry(e)) if f else None for e, f in ents_and_feats
    ]
    projs = calibrate(_solvers, solvers_args)
    assert len(projs) == len(ents_and_feats)
    return evaluate(ents_and_feats, projs)