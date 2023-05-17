from dataclasses import dataclass

import pandas as pd
from tqdm.contrib.concurrent import process_map

from calibration.benchmark.features import Features
from calibration.data.babelcalib.entry import Entry
from calibration.projector.camera import Camera
from calibration.projector.projector import Projector


@dataclass
class BenchmarkResult:
    input: tuple[Entry, Camera] | Projector
    features: Features
    prediction: Projector

    def _calc_error(self) -> float:
        board_ = self.prediction.backproject(self.features.corners)
        return ((board_ - self.features.board) ** 2).mean()


def eval_simul(results: list[BenchmarkResult]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df["error"] = process_map(
        BenchmarkResult._calc_error,
        results,
        chunksize=100,
        leave=False,
        desc="Calculating error",
    )
    return df
