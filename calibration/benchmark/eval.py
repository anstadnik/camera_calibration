from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from calibration.benchmark.features import Features
from calibration.data.babelcalib.entry import Entry
from calibration.projector.projector import Projector


@dataclass
class BenchmarkResult:
    input: Entry | Projector
    features: Features | None
    prediction: Projector | None

    def _calc_error(self) -> float | None:
        if self.features is None or self.prediction is None:
            return None
        try:
            max_point_img_space = np.r_[self.prediction.camera.resolution, 1]
            max_point = (
                np.linalg.inv(self.prediction.camera.intrinsic_matrix)
                @ max_point_img_space
            )
            max_r = float(np.linalg.norm(max_point[:2]))
            corners_ = self.prediction.project(self.features.board, max_r * 10)
        except ValueError:
            return -1.0
        return np.sqrt(((corners_ - self.features.corners) ** 2).mean())


def eval_simul(results: list[BenchmarkResult]) -> pd.DataFrame:
    # df = pd.DataFrame(asdict(obj) if is_dataclass(obj) else obj for obj in results)
    df = pd.json_normalize(
        [{k: v for k, v in asdict(obj).items() if v is not None} for obj in results]
    )
    df["error"] = process_map(
        BenchmarkResult._calc_error,
        results,
        chunksize=100,
        leave=False,
        desc="Calculating error",
    )
    return df
