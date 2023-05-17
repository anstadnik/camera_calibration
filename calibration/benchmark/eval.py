from dataclasses import asdict, dataclass

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
        board_ = self.prediction.backproject(self.features.corners)
        return ((board_ - self.features.board) ** 2).mean()


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
