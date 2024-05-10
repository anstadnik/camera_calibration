from dataclasses import asdict, dataclass, field
import numpy as np
import pandas as pd
from src.benchmark.features import Features
from src.data.babelcalib.entry import Entry
from src.projector.projector import Projector

def calc_error(proj: Projector, features: Features) -> float:
    try:
        max_point_img_space = np.r_[proj.camera.resolution, 1]
        max_point = np.linalg.inv(proj.camera.intrinsic_matrix) @ max_point_img_space
        max_r = float(np.linalg.norm(max_point[:2]))
        corners_ = proj.project(features.board, max_r)
    except ValueError:
        return -1.0
    return np.sqrt(((corners_ - features.corners) ** 2).mean())


@dataclass
class BenchmarkResult:
    input: Entry | Projector
    features: Features
    predictions: dict[str, Projector]
    errors: dict[str, float] = field(init=False)

    def __post_init__(self):
        self.errors = {
            k: calc_error(v, self.features) for k, v in self.predictions.items()
        }


def results_into_df(results: list[BenchmarkResult]) -> pd.DataFrame:
    return pd.json_normalize(
        [{k: v for k, v in asdict(r).items() if v is not None} for r in results]
    )
