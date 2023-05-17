import numpy as np
import pandas as pd
from cbdetect_py import CornerType, Params, boards_from_corners, find_corners
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from calibration.benchmark.benchmark import benchmark_babelcalib, benchmark_simul

from calibration.data.babelcalib.babelcalib import load_babelcalib
from calibration.data.babelcalib.entry import Entry
from calibration.data.babelcalib.target import BoardType


def _process_entry(entry: Entry):
    img = np.array(entry.image)
    params = Params()
    params.show_processing = False
    params.corner_type = (
        CornerType.SaddlePoint
        # if target.type == BoardType.RECTANGULAR
        # else CornerType.MonkeySaddlePoint
    )

    corners = find_corners(img, params)
    boards = boards_from_corners(img, corners, params)

    return {
        "corners": list(zip(corners.p, corners.score)),
        "boards": [b.idx for b in boards],
    }


def gen_features():
    datasets = load_babelcalib()
    return
    results = []
    for ds in tqdm(datasets):
        for subds_name, subds in zip(
            tqdm(["train", "test"], leave=False), [ds.train, ds.test]
        ):
            # print(ds.name, subds_name, subds[28].name)
            # process_entry(subds[22])
            # process_entry(subds[23])
            # process_entry(subds[24])
            assert ds.targets[0].type == BoardType.RECTANGULAR
            try:
                res = process_map(_process_entry, subds, leave=False)
            except:
                # breakpoint()
                raise
            res = [
                d | {"dataset": ds.name, "subdataset": subds_name, "image": i}
                for i, d in enumerate(res)
            ]
            results.extend(res)

    df = pd.DataFrame(results)
    df.to_pickle("features.pkl")


if __name__ == "__main__":
    benchmark_simul(1000)
    benchmark_babelcalib()
