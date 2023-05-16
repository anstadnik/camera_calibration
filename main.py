import numpy as np
import pandas as pd
from cbdetect_py import CornerType, Params, boards_from_corners, find_corners
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from calibration.data.babelcalib.babelcalib import load_babelcalib
from calibration.data.babelcalib.entry import Entry
from calibration.data.babelcalib.target import BoardType


def process_entry(entry: Entry):
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
        # "dataset": ds_name,
        # "subdataset": subds_name,
        # "image": i,
        "corners": [(p, s) for p, s in zip(corners.p, corners.score)],
        "boards": [b.idx for b in boards],
    }


def gen_features():
    datasets = load_babelcalib()
    results = []
    for ds in tqdm(datasets[32:]):
        # for t in ds.targets:
        #     print(f"Ds: {ds.name}, {t.type} board {t.rows}x{t.cols}")
        #     if not all(p[2] == 0 for p in t.pts):
        #         print(" Has weird points")
        #         # for i, t in enumerate(ds.targets):
        #         #     for j, p in enumerate(t.pts):
        #         #         if p[2] != 0:
        #         #             print(f"{i=}, {j=}, {p=}")
        #         # break
        # continue
        for subds_name, subds in zip(
            tqdm(["train", "test"], leave=False), [ds.train, ds.test]
        ):
            # print(ds.name, subds_name, subds[28].name)
            # process_entry(subds[22])
            # process_entry(subds[23])
            # process_entry(subds[24])
            assert ds.targets[0].type == BoardType.RECTANGULAR
            try:
                res = process_map(process_entry, subds, leave=False)
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
    gen_features()
