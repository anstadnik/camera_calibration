import os
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass
from glob import glob

from PIL import Image
from tqdm.auto import tqdm

from .entry import Entry
from .target import load_from_dsc_file_tp_file, Board


@dataclass
class Dataset:
    name: str
    targets: list[Board]
    train: list[Entry]
    test: list[Entry]

    @classmethod
    def from_dir(cls, dir_path: str, name: str, targets: list[Board]) -> "Dataset":
        train = cls._load(os.path.join(dir_path, "train"))
        test = cls._load(os.path.join(dir_path, "test"))
        return cls(name, targets, train, test)

    @staticmethod
    def _load(dir_path: str) -> list[Entry]:
        files = sorted(os.listdir(dir_path))
        data = [
            Entry(os.path.join(dir_path, path))
            for path in files
            if path.endswith(".orpc")
        ]

        img_extensions = [".pgm", ".jpg", ".png", ""]

        possible_img_name = os.path.join(dir_path, data[0].name)
        possible_image_names = [possible_img_name + ext for ext in img_extensions]
        try:
            ext_id = list(map(os.path.isfile, possible_image_names)).index(True)
            ext = img_extensions[ext_id]
            assert all(
                os.path.isfile(os.path.join(dir_path, f"{cd.name}{ext}")) for cd in data
            )
        except ValueError:
            img_extensions.remove("")
            img_pathes = sorted(
                [f for f in files if any(map(f.endswith, img_extensions))]
            )
            assert len({p.split(".")[-1] for p in img_pathes}) == 1
            for cal, img_path in zip(data, img_pathes):
                img_path = os.path.join(dir_path, img_path)
                with Image.open(img_path) as image:
                    image.load()
                    cal.image = image
        return data


def load_babelcalib(
    data_dir="./data/BabelCalib",
    root_dir=None,
    targets: list[Board] | dict[str, list[Board]] | None = None,
) -> list[Dataset]:
    pkl_path = os.path.join(data_dir, "ds.pkl")
    if root_dir is None and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    datasets = []

    if glob(os.path.join(data_dir, "*.dsc")):
        assert targets is None
        dsc_paths = sorted(glob(os.path.join(data_dir, "*.dsc")))
        tp_paths = sorted(glob(os.path.join(data_dir, "*.tp")))
        if len(dsc_paths) == 1:
            assert len(tp_paths) == 1
            targets = load_from_dsc_file_tp_file(dsc_paths[0], tp_paths[0])
        else:
            assert len(dsc_paths) == len(tp_paths)
            targets = {}
            for dsc_path, tp_path in zip(dsc_paths, tp_paths):
                assert Path(dsc_path).stem == Path(tp_path).stem
                key = Path(dsc_path).stem
                assert key not in targets
                targets[key] = load_from_dsc_file_tp_file(dsc_path, tp_path)

    for path in tqdm(os.listdir(data_dir), leave=False):
        full_path = os.path.join(data_dir, path)

        if os.path.isdir(full_path):
            dirs = os.listdir(full_path)
            if "train" in dirs and "test" in dirs:
                assert targets is not None
                if isinstance(targets, dict):
                    try:
                        # if "Fisheye2" in full_path:
                        #     __import__("ipdb").set_trace()
                        target = targets[path]
                    except KeyError:
                        logging.warning(f"no target for {full_path}")
                        continue
                else:
                    target = targets
                name = os.path.relpath(full_path, start=root_dir or data_dir)
                datasets.append(Dataset.from_dir(full_path, name=name, targets=target))
            else:
                if isinstance(targets, dict) and path in targets:
                    target = targets[path]
                else:
                    target = targets
                datasets.extend(
                    load_babelcalib(full_path, root_dir or data_dir, target)
                )
    if root_dir is None:
        with open(pkl_path, "wb") as f:
            pickle.dump(datasets, f)
    return datasets
