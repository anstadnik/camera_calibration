import os
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass
from glob import glob

from PIL import Image
from tqdm.auto import tqdm

from src.data.babelcalib.download import assure_babelcalib_downloaded

from .entry import Entry
from .target import load_from_dsc_file_tp_file, Target


@dataclass
class Dataset:
    name: str
    targets: list[Target]
    train: list[Entry]
    test: list[Entry]

    @classmethod
    def from_dir(cls, dir_path: str, name: str, targets: list[Target]) -> "Dataset":
        train = cls._load(name, "train", os.path.join(dir_path, "train"))
        test = cls._load(name, "test", os.path.join(dir_path, "test"))
        return cls(name, targets, train, test)

    @staticmethod
    def _load(ds_name: str, subds_name: str, dir_path: str) -> list[Entry]:
        files = sorted(os.listdir(dir_path))
        data = [
            Entry(ds_name, subds_name, os.path.join(dir_path, path))
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
                os.path.isfile(os.path.join(dir_path, f"{d.name}{ext}")) for d in data
            )
            img_names = [f"{d.name}{ext}" for d in data]
        except ValueError:
            img_extensions.remove("")
            img_names = sorted(
                [f for f in files if any(map(f.endswith, img_extensions))]
            )
            assert len({p.split(".")[-1] for p in img_names}) == 1

        for cal, img_path in zip(data, img_names):
            img_path = os.path.join(dir_path, img_path)
            with Image.open(img_path) as image:
                image.load()
            cal.image = image
        return data


def load_babelcalib(
    data_dir="./data/BabelCalib",
    root_dir=None,
    targets: list[Target] | dict[str, list[Target]] | None = None,
) -> list[Dataset]:
    pkl_path = os.path.join(data_dir, "ds.pkl")
    if root_dir is None and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    if root_dir is None:
        assure_babelcalib_downloaded(data_dir)

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

    for path in tqdm(os.listdir(data_dir), leave=False, desc="Loading BabelCalib"):
        full_path = os.path.join(data_dir, path)

        if os.path.isdir(full_path):
            dirs = os.listdir(full_path)
            if "train" in dirs and "test" in dirs:
                assert targets is not None
                if isinstance(targets, dict):
                    try:
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
