import os
import pickle
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
from PIL import Image
from plotly.graph_objs import Figure
from tqdm.auto import tqdm


@dataclass
class Corner:
    camera_id: int
    corner_id: int
    marker_id: int
    x: float
    y: float


@dataclass
class CalibrationData:
    name: str
    width: int
    height: int
    num_corners: int
    encoding: str
    corners: list[Corner]
    image: Image.Image | None


@dataclass
class Dataset:
    name: str
    train: list[CalibrationData]
    test: list[CalibrationData]

    @classmethod
    def from_dir(cls, dir_path: str, name: str) -> "Dataset":
        train = cls._load(os.path.join(dir_path, "train"))
        test = cls._load(os.path.join(dir_path, "test"))
        return cls(name, train, test)

    @staticmethod
    def _load(dir_path: str) -> list[CalibrationData]:
        files = sorted(os.listdir(dir_path))
        data = [
            Dataset._load_orpc(os.path.join(dir_path, path))
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

    @staticmethod
    def _load_orpc(orpc_path: str) -> CalibrationData:
        with open(orpc_path) as f:
            lines = f.readlines()
            name = lines[0].split(":")[1].strip()
            width = int(lines[1].split(":")[1].strip())
            height = int(lines[2].split(":")[1].strip())
            num_corners = int(lines[3].split(":")[1].strip())
            encoding = lines[4].split(":")[1].strip()
            corners = []
            for row in lines[5:]:
                data = row.strip().split(",")
                corner = Corner(
                    int(data[0]),
                    int(data[1]),
                    int(data[2]),
                    float(data[3]),
                    float(data[4]),
                )
                corners.append(corner)
        return CalibrationData(
            name, width, height, num_corners, encoding, corners, None
        )


def load_babelcalib(data_dir="./data/BabelCalib", root_dir=None) -> list[Dataset]:
    pkl_path = os.path.join(data_dir, "ds.pkl")
    if root_dir is None and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    datasets = []
    for file in tqdm(os.listdir(data_dir), leave=False):
        full_path = os.path.join(data_dir, file)
        if os.path.isdir(full_path):
            dirs = os.listdir(full_path)
            if "train" in dirs and "test" in dirs:
                name = os.path.relpath(full_path, start=root_dir or data_dir)
                datasets.append(Dataset.from_dir(full_path, name=name))
            else:
                datasets.extend(load_babelcalib(full_path, root_dir or data_dir))
    if root_dir is None:
        with open(pkl_path, "wb") as f:
            pickle.dump(datasets, f)
    return datasets


def visualize(calibration_data: CalibrationData) -> Figure:
    df = pd.DataFrame(
        [[corner.x, corner.y] for corner in calibration_data.corners],
        columns=["x", "y"],
    )
    fig = px.scatter(
        df, x="x", y="y", width=calibration_data.width, height=calibration_data.height
    )
    fig.update_traces(
        marker=dict(size=10, color="red", line=dict(width=2, color="DarkSlateGrey"))
    )
    fig.update_layout(
        images=[
            dict(
                source=calibration_data.image,
                xref="x",
                yref="y",
                x=0,
                y=calibration_data.height,
                sizex=calibration_data.width,
                sizey=calibration_data.height,
                sizing="stretch",
                opacity=1,
                layer="below",
            )
        ]
    )
    return fig
