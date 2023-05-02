import os
import plotly.express as px
import pandas as pd

from tqdm.auto import tqdm
from typing import List
from dataclasses import dataclass
from PIL import Image


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
    corners: List[Corner]
    image: Image.Image


@dataclass
class Dataset:
    name: str
    train: List[CalibrationData]
    test: List[CalibrationData]

    @classmethod
    def from_dir(cls, dir_path: str, name: str) -> "Dataset":
        train_dir = os.path.join(dir_path, "train")
        test_dir = os.path.join(dir_path, "test")
        train = cls._load(train_dir)
        test = cls._load(test_dir)
        return cls(name, train, test)

    @staticmethod
    def _load(dir_path: str) -> List[CalibrationData]:
        calibration_data = []
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith(".orpc"):
                    orpc_file = os.path.join(dir_path, file)
                    calibration_data.append(Dataset._load_file(orpc_file))
        return calibration_data

    @staticmethod
    def _load_file(orpc_file: str) -> CalibrationData:
        with open(orpc_file) as f:
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
        jpeg_file = os.path.join(os.path.dirname(orpc_file), name + ".jpg")
        image = Image.open(jpeg_file)
        return CalibrationData(
            name, width, height, num_corners, encoding, corners, image
        )


def load_babelcalib(data_dir="./data/BabelCalib", root_dir=None) -> List[Dataset]:
    root_dir = root_dir or data_dir
    datasets = []
    for file in tqdm(os.listdir(data_dir), leave=False):
        full_path = os.path.join(data_dir, file)
        if os.path.isdir(full_path):
            dirs = os.listdir(full_path)
            if "train" in dirs and "test" in dirs:
                name = os.path.relpath(full_path, start=root_dir)
                datasets.append(Dataset.from_dir(full_path, name=name))
            else:
                datasets.extend(load_babelcalib(full_path, root_dir))
    return datasets


def visualize(calibration_data: CalibrationData):
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
    fig.show()
