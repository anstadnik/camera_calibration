from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    focal_length: float = 35.0
    sensor_size: tuple = (36.0, 24.0)
    resolution: tuple = (1200, 800)
    skew: float = 0.0

    @property
    def pixel_size(self) -> tuple:
        return (
            self.sensor_size[0] / self.resolution[0],
            self.sensor_size[1] / self.resolution[1],
        )

    @property
    def fx(self) -> float:
        return self.focal_length / self.pixel_size[0]

    @property
    def fy(self) -> float:
        return self.focal_length / self.pixel_size[1]

    @property
    def image_center(self) -> tuple[float, float]:
        return self.resolution[0] / 2, self.resolution[1] / 2

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, self.skew, self.image_center[0]],
                [0, self.fy, self.image_center[1]],
                [0, 0, 1],
            ]
        )
