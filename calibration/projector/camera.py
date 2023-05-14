from dataclasses import dataclass, field

import numpy as np


# TODO: use cached_property
@dataclass
class Camera:
    focal_length: float = 35.0
    sensor_size: np.ndarray = field(default_factory=lambda: np.array([36.0, 24.0]))
    resolution: np.ndarray = field(default_factory=lambda: np.array([1200, 800]))
    skew: float = 0.0

    @property
    def pixel_size(self) -> np.ndarray:
        return self.sensor_size / self.resolution

    @property
    def focal_length_pixels(self) -> np.ndarray:
        return self.focal_length / self.pixel_size

    @property
    def image_center(self) -> np.ndarray:
        return self.resolution / 2

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        fx, fy = self.focal_length_pixels
        cx, cy = self.image_center
        return np.array([[fx, self.skew, cx], [0, fy, cy], [0, 0, 1]])

    @property
    def principal_point(self) -> np.ndarray:
        return self.image_center * self.pixel_size
