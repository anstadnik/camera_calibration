import numpy as np
import pysnooper


# @pysnooper.snoop()
def generate_intrinsic_matrix(
    focal_length=35.0, sensor_size=(36.0, 24.0), resolution=(1200, 800), skew=0.0
):
    # Compute the pixel size in millimeters
    pixel_size = (sensor_size[0] / resolution[0], sensor_size[1] / resolution[1])

    # Compute the focal length in pixels
    fx = focal_length / pixel_size[0]
    fy = focal_length / pixel_size[1]

    # Compute the principal point (optical center) in pixels
    cx = resolution[0] / 2
    cy = resolution[1] / 2

    return np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]])
