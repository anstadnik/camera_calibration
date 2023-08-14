import jax.numpy as jnp
import numpy as np
import jax
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from calibration.projector.camera import Camera
from calibration.projector.projector import Projector
from calibration.solver.optimization.helpers import params_to_proj
from calibration.solver.optimization.optimize import optimize_optax
from calibration.solver.scaramuzza.solve import solve as solve_scaramuzza

jArr = jax.Array
nArr = NDArray[np.float64]


def solve(corners: nArr, board: nArr, camera: Camera) -> Projector | None:
    resolution = camera.resolution
    init_params = solve_scaramuzza(corners, board, camera)
    theta = Rotation.from_matrix(init_params.R).as_euler("xyz")
    # init_params = {
    # "theta_x": jnp.array([0.0]),
    # "theta_y": jnp.array([0.0]),
    # "theta_z": jnp.array([0.0]),
    # "t": jnp.array([1.0, 1.0, 1.0]),
    # "lambdas": jnp.array([0.0, 0.0]),
    # "focal_length": jnp.array([35.0]),
    # "sensor_size": jnp.array([36.0, 24.0]),
    # }
    init_params = {
        "theta_x": jnp.array([theta[0]]),
        "theta_y": jnp.array([theta[1]]),
        "theta_z": jnp.array([theta[2]]),
        "t": jnp.array(init_params.t),
        "lambdas": jnp.array(init_params.lambdas),
        # "lambdas": jnp.array([init_params.lambdas[0], 0.]),
        "focal_length": jnp.array([camera.focal_length]),
        "sensor_size": jnp.array(camera.sensor_size),
    }
    # init_paramss = [
    #     {
    #         "theta_x": jnp.array([0.0]),
    #         "theta_y": jnp.array([0.0]),
    #         "theta_z": jnp.array([th_z]),
    #         "t": jnp.array([1.0, 1.0, 1.0]),
    #         "lambdas": jnp.array([0.0, 0.0]),
    #         "focal_length": jnp.array([35.0]),
    #         "sensor_size": jnp.array([36.0, 24.0]),
    #     }
    #     for th_z in (0.0, jnp.pi)
    # ]
    args = jnp.array(corners), jnp.array(board), jnp.array(resolution)
    params, _ = optimize_optax(init_params, *args)
    # losses = [backprojection_loss(params, *args) for params in paramss]
    # params = paramss[np.argmin(losses)]
    # print(f"Final error: {calc_error(ret, Features(board, corners)):0.3f}")
    # return ret, hist
    return params_to_proj(params, resolution)
