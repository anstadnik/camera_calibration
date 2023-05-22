from functools import partial
from typing import Callable
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from calibration.projector.camera import Camera
from calibration.projector.projector import Projector

from .features import Features


_SOLVER = Callable[[np.ndarray, np.ndarray, Camera], Projector]


def _calibrate_helper(
    arg: tuple[Features | None, Camera], solvers: list[tuple[str, _SOLVER]]
) -> dict[str, Projector]:
    features, camera = arg
    if features is None:
        return {}
    return features and {
        solver_name: solver(features.corners, features.board, camera)
        for solver_name, solver in solvers
    }


def calibrate(
    solvers: list[tuple[str, _SOLVER]],
    feature_and_camera: list[tuple[Features, Camera]],
) -> list[dict[str, Projector]]:
    return list(
        map(
            partial(_calibrate_helper, solvers=solvers),
            tqdm(feature_and_camera),
        )
    )
    # return process_map(
    #     partial(_calibrate_helper, solvers=solvers),
    #     feature_and_camera,
    #     # chunksize=1000,
    #     chunksize=10,
    #     leave=False,
    #     desc="Calibrating",
    # )


#
# def calibrate_optimization(feature_and_camera: list[tuple[Features, Camera]]):
#     cornerss = np.stack([feature.corners for feature, _ in feature_and_camera])
#     boards = np.stack([feature.board for feature, _ in feature_and_camera])
#     resolutions = np.stack([camera.resolution for _, camera in feature_and_camera])
#
#     f = jax.vmap(optimize_optax)
#     optimized_params = f(
#         jnp.array(cornerss), jnp.array(boards), jnp.array(resolutions), num_steps=10000
#     )
#
#     params = [dict(zip(optimized_params, t)) for t in zip(*optimized_params.values())]
#     return [
#         params_to_projector(param, resolution)
#         for param, resolution in zip(params, resolutions)
#     ]
