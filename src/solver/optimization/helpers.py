import numpy as np
import jax.numpy as jnp
import jax
from src.projector.camera import Camera

from src.projector.projector import Projector
from src.solver.optimization.rotation import euler_angles_to_rotation_matrix

jArr = jax.Array
nArr = np.ndarray

def params_to_proj(jparams: dict[str, jArr], resolution: nArr) -> Projector | None:
    if any(np.isnan(v).any() for v in jparams.values()):
        return None
    params = {k: np.array(v) for k, v in jparams.items()}

    camera = Camera(
        float(jparams["focal_length"].squeeze()), params["sensor_size"], np.array(resolution)
    )

    theta = jnp.concatenate(
        [jparams["theta_x"], jparams["theta_y"], jparams["theta_z"]]
    )
    params["R"] = np.array(euler_angles_to_rotation_matrix(theta))

    for p in ["focal_length", "sensor_size", "theta_x", "theta_y", "theta_z"]:
        del params[p]

    return Projector(**params, camera=camera)
