import optax

import jax.numpy as jnp
import numpy as np
import jax
from calibration.projector.camera import Camera
from calibration.projector.projector import Projector
from calibration.solver.optimization.backproject import backprojection_loss
from calibration.solver.optimization.rotation import euler_angles_to_rotation_matrix


# @partial(jax.jit, static_argnames=("step_size", "num_steps"))
def optimize_optax(
    corners: jax.Array,
    board: jax.Array,
    resolution: jax.Array,
    step_size=0.01,
    num_steps=30000,
    # num_steps=10000,
) -> dict[str, jax.Array]:
    params = {
        "theta": jnp.array([1.0, 1.0, 1.0]),
        "t": jnp.array([1.0, 1.0, 1.0]),
        "lambdas": jnp.array([-1.0, -1.0]),
        "focal_length": jnp.array([35.0]),
        "sensor_size": jnp.array([36.0, 24.0]),
    }

    # Create an Adam optimizer
    optimizer = optax.adam(step_size)

    # Initialize the optimizer state.
    opt_state = optimizer.init(params)

    # Define a function to compute the gradient and update the parameters
    @jax.jit
    def step(i, params, opt_state):
        loss_value, grads = jax.value_and_grad(backprojection_loss)(
            params, corners, board, resolution
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        jax.lax.cond(
            i % 1000 == 0,
            # lambda _: jax.debug.print(f"iteration: {i}, loss: {loss_value}"),
            lambda _: jax.debug.print("iteration: {}, loss: {}",i, loss_value),
            lambda _: None,
            None)
        return params, opt_state, loss_value

    # Run the optimization loop
    params, opt_state, _ = jax.lax.fori_loop(
        0, num_steps, lambda i, x: step(i, *x[:2]), (params, opt_state, 0.0)
    )

    assert isinstance(params, dict)
    return params


def solve(corners: np.ndarray, board: np.ndarray, camera: Camera) -> Projector | None:
    resolution = camera.resolution
    # params = optimize_gauss_newton(
    params = optimize_optax(jnp.array(corners), jnp.array(board), jnp.array(resolution))
    params = {k: np.array(v) for k, v in params.items()}

    if any(np.isnan(v).any() for v in params.values()):
        return None

    camera = Camera(
        focal_length=int(params["focal_length"]),
        sensor_size=params["sensor_size"],
        resolution=np.array(resolution),
    )

    del params["focal_length"]
    del params["sensor_size"]

    params["R"] = euler_angles_to_rotation_matrix(params["theta"])
    del params["theta"]

    return Projector(**params, camera=camera)
