from functools import partial
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
    init_params: dict[str, jax.Array],
    corners: jax.Array,
    board: jax.Array,
    resolution: jax.Array,
    step_size=0.05,
    patience=1000,
) -> dict[str, jax.Array]:
    optimizer = optax.adam(step_size)
    zero_all_other_than_t = optax.masked(
        optax.set_to_zero(), {k: k != "t" for k in init_params}
    )
    zero_state = zero_all_other_than_t.init(init_params)

    opt_state = optimizer.init(init_params)

    loss_history = jnp.full(patience, jnp.inf)

    # Define a function to compute the gradient and update the parameters
    @partial(jax.jit, static_argnames=("update_only_t"))
    def step(i, params, opt_state, loss_history, update_only_t=False):
        loss_value, grads = jax.value_and_grad(backprojection_loss)(
            params, corners, board, resolution
        )
        if update_only_t:
            grads = zero_all_other_than_t.update(grads, zero_state)[0]
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params_ = optax.apply_updates(params, updates)

        jax.lax.cond(
            i % 1000 == 0,
            lambda _: jax.debug.print(
                "iteration: {}, loss: {}, update_only_t: {}",
                i,
                loss_value,
                update_only_t,
            ),
            lambda _: None,
            None,
        )

        # Update the loss history
        loss_history = loss_history.at[i % patience].set(loss_value)
        return params_, opt_state, loss_history

    # Run the optimization loop
    i = 0
    phase = 0
    while True:
        init_params, opt_state, loss_history = step(
            i, init_params, opt_state, loss_history, phase == 0
        )
        if i >= patience and loss_history[i % patience] >= loss_history.max():
            phase += 1
            loss_history = jnp.full(patience, jnp.inf)
            if phase >= 2:
                break

        i += 1

    assert isinstance(init_params, dict)
    return init_params


def solve(corners: np.ndarray, board: np.ndarray, camera: Camera) -> Projector | None:
    resolution = camera.resolution
    init_params1 = {
        "theta": jnp.array([1.0, 1.0, 1.0]),
        "t": jnp.array([1.0, 1.0, 1.0]),
        "lambdas": jnp.array([-1.0, -1.0]),
        "focal_length": jnp.array([35.0]),
        "sensor_size": jnp.array([36.0, 24.0]),
    }
    init_params2 = init_params1.copy()
    init_params2["theta"] = jnp.array([1.0, 1.0, 90.0])
    args = jnp.array(corners), jnp.array(board), jnp.array(resolution)
    paramss = [optimize_optax(init_p, *args) for init_p in (init_params1, init_params2)]
    losses = [backprojection_loss(params, *args) for params in paramss]
    params = paramss[np.argmin(losses)]
    if any(np.isnan(v).any() for v in params.values()):
        return None
    params = {k: np.array(v) for k, v in params.items()}

    camera = Camera(
        focal_length=int(params["focal_length"]),
        sensor_size=params["sensor_size"],
        resolution=np.array(resolution),
    )

    params["R"] = euler_angles_to_rotation_matrix(params["theta"])

    del params["focal_length"]
    del params["sensor_size"]
    del params["theta"]

    return Projector(**params, camera=camera)
