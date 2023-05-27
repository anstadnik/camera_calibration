from functools import partial
import optax

import jax.numpy as jnp
import numpy as np
import jax
from scipy.spatial.transform import Rotation
from calibration.projector.camera import Camera
from calibration.projector.projector import Projector
from calibration.solver.optimization.backproject import backprojection_loss
from calibration.solver.optimization.rotation import euler_angles_to_rotation_matrix
from calibration.solver.scaramuzza.solve import solve as solve_scaramuzza

jArr = jax.Array
nArr = np.ndarray


def params_to_proj(jparams: dict[str, jArr], resolution: nArr) -> Projector | None:
    if any(np.isnan(v).any() for v in jparams.values()):
        return None
    params = {k: np.array(v) for k, v in jparams.items()}

    camera = Camera(
        int(jparams["focal_length"]), params["sensor_size"], np.array(resolution)
    )

    theta = jnp.concatenate(
        [jparams["theta_x"], jparams["theta_y"], jparams["theta_z"]]
    )
    params["R"] = np.array(euler_angles_to_rotation_matrix(theta))

    for p in ["focal_length", "sensor_size", "theta_x", "theta_y", "theta_z"]:
        del params[p]

    return Projector(**params, camera=camera)


# @partial(jax.jit, static_argnames=("step_size", "num_steps"))
def optimize_optax(
    params: dict[str, jArr],
    corners: jArr,
    board: jArr,
    resolution: jArr,
    # step_size=0.001,
    step_size=0.005,
    patience=100,
) -> dict[str, jArr]:
    # optimizer = optax.rmsprop(step_size)
    # optimizer = optax.adam(step_size)
    optimizer = optax.amsgrad(step_size)

    start_optimizing_during_phase = [
        ["t", "theta_z"],
        ["theta_x", "theta_y"],
        ["lambdas"],
        ["focal_length", "sensor_size"],
    ]
    # start_optimizing_during_phase = [
    #     ["t", "theta_z", "theta_x", "theta_y", "lambdas", "focal_length", "sensor_size"]
    # ]
    phase_when_optimize = {
        p: i for i, ps in enumerate(start_optimizing_during_phase) for p in ps
    }

    phases = [
        optax.masked(
            optax.set_to_zero(),
            {p: phase_ > phase for p, phase_ in phase_when_optimize.items()},
        )
        for phase in range(len(start_optimizing_during_phase))
    ]
    inits = [v.init(params) for v in phases]

    opt_state = optimizer.init(params)

    loss_history = jnp.full(patience, jnp.inf)

    # Define a function to compute the gradient and update the parameters
    @partial(jax.jit, static_argnames=("phase"))
    def step(i, params, opt_state, loss_history, phase):
        loss_val, grads = jax.value_and_grad(backprojection_loss)(
            params, corners, board, resolution
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        if phase < len(phases):
            updates, _ = phases[phase].update(updates, inits[phase])
        params_ = optax.apply_updates(params, updates)

        jax.lax.cond(
            i % 1000 == 0,
            lambda _: jax.debug.print(
                "iteration: {}, loss {}, phase: {}", i, loss_val, phase
            ),
            lambda _: None,
            None,
        )

        # Update the loss history
        loss_history = loss_history.at[i % patience].set(loss_val)
        return params_, opt_state, loss_history

    # Run the optimization loop
    i = 0
    phase = 0
    while True:
        params, opt_state, loss_history = step(
            i, params, opt_state, loss_history, phase
        )
        if i >= patience and loss_history[i % patience] >= loss_history.max():
            # proj = params_to_proj(params.copy(), np.array([1200, 800]))
            # assert proj is not None
            # th = jnp.concatenate([params["th_x"], params["th_y"], params["th_z"]])
            # print(f"{np.array(th)=}")
            # print(f"{np.array(proj.t)=}")
            # print(f"{np.array(proj.lambdas)=}")
            # print(f"{np.array(proj.camera.focal_length)=}")
            # print(f"{np.array(proj.camera.sensor_size)=}")
            # print()
            # try:
            #     corners_ = proj.project(np.array(board), 100)
            #     # corners_ = r.prediction.project(r.features.board, 100)
            #     # display(
            #     #     px.scatter(
            #     #         corners_,
            #     #         x=0,
            #     #         y=1,
            #     #         title="Projected corners",
            #     #         range_x=[0, w],
            #     #         range_y=[0, h],
            #     #         # height=h,
            #     #         # width=w,
            #     #     ).update_yaxes(autorange="reversed")
            #     # )
            #     px.scatter(x=corners_[:, 0], y=corners_[:, 1], title=str(phase)).show()
            # except Exception as e:
            #     pass

            phase += 1
            loss_history = jnp.full(patience, jnp.inf)
            if phase >= len(phases):
                break

        i += 1

    assert isinstance(params, dict)
    return params


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
    params = optimize_optax(init_params, *args)
    # losses = [backprojection_loss(params, *args) for params in paramss]
    # params = paramss[np.argmin(losses)]
    return params_to_proj(params, resolution)
