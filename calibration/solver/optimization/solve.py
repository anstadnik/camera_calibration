from functools import partial
import optax

import jax.numpy as jnp
import numpy as np
import jax
from scipy.spatial.transform import Rotation
from calibration.benchmark.benchmark_result import calc_error
from calibration.benchmark.features import Features
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
    step_size=0.05,
    patience=1000,
) -> dict[str, jArr]:
    # optimizer = optax.rmsprop(step_size)
    # optimizer = optax.adam(step_size)
    # schedule1 = optax.warmup_cosine_decay_schedule(0.001, 0.05, 1000, 2000,
    #                                               0.001)

    schedule1 = optax.warmup_cosine_decay_schedule(0.001, 0.1, 1000, 5000, 0.000001)
    schedule2 = optax.warmup_cosine_decay_schedule(
        0.000001, 0.001, 1000, 5000, 0.000001
    )
    # schedule2=    optax.warmup_cosine_decay_schedule(0.001, 0.1, 1000, 5000, 0.001)
    schedule = optax.join_schedules([schedule1, schedule2], [5000] * 3)
    # schedule = optax.join_schedules([schedule1, schedule2], [5000])
    # optimizer = optax.amsgrad(step_size)
    optimizer = optax.amsgrad(schedule)

    # start_optimizing_during_phase = [
    #     ["t", "theta_z"],
    #     ["theta_x", "theta_y"],
    #     ["lambdas"],
    #     ["focal_length", "sensor_size"],
    # ]
    start_optimizing_during_phase = [
        ["t", "theta_z", "theta_x", "theta_y", "lambdas", "focal_length", "sensor_size"]
    ]
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

        # Update the loss history
        loss_history = loss_history.at[i % patience].set(loss_val)
        return params_, opt_state, loss_history

    # Run the optimization loop
    i = 0
    phase = 0
    hist = []
    best_params, best_loss = None, np.inf
    best_corners_error, best_board_error, best_error = np.inf, np.inf, np.inf
    features = Features(np.array(board), np.array(corners))
    while True:
        params, opt_state, loss_history = step(
            i, params, opt_state, loss_history, phase
        )
        loss_val = loss_history[i % patience]
        if i % 1000 == 0:
            proj_ = params_to_proj(params.copy(), np.array(resolution))
            if proj_ is not None:
                try:
                    corners_ = proj_.project(np.array(board))
                    corners_error = np.linalg.norm(corners_ - corners)
                    # print(f"Corners error: {np.linalg.norm(corners_ - corners)}")
                except Exception:
                    corners_error = np.inf
                    # print("failed projecting")
                board_ = proj_.backproject(np.array(corners).astype(np.float32))
                board_error = np.linalg.norm(board_ - board)
                # print(f"Board error: {np.linalg.norm(board_ - board)}")

                weights = jnp.abs(corners.astype(np.float32) / resolution - 0.5).mean(axis=1)
                loss_= jnp.mean(jnp.abs(board_ - board) * (1 + weights * 10).reshape(-1, 1))

                print(
                    f"iteration: {i}, loss {loss_val}/{np.array(loss_)}, phase: {phase}, "
                    f"corners error: {corners_error:0.3f} vs {best_corners_error:0.3f}, "
                    f"board error: {board_error:0.3f} vs {best_board_error:0.3f}, "
                    f"error: {calc_error(proj_, features):0.3f} vs {best_error:0.3f}"
                )

        hist.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params
            proj_ = params_to_proj(params.copy(), np.array(resolution))
            if proj_ is not None:
                try:
                    corners_ = proj_.project(np.array(board))
                    best_corners_error = np.linalg.norm(corners_ - corners)
                except Exception:
                    corners_error = np.inf
                board_ = proj_.backproject(np.array(corners))
                best_board_error = np.linalg.norm(board_ - board)
                best_error = calc_error(proj_, features)

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
            # if phase >= len(phases):
            #     break
            if i > 10000:
                break

        i += 1

    assert isinstance(params, dict)
    return best_params, hist


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
    params, hist = optimize_optax(init_params, *args)
    # losses = [backprojection_loss(params, *args) for params in paramss]
    # params = paramss[np.argmin(losses)]
    ret = params_to_proj(params, resolution)
    print(f"Final error: {calc_error(ret, Features(board, corners)):0.3f}")
    return ret, hist
