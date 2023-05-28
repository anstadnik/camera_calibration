from functools import partial
import numpy as np
import jax.numpy as jnp
import optax
import jax
from calibration.benchmark.benchmark_result import calc_error
from calibration.benchmark.features import Features

from calibration.solver.optimization.backproject import backproject, backprojection_loss
from calibration.solver.optimization.helpers import params_to_proj
from calibration.solver.optimization.rotation import euler_angles_to_rotation_matrix

jArr = jax.Array


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
    schedule = optax.join_schedules([schedule1, schedule2], [5000] * 1)
    # schedule = optax.join_schedules([schedule1, schedule2], [5000])
    # optimizer = optax.amsgrad(step_size)
    optimizer = optax.amsgrad(schedule)

    phase_l = [
        ["t", "theta_z"],
        ["theta_x", "theta_y"],
        ["lambdas"],
        ["focal_length", "sensor_size"],
    ]
    phase_l = [list(params)]
    param_to_phase = {p: i for i, ps in enumerate(phase_l) for p in ps}
    mask_for_phase = [
        {p: phase_ > phase for p, phase_ in param_to_phase.items()}
        for phase in range(len(phase_l))
    ]
    phases = [optax.masked(optax.set_to_zero(), mask) for mask in mask_for_phase]
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
                theta = jnp.concatenate([params["theta_x"], params["theta_y"], params["theta_z"]])
                R = euler_angles_to_rotation_matrix(theta)
                t = params["t"]
                lambdas = params["lambdas"]

                pixel_size = params["sensor_size"] / resolution
                fx, fy = params["focal_length"] / pixel_size
                cx, cy = resolution / 2
                intrinsic_matrix = jnp.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                print(f"R: {np.linalg.norm(R - proj_.R)}, "
                      f"t: {np.linalg.norm(t - proj_.t)}, "
                      f"lambdas: {np.linalg.norm(lambdas - proj_.lambdas)}, "
                      f"intrinsics: {np.linalg.norm(intrinsic_matrix - proj_.camera.intrinsic_matrix)}")
                print(f"Resolution: {np.linalg.norm(resolution - proj_.camera.resolution)}"
                      f"Sensor size: {np.linalg.norm(params['sensor_size'] - proj_.camera.sensor_size)}"
                      f"Focal length: {np.linalg.norm(params['focal_length'] - proj_.camera.focal_length)}"
                      f"Skew: {proj_.camera.skew}")

                board__ = backproject(corners, R, t, lambdas, intrinsic_matrix)
                print(f"Board error: {np.linalg.norm(board_ - board__)}")
                board_error = np.linalg.norm(board_ - board)
                # print(f"Board error: {np.linalg.norm(board_ - board)}")

                weights = jnp.abs(corners.astype(np.float32) / resolution - 0.5).mean(
                    axis=1
                )
                loss_ = jnp.mean(
                    jnp.abs(board_ - board) * (1 + weights * 10).reshape(-1, 1)
                )

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
            phase += 1
            loss_history = jnp.full(patience, jnp.inf)
            # if phase >= len(phases):
            #     break
            if i > 10000:
                break

        i += 1

    assert isinstance(params, dict)
    return best_params, hist
