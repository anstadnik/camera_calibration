import jax
import jax.numpy as jnp

from .rotation import euler_angles_to_rotation_matrix


@jax.jit
def backproject(
    x: jax.Array,
    R: jax.Array,
    t: jax.Array,
    lambdas: jax.Array,
    intrinsic_matrix: jax.Array,
) -> jax.Array:
    """
    Simulates the backprojection of image points (x) onto a board plane,
    given the simulation parameters (p).

    Args:

    x (np.ndarray):
        An array of point in the image space, with shape (n, 2),
        where n is the number of points and each point is represented as [x, y].

    Returns:
        X (np.ndarray):
            An array of point in the board space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].
    """
    # Intrinsics
    x = jnp.c_[x, jnp.ones(x.shape[0])]
    x = (jnp.linalg.inv(intrinsic_matrix) @ x.T).T
    # Pyright bug
    x = x / x[:, 2][:, None]

    # Distortion
    x = x.at[:, 2].set(psi(lambdas, jnp.linalg.norm(x[:, :2], axis=1)))
    x = x / x[:, 2][:, None]

    # Extrinsics
    P = jnp.c_[R[:, :2], t]
    x = (jnp.linalg.inv(P) @ x.T).T
    x = x / x[:, 2][:, None]
    return x[:, :2]


@jax.jit
def psi(lambdas: jax.Array, r: jax.Array) -> jax.Array:
    l1, l2 = lambdas
    return 1 + l1 * r**2 + l2 * r**4


@jax.jit
def backprojection_loss(
    params: dict[str, jax.Array],
    corners: jax.Array,
    board: jax.Array,
    resolution: jax.Array,
) -> jax.Array:
    R = euler_angles_to_rotation_matrix(params["theta"])
    t = params["t"]
    lambdas = params["lambdas"]

    pixel_size = params["sensor_size"] / resolution
    fx, fy = params["focal_length"] / pixel_size
    cx, cy = resolution / 2
    intrinsic_matrix = jnp.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    board_ = backproject(corners, R, t, lambdas, intrinsic_matrix)
    return jnp.mean(jnp.square(board_ - board))
