import jax
import jax.numpy as jnp


@jax.jit
def rotation_matrix_to_euler_angles(R: jax.Array) -> jax.Array:
    sy = jnp.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    @jax.jit
    def f1(sy):
        x = jnp.arctan2(R[2, 1], R[2, 2])
        y = jnp.arctan2(-R[2, 0], sy)
        z = jnp.arctan2(R[1, 0], R[0, 0])
        return jax.Array([x, y, z])

    @jax.jit
    def f2(sy):
        x = jnp.arctan2(-R[1, 2], R[1, 1])
        y = jnp.arctan2(-R[2, 0], sy)
        z = 0
        return jax.Array([x, y, z])

    return jax.lax.cond(sy < 1e-6, f2, f1, operand=sy)


@jax.jit
def euler_angles_to_rotation_matrix(theta: jax.Array) -> jax.Array:
    R_x = jnp.array(
        [
            [1, 0, 0],
            [0, jnp.cos(theta[0]), -jnp.sin(theta[0])],
            [0, jnp.sin(theta[0]), jnp.cos(theta[0])],
        ]
    )

    R_y = jnp.array(
        [
            [jnp.cos(theta[1]), 0, jnp.sin(theta[1])],
            [0, 1, 0],
            [-jnp.sin(theta[1]), 0, jnp.cos(theta[1])],
        ]
    )

    R_z = jnp.array(
        [
            [jnp.cos(theta[2]), -jnp.sin(theta[2]), 0],
            [jnp.sin(theta[2]), jnp.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    return jnp.dot(R_z, jnp.dot(R_y, R_x))
