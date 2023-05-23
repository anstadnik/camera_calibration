import numpy as np


def solve_intrinsic(
    x: np.ndarray, X: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the intrinsic parameters of a camera, given the 2D image points,
    the 3D world points, and the extrinsic parameters.

    Args:
        x (np.ndarray): Points in image space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].
        X (np.ndarray): Points in the board space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].
        p (np.ndarray): A 3x3 array representing the extrinsic parameters of the camera.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the intrinsic parameters
            (lambdas) and the updated extrinsic parameters (H)
            with the third translation component (t_3).
    """
    r_11, r_12, t_1, r_21, r_22, t_2, r_31, r_32, _ = p.flatten()

    N = 2
    A_vec = r_21 * X[:, 0] + r_22 * X[:, 1] + t_2
    C_vec = r_11 * X[:, 0] + r_12 * X[:, 1] + t_1
    B_vec = x[:, 1] * (r_31 * X[:, 0] + r_32 * X[:, 1])
    D_vec = x[:, 0] * (r_31 * X[:, 0] + r_32 * X[:, 1])

    A_C_vec = np.r_[A_vec, C_vec]

    p_vals = np.linalg.norm(x[:, :2], axis=1)
    p_vals = np.tile(p_vals, 2)
    # p_vals = p_vals[:, np.newaxis] ** (2 * np.arange(1, N + 1)[np.newaxis, :])
    p_vals = np.c_[p_vals**2, p_vals**4]

    A_C_p_mat = A_C_vec[:, None] * p_vals

    # Change it if we have multiple images
    v_u_mat = np.r_[-x[:, 1], -x[:, 0]]

    M = np.c_[A_C_p_mat, v_u_mat]

    B_D_vec = np.r_[B_vec, D_vec]
    B_D_vec -= A_C_vec

    # a_t = np.linalg.lstsq(M, B_D_vec, rcond=None)[0]
    a_t = np.linalg.pinv(M) @ B_D_vec

    return a_t[:N], a_t[N]
