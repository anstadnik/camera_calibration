import numpy as np
from icecream import ic


def solve_intrinsic(
    x: np.ndarray, X: np.ndarray, H: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the intrinsic parameters of a camera, given the 2D image points,
    the 3D world points, and the extrinsic parameters.

    Args:
        x (np.ndarray): Points in image space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].
        X (np.ndarray): Points in the board space, with shape (n, 2),
            where n is the number of points and each point is represented as [x, y].
        H (np.ndarray): A 3x3 array representing the extrinsic parameters of the camera.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the intrinsic parameters
            (lambdas) and the updated extrinsic parameters (H)
            with the third translation component (t_3).
    """
    r_11, r_12, t_1, r_21, r_22, t_2, r_31, r_32, _ = H.flatten()

    def A(X):
        return r_21 * X[0] + r_22 * X[1] + t_2

    def B(X, x):
        return x[1] * (r_31 * X[0] + r_32 * X[1])

    def C(X):
        return r_11 * X[0] + r_12 * X[1] + t_1

    def D(X, x):
        return x[0] * (r_31 * X[0] + r_32 * X[1])

    def p(x):
        return np.linalg.norm(x[:2])

    K = len(x)
    N = 2

    A_vec = np.array([A(X[k]) for k in range(K)])
    C_vec = np.array([C(X[k]) for k in range(K)])

    # Interleaving A
    A_C_vec_ = np.empty(K * 2)
    A_C_vec_[::2] = A_vec
    A_C_vec_[1::2] = C_vec

    A_C_vec = np.c_[A_vec, C_vec].flatten()

    np.testing.assert_array_equal(A_C_vec, A_C_vec_)

    p_vals = np.array([p(x_) for x_ in x])
    # Duplicate values [p_1, p_1, p_2, p_2, ..., p_k, p_k]
    p_vals = p_vals.repeat(2)
    p_vals = p_vals.reshape(-1, 1) ** np.arange(N + 1).reshape(1, -1)

    A_C_p_mat = A_C_vec[:, None] * p_vals

    # Change it if we have multiple images
    v_u_mat = -x.reshape(-1)

    M = np.c_[A_C_p_mat, v_u_mat]

    B_vec = np.array([B(X[k], x[k]) for k in range(K)])
    D_vec = np.array([D(X[k], x[k]) for k in range(K)])

    # Interleaving B and D
    B_D_vec = np.c_[B_vec, D_vec].flatten()

    ic(M.shape)
    ic(B_D_vec.shape)
    a_t = np.linalg.lstsq(M, B_D_vec, rcond=None)[0]
    print(a_t)

    return a_t[: N + 1], a_t[N + 1]
