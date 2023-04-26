import numpy as np


def solve_intrinsic(x, X, H) -> np.ndarray:
    r_11, r_21, r_31, r_12, r_22, r_32, t_1, t_2, t_3 = H

    def A(X):
        return r_21 * X[0] + r_22 * X[1] + t_2

    def B(X, x):
        return x[1] * (r_31 * X[0] + r_32 * X[1])

    def C(X):
        return r_11 * X[0] + r_12 * X[1] + t_1

    def D(X, x):
        return x[0] * (r_31 * X[0] + r_32 * X[1])

    def ρ(x):
        return np.linalg.norm(x[:2])

    K = len(x)
    N = 2

    A_vec = np.array([A(X[k]) for k in range(K)])
    C_vec = np.array([C(X[k]) for k in range(K)])

    # Interleaving A
    A_C_vec = np.hstack((A_vec, C_vec)).T.ravel()

    ρ_vals = np.array([ρ(x[k]) for k in range(K)])
    # Duplicate values [ρ₁, ρ₁, ρ₂, ρ₂, ..., ρₖ, ρₖ]
    ρ_vals = np.hstack((ρ_vals, ρ_vals)).T.ravel()
    # Create the matrix with ρ^0, ρ^1, ..., ρ^n in each row
    ρ_mat = np.vstack([ρ_vals**i for i in range(N + 1)]).T

    A_C_ρ_mat = ρ_mat * A_C_vec.reshape(-1, 1)

    # Change it if we have multiple images
    v_u_mat = np.vstack([x[k][:2] for k in range(K)])

    M = np.hstack((A_C_ρ_mat, v_u_mat))

    B_vec = np.array([B(X[k], x[k]) for k in range(K)])
    D_vec = np.array([D(X[k], x[k]) for k in range(K)])

    # Interleaving B and D
    B_D_vec = np.hstack((B_vec, D_vec)).T.ravel()

    a_t = np.linalg.solve(M, B_D_vec)

    return a_t[: N + 1], a_t[N + 1 :]
