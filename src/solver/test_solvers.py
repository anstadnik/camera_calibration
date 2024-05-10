import numpy as np
from itertools import product
from scipy.spatial.transform import Rotation

import pytest
from src.projector.board import gen_charuco_grid, gen_checkerboard_grid
from src.projector.camera import Camera
from src.projector.projector import Projector
from .optimization.solve import solve as solve_optimization
from .scaramuzza.solve import solve as solve_scaramuzza

Rs = [
    np.eye(3),
    Rotation.from_euler("xyz", [5, 5, 5], degrees=True).as_matrix(),
    Rotation.from_euler("xyz", [-5, 3, 5], degrees=True).as_matrix(),
    Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix(),
]
lambdass = [
    np.array([l1, l2])
    for l1 in np.arange(-5.0, 1.1, 2)
    for l2 in np.arange(
        -2.61752136752137 * l1 - 6.85141810943093,
        -2.61752136752137 * l1 - 4.39190876941320,
        1.0,
    )
]
cameras = [
    Camera(),
    Camera(135.0, np.array([40, 30]), np.array([1920, 1080]), 1.0),
]
ts_for_cameras = [
    list(map(np.array, product([-1.0, 0.0], [-0.7, -0.3], [3.0, 4.0]))),
    list(map(np.array, product([-1.7, -0.8], [-1.2, -0.8], [20.0]))),
]

boards = [gen_checkerboard_grid(7, 9), gen_charuco_grid(7, 9, 0.4, 0.2)]

solve_functions = [
    ("optimization", solve_optimization),
    ("scaramuzza", solve_scaramuzza),
]

parameters = [
    args
    for camera, ts in zip(cameras, ts_for_cameras)
    for args in product(solve_functions, [camera], ts, Rs, lambdass, boards)
]


@pytest.mark.parametrize("solve_name_solve, camera, t, R, lambdas, board", parameters)
def test_proj_equal_backproj(solve_name_solve, camera, t, R, lambdas, board):
    solve_name, solve = solve_name_solve
    proj = Projector(R=R, t=t, lambdas=lambdas, camera=camera)
    x = proj.project(board)

    assert (x > 0).all()
    assert (x < proj.camera.resolution).all()

    X_ = proj.backproject(x)
    np.testing.assert_allclose(X_, board, atol=1e-10, rtol=1e-6)

    proj_, hist = solve(x, board, camera)
    atol = 1e-3 if camera.focal_length == 135 else 1e-5
    np.testing.assert_allclose(proj.t, proj_.t, atol=atol)
    np.testing.assert_allclose(proj.R, proj_.R, atol=atol)
    np.testing.assert_allclose(proj.lambdas, proj_.lambdas, atol=atol)

    x_ = proj_.project(board)
    np.testing.assert_allclose(x_, x, atol=1e-10, rtol=1e-5)
