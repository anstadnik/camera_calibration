# from dataclasses import fields
# from tqdm.auto import tqdm
# from itertools import product
# from unittest import TestCase
# import numpy as np
# from scipy.spatial.transform import Rotation
#
# from src.benchmark.benchmark import benchmark_simul
# from src.projector.camera import Camera
#
#
# class TestBenchmark(TestCase):
#     def test_benchmark(self):
#         Rs = [
#             np.eye(3),
#             Rotation.from_euler("z", 10, degrees=True).as_matrix(),
#             Rotation.from_euler("xyz", [10, 10, 10], degrees=True).as_matrix(),
#         ]
#         lambdass = [
#             np.array([l1, l2])
#             for l1 in np.arange(-5.0, 1.1, 2)
#             for l2 in np.arange(
#                 -2.61752136752137 * l1 - 6.85141810943093,
#                 -2.61752136752137 * l1 - 4.39190876941320,
#                 1.0,
#             )
#         ]
#         cameras = [
#             Camera(),
#             Camera(135.0, np.array([40, 30]), np.array([1920, 1080]), 1.0),
#         ]
#         ts_for_cameras = [
#             list(map(np.array, product([-1.0, 0.0], [-0.7, -0.3], [3.0, 4.0]))),
#             list(map(np.array, product([-1.7, -0.8], [-1.2, -0.8], [13.0, 20.0]))),
#         ]
#
#         def f(R, t, lambdas, camera):
#             with self.subTest(t=t, R=R, lambdas=lambdas, camera=camera):
#                 kwargs = dict(R=R, lambdas=lambdas, t=t, camera=camera)
#                 try:
#                     results = benchmark_simul(1, kwargs=kwargs)
#                 except ValueError:
#                     self.fail("ValueError in benchmark_simul")
#                 for k, v in kwargs.items():
#                     for r in results:
#                         if isinstance(v, np.ndarray):
#                             np.testing.assert_array_equal(getattr(r.input, k), v)
#                         else:
#                             for f in fields(v):
#                                 if isinstance(getattr(v, f.name), np.ndarray):
#                                     np.testing.assert_array_equal(
#                                         getattr(v, f.name), getattr(camera, f.name)
#                                     )
#                                 else:
#                                     self.assertEqual(
#                                         getattr(v, f.name), getattr(camera, f.name)
#                                     )
#                 self.assertIsNotNone(results[0].error)
#                 assert results[0].error is not None
#                 self.assertLess(results[0].error, 1e-5)
#
#         for camera, ts in zip(
#             tqdm(cameras, leave=False, desc="Testing benchmark"), ts_for_cameras
#         ):
#             for R, lambdas, t in tqdm(
#                 product(Rs, lambdass, ts),
#                 leave=False,
#                 total=len(Rs) * len(lambdass) * len(ts),
#             ):
#                 f(R, t, lambdas, camera)
