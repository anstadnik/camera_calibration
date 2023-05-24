import pickle as pkl
import numpy as np

from pandas.compat import os

from calibration.benchmark.benchmark import (
    benchmark_babelcalib,
    benchmark_simul,
)
from calibration.projector.camera import Camera
from calibration.solver.optimization.solve import solve


def run_benchmark():
    if not os.path.isfile("babelcalib_results.pkl"):
        babelcalib_results = benchmark_babelcalib()
        with open("babelcalib_results.pkl", "wb") as f:
            pkl.dump(babelcalib_results, f)
    if not os.path.isfile("simul_results.pkl"):
        simul_results = benchmark_simul(int(1e3))
        with open("simul_results.pkl", "wb") as f:
            pkl.dump(simul_results, f)

def debug_test():
    corners = np.array([[558.33307563, 504.842542  ],
       [531.8274827 , 522.49001733],
       [504.42809411, 540.51581882],
       [476.34149001, 558.75704413],
       [447.82010119, 577.02480587],
       [419.15285163, 595.11314233],
       [390.64922842, 612.81203692],
       [362.61840085, 629.92284786],
       [335.34685326, 646.27333435],
       [511.35216838, 490.38560834],
       [484.99160889, 506.90884558],
       [458.0405242 , 523.70718971],
       [430.71995865, 540.63305707],
       [403.28141178, 557.52216574],
       [375.99454313, 574.20216615],
       [349.13086666, 590.50354641],
       [322.94579252, 606.27114647],
       [297.66215734, 621.37421239],
       [467.90381902, 476.37056393],
       [442.09937817, 491.83708643],
       [415.98158061, 507.49873814],
       [389.76798898, 523.22565067],
       [363.69171421, 538.87843524],
       [337.98886031, 554.31565875],
       [312.88443081, 569.40226451],
       [288.57889546, 584.01762736],
       [265.23760915, 598.06193028],
       [428.23202111, 463.00519502],
       [403.29432905, 477.50043595],
       [378.27868642, 492.13125218],
       [353.38568573, 506.78572495],
       [328.81933963, 521.34742067],
       [304.77617481, 535.7013782 ],
       [281.4345261 , 549.74025258],
       [258.94561842, 563.36972239],
       [237.42765814, 576.51241424],
       [392.36489229, 450.42483391],
       [368.50390718, 464.04033855],
       [344.75165555, 477.74831004],
       [321.28381983, 491.45354107],
       [298.27112056, 505.05949276],
       [275.87090854, 518.47284479],
       [254.21984799, 531.60782414],
       [233.42862867, 544.38975996],
       [213.57923082, 556.75747458],
       [360.17018046, 438.70243227],
       [337.50629331, 451.52765186],
       [315.08958156, 464.4146187 ],
       [293.06792477, 477.28285373],
       [271.5794151 , 490.05240141],
       [250.74645901, 502.64717861],
       [230.67122574, 514.99797126],
       [211.43289463, 527.04475884],
       [193.0868275 , 538.73817744],
       [331.4095085 , 427.86126129],
       [309.99090131, 439.97872688],
       [288.91509567, 452.13612891],
       [268.30291753, 464.26553799],
       [248.26332628, 476.30052788],
       [228.88959598, 488.17859829],
       [210.25674714, 499.84323828],
       [192.42038779, 511.24545362],
       [175.41690689, 522.34467342]])

    board= np.array([[0.        , 0.        ],
       [0.125     , 0.        ],
       [0.25      , 0.        ],
       [0.375     , 0.        ],
       [0.5       , 0.        ],
       [0.625     , 0.        ],
       [0.75      , 0.        ],
       [0.875     , 0.        ],
       [1.        , 0.        ],
       [0.        , 0.16666667],
       [0.125     , 0.16666667],
       [0.25      , 0.16666667],
       [0.375     , 0.16666667],
       [0.5       , 0.16666667],
       [0.625     , 0.16666667],
       [0.75      , 0.16666667],
       [0.875     , 0.16666667],
       [1.        , 0.16666667],
       [0.        , 0.33333333],
       [0.125     , 0.33333333],
       [0.25      , 0.33333333],
       [0.375     , 0.33333333],
       [0.5       , 0.33333333],
       [0.625     , 0.33333333],
       [0.75      , 0.33333333],
       [0.875     , 0.33333333],
       [1.        , 0.33333333],
       [0.        , 0.5       ],
       [0.125     , 0.5       ],
       [0.25      , 0.5       ],
       [0.375     , 0.5       ],
       [0.5       , 0.5       ],
       [0.625     , 0.5       ],
       [0.75      , 0.5       ],
       [0.875     , 0.5       ],
       [1.        , 0.5       ],
       [0.        , 0.66666667],
       [0.125     , 0.66666667],
       [0.25      , 0.66666667],
       [0.375     , 0.66666667],
       [0.5       , 0.66666667],
       [0.625     , 0.66666667],
       [0.75      , 0.66666667],
       [0.875     , 0.66666667],
       [1.        , 0.66666667],
       [0.        , 0.83333333],
       [0.125     , 0.83333333],
       [0.25      , 0.83333333],
       [0.375     , 0.83333333],
       [0.5       , 0.83333333],
       [0.625     , 0.83333333],
       [0.75      , 0.83333333],
       [0.875     , 0.83333333],
       [1.        , 0.83333333],
       [0.        , 1.        ],
       [0.125     , 1.        ],
       [0.25      , 1.        ],
       [0.375     , 1.        ],
       [0.5       , 1.        ],
       [0.625     , 1.        ],
       [0.75      , 1.        ],
       [0.875     , 1.        ],
       [1.        , 1.        ]])

    camera = Camera(focal_length=35.0, sensor_size=np.array([36., 24.]),
                    resolution=np.array([1200,  800]), skew=0.0)
    solve(corners, board, camera)
          


# def run_corner_refinement():
#     key = "OV/cube/ov00", "train", "ov00/0031.pgm"
#     with open("babelcalib_results.pkl", "rb") as f:
#         r: BenchmarkResult = next(
#             r
#             for r in pkl.load(f)
#             if (r.input.ds_name, r.input.subds_name, r.input.name) == key
#         )
#         assert isinstance(r.input, Entry)
#         assert r.input.image is not None
#         assert r.features is not None
#     show_boards(np.array(r.input.image), r.features.corners, r.features.board).show()


if __name__ == "__main__":
    # debug_test()
    run_benchmark()
    # run_corner_refinement()
