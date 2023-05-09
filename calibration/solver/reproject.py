import numpy as np


def reprojectpoints(x, X, lambdas, R, t):
    pass
    # XX = 
    # err = []
    # stderr = []
    # x.shape[0]
    # MSE = 0
    #
    # xx = np.dot(R, np.vstack((X.T, np.ones(X.T.shape)))) + t
    # for i in range(R.shape[2]):
    #     counterr += 1
    #
    #     # Assuming omni3d2pixel() function is defined
    #     Xp_reprojected, Yp_reprojected = omni3d2pixel(lambdas, xx)
    #
    #     stt = np.sqrt(
    #         (x[:, 0] - Xp_reprojected) ** 2 + (x[:, 1] - Yp_reprojected) ** 2
    #     )
    #     err.append(np.mean(stt))
    #     stderr.append(np.std(stt))
    #     return np.sum(
    #         (x[:, 0] - Xp_reprojected) ** 2 + (x[:, 1] - Yp_reprojected) ** 2
    #     )
    #
