import numpy as np


def mean_squared_error(actual, predicted, squared=False):
    mse = np.mean((np.power(np.subtract(actual, predicted), 2)))
    rmse = np.sqrt(mse)

    return rmse if squared else mse
