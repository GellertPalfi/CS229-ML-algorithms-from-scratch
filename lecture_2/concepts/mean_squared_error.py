import numpy as np


def mean_squared_error(actual, predicted, squared=True) -> float:
    mse: float = np.mean((np.power(np.subtract(actual, predicted), 2)))
    rmse: float = np.sqrt(mse)

    return mse if squared else rmse
