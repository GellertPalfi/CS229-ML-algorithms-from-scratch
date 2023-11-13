import numpy as np
from numpy.typing import ArrayLike
from concepts.mean_squared_error import mean_squared_error
from typing import Tuple

from sklearn.linear_model import LinearRegression

class LinReg:
    def __init__(self) -> None:
        self.intercept = 0
        self.coefs = []
        self.loss = []

    def fit(self, iterations: int, learning_rate: float, X: ArrayLike, y: ArrayLike) -> tuple[list, list]:
        """Calculate optimal linear regression parameters using batch gradient descent.

        This is a really primitive implementation of linear regression and
        early stopping.

        Args:
            iterations: Number of iterations to train the model for.
            learning_rate: The step size for each iteration of gradient descent.
            X (array-like): The input feature values.
            y (array-like): The true output values corresponding to each input.
                Should be same length as X.

        Returns:
            steps: List containing m and b in tuples for every iteration.
            loss: List containing MSE for every iteration.
        """
        # initalize 0 coeffs and intercept
        # add fake columns of 1 for better intercept fit

        theta = np.random.rand(X.shape[1] + 1)*10
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        steps = []
        self.loss = []

        for _ in range(iterations):
            # predict y with current coeffs
            y_predicted = np.dot(X, theta)
            # calculate and update by gradient
            gradients = self._compute_gradients(X, y, y_predicted)
            theta -= learning_rate * gradients
            print(theta)
            mse = mean_squared_error(y, y_predicted)

            # primtive early stopping
            if np.isnan(mse) or np.isinf(mse):
                break

            # log metrics
            steps.append(np.copy(theta))
            self.loss.append(mse)


        self.intercept = steps[-1][0]
        self.coefs = steps[-1][1:]
        return steps


    def _compute_gradients(self, X, y_actual, y_predicted) -> Tuple[float]:
        """Calculate gradient for MSE."""
        errors = y_actual - y_predicted
        gradients = -2 * np.mean(X * errors[:, np.newaxis], axis=0)
        return gradients



    def predict(self, x_new: ArrayLike):
        """Predict new values with the trained linear regression model."""
        x_new = np.array(x_new)

        if x_new.ndim == 1:
            return np.dot(self.coefs, x_new) + self.intercept
        else:
            return np.dot(x_new, self.coefs) + self.intercept
         



if __name__ == "__main__":
    iterations = 1000
    learning_rate = 0.01

    # example usage 1D
    y = np.array([1, 3, 5, 8, 10, 13])
    x = np.array([1, 2, 6, 10, 11, 14]).reshape(-1,1)
    
    own_reg = LinReg()
    own_reg.fit(iterations, learning_rate, x, y)

    reg = LinearRegression()
    reg.fit(x,y)
    
    print(own_reg.coefs, own_reg.intercept) # [0.5] 1.8763348981580675e-09
    print(reg.coef_, reg.intercept_) # [0.5] -1.7763568394002505e-15
    raise
    new_x = np.ndarray([6]).reshape(-1,1)
    print(reg.predict(new_x)) # [-2.83333333 -1.83333333 -0.83333333  0.66666667  1.66666667  3.16666667]
    print(own_reg.predict(new_x)) # [-2.83333333 -1.83333333 -0.83333333  0.66666667  1.66666667  3.16666667]

    # example usage 2D
    x = np.array([[ 0,  0],
       [ 1,  1],
       [ 2,  4],
       [ 3,  9],
       [ 4, 16],
       [ 5, 25]])
    
    
    own_reg = LinReg()
    own_reg.fit(iterations, learning_rate, x, y)

    reg= LinearRegression()
    reg.fit(x,y)
    
    print(own_reg.coefs, own_reg.intercept) # [1.95327313 0.08933628] 0.9646065741404198
    print(reg.coef_, reg.intercept_) # [1.95357143 0.08928571] 0.9642857142857162


    new_x = np.ndarray([6,2])
    print(reg.predict(new_x)) # [-4.73809524 -2.6952381  -0.47380952  1.92619048  4.5047619   7.26190476]
    print(own_reg.predict(new_x)) # [-4.73749218 -2.69488276 -0.47360078  1.92635376  4.50498087  7.26228054]
