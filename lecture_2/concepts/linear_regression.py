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

    def fit(
        self, iterations: int, learning_rate: float, X: ArrayLike, y: ArrayLike
    ) -> tuple[list, list]:
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
        theta = np.random.rand(X.shape[1] + 1) * 10
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        steps = []
        self.loss = []

        for _ in range(iterations):
            # predict y with current coeffs
            y_predicted: ArrayLike = np.dot(X, theta)
            # calculate and update by gradient
            gradients = self._compute_gradients(X, y, y_predicted)
            theta -= learning_rate * gradients
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

    def _compute_gradients(self, X: ArrayLike, y_actual: ArrayLike, y_predicted: ArrayLike) -> Tuple[float]:
        """Calculate gradient for MSE."""
        errors: ArrayLike = y_actual - y_predicted
        gradients: ArrayLike[float] = -2 * np.mean(X * errors[:, np.newaxis], axis=0)
        return gradients

    def predict(self, x_new: ArrayLike)-> ArrayLike[float]:
        """Predict new values with the trained linear regression model."""
        x_new = np.array(x_new)
        predicted: ArrayLike[float] = (
            np.dot(self.coefs, x_new) + self.intercept
            if x_new.ndim == 1
            else np.dot(x_new, self.coefs) + self.intercept
        )

        return predicted


if __name__ == "__main__":
    # running this script your results may differ because of random theta initalitaziation
    iterations = 10000
    learning_rate = 0.003

    # example usage 1D
    y = np.array([5, 9, 18, 25, 27, 40])
    x = np.array([1, 2, 6, 10, 11, 14]).reshape(-1, 1)

    own_reg = LinReg()
    own_reg.fit(iterations, learning_rate, x, y)

    reg = LinearRegression()
    reg.fit(x, y)

    print(own_reg.coefs, own_reg.intercept)  # [0.8546798] 0.3990147783251276
    print(reg.coef_, reg.intercept_, "\n")  # [0.8546798] 0.3990147783251228

    # example usage 2D
    x = np.array([[0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 25]])

    own_reg = LinReg()
    own_reg.fit(iterations, learning_rate, x, y)

    reg = LinearRegression()
    reg.fit(x, y)

    print(
        own_reg.coefs, own_reg.intercept
    )  # [1.95327313 0.08933628] 0.9646065741404198
    print(reg.coef_, reg.intercept_)  # [1.95357143 0.08928571] 0.9642857142857162

    new_x = np.array([[6,2],[2,3]])
    print(reg.coef_, reg.intercept_)
    print(
        reg.predict(new_x)
    )  # [12.86428571  5.13928571]
    print(
        own_reg.predict(new_x)
    )  # [12.84980966  5.13797303]