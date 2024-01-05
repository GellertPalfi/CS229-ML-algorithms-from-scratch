import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression

from algorithms.linear_regression.mean_squared_error import mean_squared_error


class LinReg:
    """Linear regression model using batch gradient descent."""

    def __init__(self) -> None:
        self.intercept = 0
        self.coefs = []
        self.loss = []

    def fit(
        self, iterations: int, alpha: float, X: ArrayLike, y: ArrayLike
    ) -> list[float]:
        """Calculate optimal linear regression parameters using batch gradient descent.

        This is a really primitive implementation of linear regression and
        early stopping.

        Args:
            iterations: Number of iterations to train the model for.
            alpha: The step size for each iteration of gradient descent.
            X (array-like): The input feature values.
            y (array-like): The true output values corresponding to each input.
                Should be same length as X.

        Returns:
            steps: List containing m and b in tuples for every iteration.
        """
        # initalize 0 coeffs and intercept
        # add fake columns of 1 for better intercept fit
        rng = np.random.default_rng(42)
        theta = rng.random(X.shape[1] + 1) * 10
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        steps = []
        self.loss = []

        for _ in range(iterations):
            # predict y with current coeffs
            y_predicted: ArrayLike = np.dot(X, theta)
            # calculate and update by gradient
            gradients = self._compute_gradients(X, y, y_predicted)
            theta -= alpha * gradients
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

    def _compute_gradients(
        self, X: ArrayLike, y_actual: ArrayLike, y_predicted: ArrayLike
    ) -> tuple[float]:
        """Calculate gradient for MSE."""
        errors: ArrayLike = y_actual - y_predicted
        gradients: ArrayLike[float] = -2 * np.mean(X * errors[:, np.newaxis], axis=0)
        return gradients

    def predict(self, x_new: ArrayLike) -> ArrayLike:
        """Predict new values with the trained linear regression model."""
        x_new = np.array(x_new)
        predicted: ArrayLike[float] = (
            np.dot(self.coefs, x_new) + self.intercept
            if x_new.ndim == 1
            else np.dot(x_new, self.coefs) + self.intercept
        )

        return predicted


# Example usage
if __name__ == "__main__":
    iterations = 10000
    alpha = 0.003  # learning rate

    # example usage 1D
    y = np.array([5, 9, 18, 25, 27, 40])
    x = np.array([1, 2, 6, 10, 11, 14]).reshape(-1, 1)

    own_reg = LinReg()
    own_reg.fit(iterations, alpha, x, y)

    reg = LinearRegression()
    reg.fit(x, y)

    print(own_reg.coefs, own_reg.intercept)  # [2.42857143] 2.8571428618554884
    print(reg.coef_, reg.intercept_, "\n")  # [2.42857143] 2.8571428571428577

    # example usage 2D
    x = np.array([[0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 25]])

    own_reg = LinReg()
    own_reg.fit(iterations, alpha, x, y)

    reg = LinearRegression()
    reg.fit(x, y)

    print(own_reg.coefs, own_reg.intercept)  # [5.22429202 0.30369145] 4.8221900979826
    print(reg.coef_, reg.intercept_, "\n")  # [5.225 0.30357143] 4.8214285714285765

    new_x = np.array([[6, 2], [2, 3]])
    print(reg.predict(new_x))  # [36.77857143 16.18214286]
    print(own_reg.predict(new_x))  # [36.77532513 16.18184848
