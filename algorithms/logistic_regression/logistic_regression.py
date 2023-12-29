from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score

from algorithms.logistic_regression.binary_metrics import accuracy


class LogisticRegression:
    """Logistic regression model using batch gradient descent.

    Args:
        lambda_reg: L2 regularization term. Defaults to 1.
    """

    def __init__(self, lambda_reg=1) -> None:
        self.intercept: float = 0
        self.coefs = []
        self.loss = []
        self.weights = []
        self.gradients = []
        self.lambda_reg = lambda_reg

    def fit(
        self,
        iterations: int,
        alpha: float,
        X: ArrayLike,
        y: ArrayLike,
        intercept: bool = False,
        log_gradient: bool = False,
        verbose: bool = False,
    ) -> None:
        """Calculate optimal logistic regression parameters using batch gradient descent.

        Args:
            iterations: Number of iterations to train the model for.
            alpha: The step size for each iteration of gradient descent.
            X (array-like): The input feature values.
            y (array-like): The correct label for the feature values.
            intercept: Whether to fit an intercept term. Defaults to False.
            log_gradient: Whether to log the gradient at each iteration.
                Defaults to False.
            verbose: Whether to print the loss at every 100 iteration.
              Defaults to False.
        """

        if intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        # initalize weights
        b = np.zeros(X.shape[1])

        for index in range(iterations):
            grad = self._compute_gradients(X, b, y)
            # loglikelihood is a maximization problem, so we add the gradient
            # instead of subtracting it
            b += alpha * grad
            if log_gradient:
                self.gradients.append(grad)

            # calculate and log loss
            log_likelihood = self.log_likelihood(X, y, b)
            self.loss.append(log_likelihood)

            if verbose and index % 100 == 0:
                print(log_likelihood)

        # save weights
        self.intercept: float = b[0]
        self.coefs = b[1:]

    def _compute_gradients(self, X: ArrayLike, b: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Calculate gradient for log likelihood with l2 regularization."""
        probabilities = self.logistic_function(X, b)
        regularization_term: float = self.lambda_reg * b
        # No regularization for intercept
        regularization_term[0] = 0
        gradient = np.dot(X.T, (y - probabilities)) - regularization_term

        return gradient

    def logistic_function(self, X: ArrayLike, b: ArrayLike) -> float:
        """Sigmoid function."""
        return 1 / (1 + np.exp(-np.dot(X, b)))

    def log_likelihood(
        self, features: ArrayLike, target: ArrayLike, weights: ArrayLike
    ) -> float:
        """Calculate log likelihood of the model."""
        scores: ArrayLike = np.dot(features, weights)
        ll: float = np.sum(target * scores - np.log(1 + np.exp(scores)))

        # Add the regularization term (excluding the intercept)
        ll -= (self.lambda_reg / 2) * np.sum(weights[1:] ** 2)
        return ll

    def predict(self, X) -> Literal[0, 1]:
        """Predict new values with the trained logistic regression model."""
        self.weights = np.append(self.intercept, np.array(self.coefs).flatten())
        return np.round(self.logistic_function(X, self.weights))


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    num_observations = 5000

    x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)
    features = np.vstack((x1, x2)).astype(np.float32)
    labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
    data_with_intercept = np.hstack((np.ones((features.shape[0], 1)), features))
    max_iter = 500
    learning_rate = 0.003

    log_reg = LogisticRegression()
    log_reg.fit(max_iter, learning_rate, features, labels, True)
    own_predict = log_reg.predict(data_with_intercept)

    sk_log = LR()
    sk_log.fit(features, labels)
    predicted = sk_log.predict(features)
    # the algorithm has not converged yet, but given enough time and small enough steps,
    # gradient descent on a concave function will always reach the global optimum
    print(
        log_reg.coefs, log_reg.intercept
    )  # [-3.79171593  6.3024119 ] -10.758687874923133
    print(sk_log.coef_, sk_log.intercept_)  # [[-3.66639864  6.09359415]] [-10.38791725]

    print(accuracy(own_predict, labels))  # 0.994
    print(accuracy_score(predicted, labels))  # 0.9939
