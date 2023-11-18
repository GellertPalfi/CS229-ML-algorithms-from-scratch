import numpy as np
from sklearn.linear_model import LogisticRegression as LR


class LogisticRegression:
    def __init__(self) -> None:
        self.intercept = 0
        self.coefs = []
        self.loss = []

    def fit(self, iterations, alpha, b, X, y):
        for _ in range(iterations):
            grad = self._compute_gradients(b, X, y)
            b -= alpha * grad

            log_likelihood = self.compute_log_likelihood(b, X, y)
            self.loss.append(log_likelihood)
            # print(log_likelihood)

        self.coefs = b

    def _compute_gradients(self, b, X, y):
        probabilities = self.logistic_function(b, X)
        gradient = np.dot(X.T, (y - probabilities))
        return gradient

    def logistic_function(self, b, X):
        return 1 / (1 + np.exp(-np.dot(X, b)))

    def compute_log_likelihood(self, b, X, y):
        probabilities = self.logistic_function(b, X)
        log_likelihood = np.sum(
            y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)
        )
        return log_likelihood

    def predict():
        pass


model = LogisticRegression(learning_rate=0.1, num_iterations=300000)


X_original = np.array([[2, 3], [4, 1], [5, 6], [7, 8]])

# Add a column of ones for the intercept
X = np.hstack([np.ones((X_original.shape[0], 1)), X_original])

# Initialize coefficients (including intercept)
b = np.zeros(X.shape[1])
print(b)
y = np.array([0, 0, 1, 1])
xd = np.array([5, 6, 1]).reshape(1, -1)
max_iter = 100000
learning_rate = 0.1


log_reg = LogisticRegression()
log_reg.fit(max_iter, learning_rate, b, X, y)
print(log_reg.coefs)
sk_log = LR(penalty=None)
sk_log.fit(X, y)
print(sk_log.predict(xd))
print(sk_log.coef_, sk_log.intercept_)
