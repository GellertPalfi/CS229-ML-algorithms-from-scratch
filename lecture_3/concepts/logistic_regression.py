import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from lecture_3.concepts.binary_metrics import accuracy


class LogisticRegression:
    def __init__(self) -> None:
        self.intercept = 0
        self.coefs = []
        self.loss = []

    def fit(self, iterations, alpha, X, y, intercept=False):
        if intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        b = np.zeros(X.shape[1])
        for _ in range(iterations):
            grad = self._compute_gradients(X, b, y)
            b += alpha * grad

            log_likelihood = self.log_likelihood(X, y, b)
            self.loss.append(log_likelihood)

        self.intercept = b[0]  # The first element of b is the intercept
        self.coefs = b[1:]

    def _compute_gradients(self, X, b, y):
        probabilities = self.logistic_function(X, b)
        gradient = np.dot(X.T, (y - probabilities))
        return gradient

    def logistic_function(self, X, b):
        return 1 / (1 + np.exp(-np.dot(X, b)))

    def log_likelihood(self, features, target, weights):
        scores = np.dot(features, weights)
        return np.sum(target * scores - np.log(1 + np.exp(scores)))

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Compute probabilities
        probabilities = self.logistic_function(self.coefs, X)

        # Convert probabilities to class labels
        predictions = (probabilities >= 0.5).astype(int)
        return predictions


if __name__ == "__main__":
    np.random.seed(42)
    num_observations = 5000

    x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)
    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    simulated_labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
    data_with_intercept = np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),
                                 simulated_separableish_features))
    
    max_iter = 3000
    learning_rate = 0.003

    log_reg = LogisticRegression()
    log_reg.fit(
        max_iter, learning_rate, simulated_separableish_features, simulated_labels, True
    )
    print(log_reg.intercept, log_reg.coefs)


    sk_log = LR(penalty=None)
    sk_log.fit(simulated_separableish_features, simulated_labels)
    print(sk_log.intercept_, sk_log.coef_)

    weights = np.append(np.array(log_reg.coefs).flatten(), log_reg.intercept)

    final_scores = np.dot(data_with_intercept, weights)
    preds = np.round(sigmoid(final_scores))
