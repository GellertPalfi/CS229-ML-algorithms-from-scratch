import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as conf_mat
from lecture_3.concepts.binary_metrics import accuracy, confusion_matrix



class LogisticRegression:
    def __init__(self) -> None:
        self.intercept = 0
        self.coefs = []
        self.loss = []
        self.weights = []
        self.gradients = []
        self.lambda_reg = 1


    def fit(self, iterations, alpha, X, y, intercept=False, log_gradient=False) -> None:
        if intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        b = np.zeros(X.shape[1])
        for index in range(iterations):
            grad = self._compute_gradients(X, b, y)
            b += alpha * grad

            if log_gradient:
                self.gradients.append(grad)

            log_likelihood = self.log_likelihood(X, y, b)
            self.loss.append(log_likelihood)

            if index % 200 == 0:
                print(log_likelihood)

        self.intercept = b[0]
        self.coefs = b[1:]

    def _compute_gradients(self, X, b, y):
        probabilities = self.logistic_function(X, b)
        regularization_term = self.lambda_reg * b
        regularization_term[0] = 0  # No regularization for intercept
        gradient = np.dot(X.T, (y - probabilities)) - regularization_term
        return gradient

    def logistic_function(self, X, b):
        return 1 / (1 + np.exp(-np.dot(X, b)))

    def log_likelihood(self, features, target, weights):
        scores = np.dot(features, weights)
        ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
        # Add the regularization term (excluding the intercept)
        ll -= (self.lambda_reg / 2) * np.sum(weights[1:] ** 2)
        return ll

    def predict(self, X):
        self.weights = np.append(self.intercept, np.array(self.coefs).flatten())
        return np.round(self.logistic_function(X, self.weights))

if __name__ == "__main__":

    np.random.seed(1)
    num_observations = 5000

    x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)
    features = np.vstack((x1, x2)).astype(np.float32)
    labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
    data_with_intercept = np.hstack((np.ones((features.shape[0], 1)),
                                 features))
    max_iter = 500
    learning_rate = 0.003

    log_reg = LogisticRegression()
    log_reg.fit(
        max_iter, learning_rate, features, labels, True
    )
    own_predict = log_reg.predict(data_with_intercept)

    sk_log = LR()
    sk_log.fit(features, labels)
    predicted= sk_log.predict(features)
    print(log_reg.coefs, log_reg.intercept)
    print(sk_log.coef_, sk_log.intercept_)

    print(accuracy(own_predict, labels))
    print(accuracy_score(predicted, labels))
    print(confusion_matrix(own_predict, labels))
    print(conf_mat(labels, own_predict))
    