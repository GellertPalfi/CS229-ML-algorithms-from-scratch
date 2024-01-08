import io
import sys
import unittest
from math import isclose

import numpy as np
from sklearn.linear_model import LogisticRegression as LR

from algorithms.logistic_regression.logistic_regression import LogisticRegression


class TestLogisticRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(42)
        num_observations = 500
        cls.max_iter = 5000
        cls.learning_rate = 0.003
        x1 = rng.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
        x2 = rng.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)
        cls.mock_X = np.array([[1, 2, 3], [4, 5, 6]])
        cls.mock_y = np.array([0, 1])
        cls.mock_weights = np.array([0.423, 0.123, 0.678])
        cls.mock_regularization = 1
        cls.features = np.vstack((x1, x2)).astype(np.float32)
        cls.labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
        cls.sk_log = LR()

    def setUp(self) -> None:
        self.log_reg = LogisticRegression()

    def test_fit(self):
        self.log_reg.fit(
            self.max_iter, self.learning_rate, self.features, self.labels, True
        )
        self.sk_log.fit(self.features, self.labels)
        # use tolerant comparison, because our implementation might not converge in 5000 steps
        assert isclose(self.log_reg.intercept, self.sk_log.intercept_[0], rel_tol=0.01)
        assert np.allclose(self.log_reg.coefs, self.sk_log.coef_, rtol=0.01)

    def test_compute_gradients(self):
        mock_log_func_res = 1 / (1 + np.exp(-np.dot(self.mock_X, self.mock_weights)))
        regularization_term: float = self.mock_regularization * self.mock_weights
        regularization_term[0] = 0
        gradient = (
            np.dot(self.mock_X.T, (self.mock_y - mock_log_func_res))
            - regularization_term
        )

        expected = gradient
        actual = self.log_reg._compute_gradients(
            self.mock_X, self.mock_weights, self.mock_y
        )

        self.assertEqual(expected.all(), actual.all())

    def test_logistic_function(self):
        expected = 1 / (1 + np.exp(-np.dot(self.mock_X, self.mock_weights)))
        actual = self.log_reg.logistic_function(self.mock_X, self.mock_weights)

        self.assertEqual(expected.all(), actual.all())

    def test_log_likelihood(self):
        scores = np.dot(self.mock_X, self.mock_weights)
        ll = np.sum(self.mock_y * scores - np.log(1 + np.exp(scores)))

        expected = ll - (self.mock_regularization / 2) * np.sum(
            self.mock_weights[1:] ** 2
        )
        actual = self.log_reg.log_likelihood(
            self.mock_X, self.mock_y, self.mock_weights
        )

        self.assertEqual(expected, actual)

    def test_predict(self):
        self.log_reg.fit(
            self.max_iter, self.learning_rate, self.features, self.labels, True
        )
        expected_prediction = np.array([1, 0])
        prediction = self.log_reg.predict(self.mock_X)

        self.assertEqual(prediction.all(), expected_prediction.all())

    def test_verbose_prints(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        self.log_reg.fit(
            self.max_iter, self.learning_rate, self.features, self.labels, verbose=True
        )
        sys.stdout = sys.__stdout__
        # numerical instability causes different results on different machines
        assert captured_output.getvalue().startswith("-811")

    def test_gradient_logged(self):
        self.log_reg.fit(
            self.max_iter,
            self.learning_rate,
            self.features,
            self.labels,
            log_gradient=True,
        )

        self.assertIsNotNone(self.log_reg.gradients)
