import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from concepts.linear_regression import LinReg
from math import isclose


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_X = np.array([1, 2, 4, 6, 7, 11]).reshape(-1, 1)
        self.mock_y = np.array([5, 9, 18, 25, 27, 40])
        self.mock_y_predict = np.array([6, 10, 17, 27, 29, 41])
        self.iterations = 1000
        self.learning_rate = 0.02
        self.lin_reg = LinearRegression()
        self.own_lin_reg = LinReg()
        self.own_lin_reg.fit(self.iterations, self.learning_rate, self.mock_X, self.mock_y)

    def test_lin_reg(self):
        expected = self.lin_reg.fit(self.mock_X, self.mock_y)
        
        self.own_lin_reg.fit(self.iterations, self.learning_rate, self.mock_X, self.mock_y)
        actual_coef = self.own_lin_reg.coefs
        actual_intercept = self.own_lin_reg.intercept

        assert isclose(expected.coef_, actual_coef, rel_tol=0.01)
        assert isclose(expected.intercept_, actual_intercept, rel_tol=0.01)

    def test_compute_gradients(self):
        errors = self.mock_y - self.mock_y_predict
        expected_gradients = -2 * np.mean(self.mock_X * errors[:, np.newaxis], axis=0)

        actual_gradients= self.own_lin_reg._compute_gradients(
            self.mock_X, self.mock_y, self.mock_y_predict
        )

        np.testing.assert_array_almost_equal(actual_gradients, expected_gradients, decimal=5)


    def test_prediction(self):
        new_x = 5

        expected = 3.4912744089864622 * new_x + 2.6284101176911387
        actual = self.own_lin_reg.predict(new_x)

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
