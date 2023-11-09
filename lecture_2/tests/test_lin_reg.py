import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from concepts.linear_regression import lin_reg, compute_gradients, predict
from math import isclose


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_X = np.array([1, 2, 4, 6, 7, 11])
        self.mock_y = np.array([5, 9, 18, 25, 27, 40])
        self.mock_y_predict = np.array([6, 10, 17, 27, 29, 41])
        self.mock_b = 0
        self.mock_m = 0
        self.iterations = 1000
        self.learning_rate = 0.02
        self.lin_reg = LinearRegression()
        self.own_lin_reg, _ = lin_reg(
            self.iterations,
            self.learning_rate,
            self.mock_m,
            self.mock_b,
            self.mock_X,
            self.mock_y,
        )

    def test_lin_reg(self):
        reshaped_X = np.array(self.mock_X).reshape(-1, 1)
        expected = self.lin_reg.fit(reshaped_X, self.mock_y)

        actual_coef = self.own_lin_reg[-1][0]
        actual_intercept = self.own_lin_reg[-1][1]
        assert isclose(expected.coef_, actual_coef, rel_tol=0.01)

        assert isclose(expected.intercept_, actual_intercept, rel_tol=0.01)

    def test_compute_gradients(self):
        expected_grad_m = np.mean(
            -2 * self.mock_X * (self.mock_y - self.mock_y_predict)
        )
        expected_grad_b = np.mean(-2 * (self.mock_y - self.mock_y_predict))

        grad_m, grad_b = compute_gradients(
            self.mock_X, self.mock_y, self.mock_y_predict
        )

        self.assertAlmostEqual(grad_m, expected_grad_m)
        self.assertAlmostEqual(grad_b, expected_grad_b)

    def test_prediction(self):
        new_x = 5

        expected = 3.4912744089864622 * new_x + 2.6284101176911387

        actual_coef = self.own_lin_reg[-1][0]
        actual_intercept = self.own_lin_reg[-1][1]
        actual = predict(new_x, actual_coef, actual_intercept)

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
