import sys
import unittest
from math import isclose
from pathlib import Path

import numpy as np
import sklearn

# this is needed to import modules from parent directory
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from algorithms.linear_regression.linear_regression import LinReg  # noqa: E402


class TestLinReg(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mock_X_1D = np.array([1, 2, 4, 6, 7, 11]).reshape(-1, 1)
        cls.mock_X_2D = np.array([[0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 25]])
        cls.mock_y = np.array([5, 9, 18, 25, 27, 40])
        cls.mock_y_predict = np.array([6, 10, 17, 27, 29, 41])
        cls.iterations = 5000
        cls.learning_rate = 0.005
        cls.lin_reg = sklearn.linear_model.LinearRegression()

    def setUp(self) -> None:
        self.own_lin_reg_1D = LinReg()
        self.own_lin_reg_2D = LinReg()

    def test_lin_reg_1d(self):
        self.own_lin_reg_1D.fit(
            self.iterations, self.learning_rate, self.mock_X_1D, self.mock_y
        )
        expected = self.lin_reg.fit(self.mock_X_1D, self.mock_y)

        actual_coef = self.own_lin_reg_1D.coefs
        actual_intercept = self.own_lin_reg_1D.intercept

        # avoid numpy deprecation warning with element extract
        assert isclose(expected.coef_[0], actual_coef[0], rel_tol=0.01)
        assert isclose(expected.intercept_, actual_intercept, rel_tol=0.01)

    def test_lin_reg_2d(self):
        self.own_lin_reg_2D.fit(
            self.iterations, self.learning_rate, self.mock_X_2D, self.mock_y
        )
        expected = self.lin_reg.fit(self.mock_X_2D, self.mock_y)

        actual_coef = self.own_lin_reg_2D.coefs
        actual_intercept = self.own_lin_reg_2D.intercept

        assert np.allclose(expected.coef_, actual_coef, rtol=0.01)
        assert np.allclose(expected.intercept_, actual_intercept, rtol=0.01)

    def test_compute_gradients(self):
        errors = self.mock_y - self.mock_y_predict
        expected_gradients = -2 * np.mean(
            self.mock_X_1D * errors[:, np.newaxis], axis=0
        )

        actual_gradients = self.own_lin_reg_1D._compute_gradients(
            self.mock_X_1D, self.mock_y, self.mock_y_predict
        )

        np.testing.assert_array_almost_equal(
            actual_gradients, expected_gradients, decimal=5
        )

    def test_prediction_1d(self):
        self.own_lin_reg_1D.fit(
            self.iterations, self.learning_rate, self.mock_X_1D, self.mock_y
        )
        new_x = 5

        expected = 3.4912744089864622 * new_x + 2.6284101176911387
        actual = self.own_lin_reg_1D.predict(new_x)[0]

        self.assertAlmostEqual(expected, actual, 3)

    def test_prediction_2d(self):
        self.own_lin_reg_2D.fit(
            self.iterations, self.learning_rate, self.mock_X_2D, self.mock_y
        )
        new_x = np.array([[6, 2], [2, 3]])
        expected = 5.225 * new_x[0] + 0.30375284 * new_x[1] + 4.8214285714285765
        actual = self.own_lin_reg_2D.predict(new_x)

        assert np.allclose(expected, actual, 3)


if __name__ == "__main__":
    unittest.main()
