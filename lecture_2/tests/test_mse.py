import unittest

import numpy as np
import sklearn
from lecture_2.concepts.mean_squared_error import mean_squared_error as own_mse


class TestMse(unittest.TestCase):
    def setUp(self) -> None:
        self.test_array1 = np.array([10, 12, 15, 16])
        self.test_array2 = np.array([11, 13, 16, 18])

    def test_mse(self):
        expected_res = sklearn.metrics.mean_squared_error(
            self.test_array1, self.test_array2, squared=True
        )
        actual_res = own_mse(self.test_array1, self.test_array2)

        self.assertEqual(expected_res, actual_res)

    def test_rmse(self):
        expected_res = sklearn.metrics.mean_squared_error(
            self.test_array1, self.test_array2, squared=False
        )

        actual_res = own_mse(self.test_array1, self.test_array2, squared=False)
        self.assertEqual(expected_res, actual_res)


if __name__ == "__main__":
    unittest.main()
