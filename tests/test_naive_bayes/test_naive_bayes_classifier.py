import unittest

import polars as pl
import scipy

from algorithms.naive_bayes.naive_bayes import NaiveBayes


class TestPredict(unittest.TestCase):
    def setUp(self) -> None:
        classes = ["setosa", "versicolor", "virginica"]
        means = pl.from_dict(
            {
                "Species": ["setosa", "versicolor", "virginica"],
                "SepalLengthCm": [5.006, 5.936, 6.588],
                "SepalWidthCm": [3.428, 2.77, 2.974],
                "PetalLengthCm": [1.462, 4.26, 5.552],
                "PetalWidthCm": [0.246, 1.326, 2.026],
            }
        )
        var = pl.from_dict(
            {
                "Species": ["setosa", "versicolor", "virginica"],
                "SepalLengthCm": [
                    0.12424897959183673,
                    0.2664326530612246,
                    0.4043436734693877,
                ],
                "SepalWidthCm": [
                    0.1436897959183674,
                    0.09846938775510205,
                    0.10400408163265307,
                ],
                "PetalLengthCm": [
                    0.030106122448979587,
                    0.22081632653061237,
                    0.3045877551020408,
                ],
                "PetalWidthCm": [
                    0.01149387755102041,
                    0.03910612244897959,
                    0.07543265306122449,
                ],
            }
        )
        prior = pl.from_dict(
            {
                "Species": ["setosa", "versicolor", "virginica"],
                "Prior": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            }
        )
        self.naive_bayes = NaiveBayes(means, var, prior, classes)

    def test_predict_setosa(self):
        instance = {
            "SepalLengthCm": [5.1],
            "SepalWidthCm": [3.5],
            "PetalLengthCm": [1.4],
            "PetalWidthCm": [0.2],
        }
        expected_res = "setosa"
        actual_res = self.naive_bayes.predict(instance)
        self.assertEqual(expected_res, actual_res)

    def test_predict_versicolor(self):
        instance = {
            "SepalLengthCm": [6.0],
            "SepalWidthCm": [2.9],
            "PetalLengthCm": [4.5],
            "PetalWidthCm": [1.5],
        }
        expected_res = "versicolor"
        actual_res = self.naive_bayes.predict(instance)
        self.assertEqual(expected_res, actual_res)

    def test_predict_virginica(self):
        instance = {
            "SepalLengthCm": [6.9],
            "SepalWidthCm": [3.1],
            "PetalLengthCm": [5.4],
            "PetalWidthCm": [2.1],
        }
        expected_res = "virginica"
        actual_res = self.naive_bayes.predict(instance)
        self.assertEqual(expected_res, actual_res)

    def test_pdf(self):
        x = 5.1
        mean = 5.06
        var = 0.78
        expected_res = scipy.stats.norm(mean, var).pdf(x)
        actual_res = self.naive_bayes.pdf(x, mean, var)
        self.assertAlmostEqual(expected_res, actual_res)


if __name__ == "__main__":
    unittest.main()
