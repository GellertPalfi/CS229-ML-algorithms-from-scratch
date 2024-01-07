import unittest
from collections import namedtuple

import numpy as np
import sklearn

from algorithms.logistic_regression.binary_metrics import (
    _eval_classifier,
    accuracy,
    confusion_matrix,
    f1_score,
    precision,
    recall,
)


class TestClassificationMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.actual = np.array([1, 0, 1, 1, 0, 0, 1])
        self.predicted = np.array([1, 0, 0, 1, 0, 1, 1])

    def test_eval_classifier(self):
        metrics = namedtuple("Metrics", ["TP", "TN", "FP", "FN"])

        expected = metrics(3, 2, 1, 1)
        actual = _eval_classifier(self.actual, self.predicted)

        self.assertEqual(expected, actual)

    def test_accuracy(self):
        expected = sklearn.metrics.accuracy_score(self.actual, self.predicted)
        actual = accuracy(self.actual, self.predicted)

        self.assertEqual(expected, actual)

    def test_confusion_matrix(self):
        expected = sklearn.metrics.confusion_matrix(self.actual, self.predicted)
        actual = confusion_matrix(self.actual, self.predicted)

        self.assertTrue((expected == actual).all())

    def test_precision(self):
        expected = sklearn.metrics.precision_score(self.actual, self.predicted)
        actual = precision(self.actual, self.predicted)

        self.assertEqual(expected, actual)

    def test_recall(self):
        expected = sklearn.metrics.recall_score(self.actual, self.predicted)
        actual = recall(self.actual, self.predicted)

        self.assertEqual(expected, actual)

    def test_f1_score(self):
        expected = sklearn.metrics.f1_score(self.actual, self.predicted)
        actual = f1_score(self.actual, self.predicted)

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
