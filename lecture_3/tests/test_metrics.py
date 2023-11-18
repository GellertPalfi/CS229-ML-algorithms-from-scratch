import unittest
import numpy as np
from lecture_3.concepts.binary_metrics import accuracy, confusion_matrix
import sklearn


class TestConfusionMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.actual = np.array([1, 0, 1, 1, 0, 0, 1])
        self.predicted = np.array([1, 0, 0, 1, 0, 1, 1])

    def test_accuracy(self):
        expected = sklearn.metrics.accuracy_score(self.actual, self.predicted)
        actual = accuracy(self.actual, self.predicted)

        self.assertEqual(expected, actual)

    def test_confusion_matrix(self):
        expected = sklearn.metrics.confusion_matrix(self.actual, self.predicted)
        actual = confusion_matrix(self.actual, self.predicted)

        self.assertTrue((expected == actual).all())


if __name__ == "__main__":
    unittest.main()
