import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from algorithms.support_vector_machine.svm import SVM


class TestSVM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.60)
        y = np.where(y <= 0, -1, 1)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, random_state=42
        )

    def test_svm_fit(self):
        svm = SVM()
        svm.fit(self.X_train, self.y_train)
        svc = SVC(kernel="linear")
        svc.fit(self.X_train, self.y_train)

        sklearn_accuracy_score = accuracy_score(svc.predict(self.X_test), self.y_test)
        own_accuracy_score = accuracy_score(svm.predict(self.X_test), self.y_test)

        self.assertAlmostEqual(sklearn_accuracy_score, own_accuracy_score)

    def test_svm_predict(self):
        svm = SVM()
        svm.fit(self.X_train, self.y_train)
        example_x = [[2.04585825e00, 9.94220561e-01]]

        actual = svm.predict(example_x)
        expected = [1]

        self.assertEqual(expected, actual)

    def test_hingeloss(self):
        svm = SVM()
        svm.w = [0.30429077, -1.00464207]
        svm.b = -1.9899999999998916

        actual = svm.hingeloss(self.X_train, self.y_train)
        expected = 0.02511028345

        self.assertAlmostEqual(expected, actual)
