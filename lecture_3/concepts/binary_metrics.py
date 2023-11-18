import numpy as np
from numpy.typing import ArrayLike
from collections import namedtuple


def accuracy(actual: ArrayLike, predicted: ArrayLike) -> float:
    metrics = _eval_classifier(actual, predicted)

    return (metrics.TP + metrics.TN) / (
        metrics.TP + metrics.TN + metrics.FP + metrics.FN
    )


def confusion_matrix(actual, predicted) -> dict[str, int]:
    metrics = _eval_classifier(actual, predicted)

    return np.array([[metrics.TN, metrics.FN], [metrics.FP, metrics.TP]])


def _eval_classifier(actual: ArrayLike, predicted: ArrayLike) -> dict[str, int]:
    metrics = namedtuple("Metrics", ["TP", "TN", "FP", "FN"])
    TP = np.sum(np.logical_and(actual == 1, predicted == 1))
    TN = np.sum(np.logical_and(actual == 0, predicted == 0))
    FP = np.sum(np.logical_and(actual == 0, predicted == 1))
    FN = np.sum(np.logical_and(actual == 1, predicted == 0))

    return metrics(TP, TN, FP, FN)


if __name__ == "__main__":
    actual = np.array([1, 0, 1, 1, 0, 0, 1])
    predicted = np.array([1, 0, 0, 1, 0, 1, 1])

    print(accuracy(actual, predicted))
    print(confusion_matrix(actual, predicted))
