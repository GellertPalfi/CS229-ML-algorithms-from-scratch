from collections import namedtuple

import numpy as np
from numpy.typing import ArrayLike


def accuracy(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Calculate accuracy of a binary classifier."""
    metrics = _eval_classifier(actual, predicted)

    return (metrics.TP + metrics.TN) / (
        metrics.TP + metrics.TN + metrics.FP + metrics.FN
    )


def confusion_matrix(actual: ArrayLike, predicted: ArrayLike) -> dict[str, int]:
    """Calculate confusion matrix of a binary classifier."""
    metrics = _eval_classifier(actual, predicted)

    return np.array([[metrics.TN, metrics.FN], [metrics.FP, metrics.TP]])


def precision(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Calculate precision of a binary classifier.

    Out of all positive predictions, how many are actually positive?
    """
    metrics = _eval_classifier(actual, predicted)

    return metrics.TP / (metrics.TP + metrics.FP)


def recall(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Calculate recall of a binary classifier.

    Out of all actual positive cases,
    how many did the model correctly identify as positive?
    """
    metrics = _eval_classifier(actual, predicted)

    return metrics.TP / (metrics.TP + metrics.FN)


def f1_score(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Calculate f1 score of a binary classifier.

    Harmonic mean of precision and recall.
    """
    metrics = _eval_classifier(actual, predicted)

    return 2 * metrics.TP / (2 * metrics.TP + metrics.FP + metrics.FN)


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
