import numpy as np
from numpy.typing import ArrayLike
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class SVM:
    """Support vector machine model using batch gradient descent and Linear Kernel.

    Args:
        alpha: The step size for each iteration of gradient descent.
        lambda_: L2 regularization term. Defaults to 0.01.
        n_iterations: Number of iterations to train the model for. Defaults to 1000.
    """

    def __init__(
        self, alpha: float = 0.001, lambda_: float = 0.01, n_iterations: int = 1000
    ) -> None:
        self.alpha = alpha
        self.lambda_ = lambda_
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        self.loss = []

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Calculate optimal SVM parameters using batch gradient descent.

        Args:
            X (array-like): The input feature values.
            y (array-like): The correct label for the feature values.
        """
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            self.loss.append(self.hingeloss(X, y))

            for i, Xi in enumerate(X):
                # check whether the point is correctly classified
                # and outside the margin:
                if y[i] * (np.dot(Xi, self.w) - self.b) >= 1:
                    self.w -= self.alpha * (2 * self.lambda_ * self.w)
                else:
                    self.w -= self.alpha * (
                        2 * self.lambda_ * self.w - np.dot(Xi, y[i])
                    )
                    self.b -= self.alpha * y[i]

    def hingeloss(self, X: ArrayLike, y: ArrayLike) -> float:
        """Calculate the hinge loss."""
        hinge_loss = np.maximum(0, 1 - y * (np.dot(X, self.w) - self.b)).mean()
        regularization_loss = self.lambda_ * np.dot(self.w, self.w)
        return hinge_loss + regularization_loss

    def predict(self, X: ArrayLike):
        """Predict new values with the trained logistic regression model."""
        pred = np.dot(X, self.w) - self.b
        # returning in the form of -1 and 1
        return [1 if val > 0 else -1 for val in pred]


# Example usage
if __name__ == "__main__":
    X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.60)
    y = np.where(y <= 0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print(X_test)
    svm = SVM()
    svm.fit(X_train, y_train)
    svc = SVC(kernel="linear")
    svc.fit(X_train, y_train)
    svc.predict(X_test)  # Accuracy: 1.0
    prediction = svm.predict(X_test)
    print("Accuracy:", accuracy_score(prediction, y_test))  # Accuracy: 1.0
