import numpy as np
from concepts.mean_squared_error import mean_squared_error


def lin_reg(iterations: int, learning_rate: float, m: float, b: float):
    """Gradient descent implementation for linear regression using mse."""
    steps = []
    loss = []
    for _ in range(iterations):
        y_pred = np.array([m * x]) + b
        mse = mean_squared_error(y, y_pred)
        grad_m, grad_b = compute_gradients(x, y, y_pred)
        m -= learning_rate * grad_m
        b -= learning_rate * grad_b
        steps.append((m, b))
        loss.append(mse)
    return steps, loss


def compute_gradients(X, y_true, y_pred):
    """Calculate gradient for MSE."""
    error = y_true - y_pred
    grad_m = np.mean(-2 * (X * error))  # Gradient for the slope (m)
    grad_b = np.mean(-2 * error)  # Gradient for the intercept (b)
    return grad_m, grad_b


x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1, 3, 5, 8, 10, 13])
m = 0
b = 0
start = 10

lin_reg(50, 0.01, m, b)
