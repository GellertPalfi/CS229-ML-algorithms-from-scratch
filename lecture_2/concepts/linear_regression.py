import numpy as np
from concepts.mean_squared_error import mean_squared_error
from typing import Tuple

def lin_reg(iterations: int, learning_rate: float, m: float, b: float, X, y):
    """Calculate optimal linear regression parameters using batch gradient descent.
    
    Args:
        iterations: Number of iterations to train the model for.
        learning_rate: The step size for each iteration of gradient descent.
        m: The initial value of the slope of the regression line.
        b: The initial value of the intercept of the regression line.
        X (array-like): The input feature values.
        y (array-like): The true output values corresponding to each input.
            Should be same length as X.
    
    Returns:
        steps: List containing m and b in tuples for every iteration.
        loss: List containing MSE for every iteration.
    
    Raises:
        ValueError: If X and y are not the same length.
    """        

    if len(X) != len(y):
        raise ValueError("X and y are different lengths.")

    # every element of steps except the last one and loss are only 
    # for visualization purposes
    # the last tuple in steps contained the final values for m and b
    steps = []
    loss = []

    for _ in range(iterations):
        y_predicted = np.array([m * X]) + b
        grad_m: float
        grad_b: float
        grad_m, grad_b = compute_gradients(X, y, y_predicted)
        m -= learning_rate * grad_m
        b -= learning_rate * grad_b
        steps.append((m, b))

        mse: float = mean_squared_error(y, y_predicted)
        loss.append(mse)

    return steps, loss


def compute_gradients(X, y_actual, y_predicted) -> Tuple[float]:
    """Calculate gradient for MSE."""
    error: float = y_actual - y_predicted
    grad_m: float = np.mean(-2 * (X * error))  # Gradient for the slope (m)
    grad_b: float = np.mean(-2 * error)  # Gradient for the intercept (b)
    return (grad_m, grad_b)

def predict(x_new, m, b):
    """Predict new values with the trained linear regression model."""
    return m * x_new + b


if __name__ == "__main__":
    # example usage
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([1, 3, 5, 8, 10, 13])
    initial_m = 0
    initial_b = 0
    iterations = 50
    learning_rate = 0.01

    steps, loss = lin_reg(iterations, learning_rate, initial_m, initial_b, x, y)
    final_m, final_b = steps[-1]
    new_x = 6
    predicted = predict(6, final_m, final_b)
    print(predicted) # 15.061807807592944

