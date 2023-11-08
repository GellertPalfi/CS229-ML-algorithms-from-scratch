from concepts.linear_regression import lin_reg
from matplotlib.animation import FuncAnimation

import numpy as np

import matplotlib.pyplot as plt


x = np.array([0, 1, 2, 3, 4, 5, 7, 9])
y = np.array([1, 3, 5, 8, 10, 13, 17, 23])
m = np.random.randint(0, 10)
b = np.random.randint(0, 10)
learning_rate = 0.03
max_iters = 100

slopes, loss = lin_reg(max_iters, learning_rate, m, b, x, y)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.set_xlim(0, max(x) + 1)
ax1.set_ylim(0, max(y) + 1)
data = ax1.scatter(x=x, y=y)
(line1,) = ax1.plot([], [], "b", lw=2)

ax2.set_xlim(0, max_iters)
ax2.set_ylim(0, max(loss))
ax2.set_xlabel("Iterations")
ax2.set_ylabel("MSE")
ax2.legend(["Loss"], loc="upper right")
(line2,) = ax2.plot(
    [],
    [],
    "ro-",
    lw=2,
)


def update(frame):
    # Update data for the regression line plot
    m, b = slopes[frame]
    x_values = np.linspace(0, max(x), 100)
    y_values = m * x_values + b
    line1.set_data(x_values, y_values)
    ax1.legend([data, line1], ["Data points", "Regression Line"], loc="lower right")
    ax2.legend(["Loss"], loc="upper right")

    # Update data for the loss plot
    line2.set_data(range(frame + 1), loss[: frame + 1])

    # Return both line objects
    return line1, line2


ani = FuncAnimation(fig, update, frames=max_iters, interval=500, blit=True)

# Save the animation
ani.save("linear_regression_progress.gif", writer="imagemagick", fps=4)
