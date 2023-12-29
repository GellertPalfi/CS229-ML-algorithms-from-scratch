import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.metrics import log_loss

from algorithms.logistic_regression.logistic_regression import LogisticRegression

np.random.seed(1)
num_observations = 5000
max_iter = 10000
learning_rate = 0.003

x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)
features = np.vstack((x1, x2)).astype(np.float32)
labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))


log_reg = LogisticRegression()
log_reg.fit(max_iter, learning_rate, features, labels, True, True)

# create a mesh to plot in
b1_range = np.linspace(-50, 50, 100)
b2_range = np.linspace(-50, 50, 100)
B1, B2 = np.meshgrid(b1_range, b2_range)

# Vectorize coefficient combinations
coeffs = np.stack(np.meshgrid(b1_range, b2_range), -1).reshape(-1, 2)

intercept = log_reg.intercept
logit_scores = intercept + np.dot(features, coeffs.T)
y_pred_probs = 1 / (1 + np.exp(-logit_scores))

# Compute log loss for each pair of coefficients
errors = np.array(
    [log_loss(labels, y_pred_probs[:, i]) for i in range(y_pred_probs.shape[1])]
)
errors = errors.reshape(B1.shape)


# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(B1, B2, errors, cmap="viridis")
fig.colorbar(surf)

ax.set_xlabel("feature1")
ax.set_ylabel("feature2")
ax.set_zlabel("Error")
ax.set_title("Error Space 3D Surface Plot")

# Inverting the feature1 axis
ax.set_ylim(ax.get_ylim()[::-1])

# skip first 8 gradients because they are too large, and last steps
data = log_reg.gradients[8:500]
# invert loss
new_data = np.abs(np.array(data))


# Initialize a line plot
(line,) = ax.plot([], [], [], color="r", linestyle="-", linewidth=2, zorder=5)
iter = 0
ax.set_ylim(-50, 50)
ax.set_zlim(0, 30)
ax.set_ylim(ax.get_ylim()[::-1])


def update(num, new_data, line):
    """Update function for the animation"""
    global iter
    print(f"{iter}/{len(new_data)}")
    line.set_data(new_data[:num, 0:2].T)
    line.set_3d_properties(new_data[:num, 2])
    iter += 1
    return (line,)


# Creating the animation
ani = FuncAnimation(fig, update, frames=len(data), fargs=(new_data, line), blit=False)
ani.save("logistic_regression_progress.gif", writer="pillow", fps=4)
