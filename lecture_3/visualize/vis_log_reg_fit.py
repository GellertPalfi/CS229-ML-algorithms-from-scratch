import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lecture_3.concepts.logistic_regression import LogisticRegression
from matplotlib.animation import FuncAnimation



np.random.seed(1)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)
features = np.vstack((x1, x2)).astype(np.float32)
labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
data_with_intercept = np.hstack((np.ones((features.shape[0], 1)),
                                features))
max_iter = 30
learning_rate = 0.003

log_reg = LogisticRegression()
log_reg.fit(
    max_iter, learning_rate, features, labels, True, True
)
own_predict = log_reg.predict(data_with_intercept)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# remove intercept term
gradients = [grad[1:] for grad in log_reg.gradients]
loss = np.array(log_reg.loss)
loss = np.abs(loss)

param1_values = [grad[0] for grad in gradients]
param2_values = [grad[1] for grad in gradients]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the trajectory
ax.plot(param1_values, param2_values, loss, label='Gradient Descent Path', color='r', marker='o')

# Labeling the axes
ax.set_xlabel('Parameter 1')
ax.set_ylabel('Parameter 2')
ax.set_zlabel('Loss')
ax.set_title('3D Trajectory of Gradient Descent')

# Show the plot
plt.legend()
plt.show()
