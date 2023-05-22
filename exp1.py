import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for binary classification
np.random.seed(0)
X = np.random.normal(size=(100, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Define the sigmoid function for logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the cost function for logistic regression
def cost_function(X, y, theta):
    h = sigmoid(X.dot(theta))
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

# Define the gradient function for logistic regression
def gradient_function(X, y, theta):
    h = sigmoid(X.dot(theta))
    return X.T.dot(h - y) / len(y)

# Perform gradient descent to optimize the parameters
theta = np.zeros(2)  # Initialize parameters to zeros
alpha = 0.1  # Learning rate
epochs = 1000  # Number of iterations
cost_history = []
for i in range(epochs):
    cost = cost_function(X, y, theta)
    grad = gradient_function(X, y, theta)
    theta -= alpha * grad  # Update parameters based on gradient
    cost_history.append(cost)

# Plot the cost history to check for convergence
plt.plot(cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()

# Plot the decision boundary on top of the data
x_plot = np.linspace(-3, 3)
print(theta)
y_plot = -(theta[0] * x_plot + np.log(0.5 / (1 - 0.5))) / theta[1]  #""" + np.log(0.5 / (1 - 0.5))"""
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
plt.plot(x_plot, y_plot, 'g--')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
