import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

# get data
path = os.getcwd() + '/challenge_dataset.txt'
data = pd.read_csv(path, header=None, names=['X_value', 'Y_value'])

data.insert(0, 'Ones', 1)

alpha = 0.01
iters = 1000

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

print "Initial cost: ", computeCost(X, y, theta)

# perform linear regression
g, cost = gradientDescent(X, y, theta, alpha, iters)
print "Theta: ", g
print "End cost:     ", computeCost(X, y, g)

# plotting data
x = np.linspace(data.X_value.min(), data.Y_value.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.X_value, data.Y_value, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('X_value')
ax.set_ylabel('Y_value')
ax.set_title('Predicted X_value vs. Y_value')
plt.show()