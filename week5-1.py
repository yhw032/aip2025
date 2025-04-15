import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[1.0], [2.0], [3.0]])
y_train = np.array([[2.0], [4.0], [6.0]])

theta = np.random.randn(2, 1)

x_train_b = np.c_[np.ones((len(x_train), 1)), x_train]  

def cal_cost(theta, x, y):
    m = len(x)
    pred = x.dot(theta)  # y_hat
    cost = (1 / (2 * m)) * np.sum(np.square(pred - y)) # 1/m · ∑(y_hat - y)^2
    return cost

def gradient_descent(x, y, theta, lr=0.01, n_iter=20):
    m = len(x)
    for iter in range(n_iter):
        pred = np.dot(x, theta) 
        theta = theta - (lr / m) * (x.T.dot(pred - y))  # θ = θ - α/m · ∑(y_hat - y)·x
    return theta


print("W=", theta[0])
print("B=", theta[1])