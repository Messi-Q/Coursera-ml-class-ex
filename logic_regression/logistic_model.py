# -*- coding:utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class logistic(object):
    def __init__(self):
        self.W = None

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def output(self, X):
        g = np.dot(X, self.W)
        return self.sigmoid(g)

    def compute_loss(self, X, y):
        num_train = X.shape[0]
        h = self.output(X)
        loss = -np.sum((y * np.log(h) + (1 - y) * np.log((1 - h))))
        loss = loss / num_train
        dw = X.T.dot((h - y)) / num_train
        return loss, dw

    def train(self, X, y, learn_rate=0.01, num_iters=2000):
        num_train, num_feature = X.shape
        self.W = 0.001 * np.random.randn(num_feature, 1).reshape(-1, 1)
        loss = []

        for i in range(num_iters):
            error, dw = self.compute_loss(X, y)
            self.W += -learn_rate * dw
            loss.append(error)
            if i % 200 == 0:
                print('i=%d error=%f' % (i, error))
        return loss

    def predict(self, X_test):
        h = self.outout(X_test)
        y_predict = np.where(h >= 0.5, 1, 0)
        return y_predict


iris = load_iris()
data = iris.data
target = iris.target
print(data[:10])
print(target[10:])
X = data[0:100, [0, 2]]
y = target[0:100]
print(X[:5])
print(y[-5:])
label = np.array(y)
index_0 = np.where(label.ravel() == 0)
plt.scatter(X[index_0, 0], X[index_0, 1], marker='x', color='b', label='0', s=15)
index_1 = np.where(label.ravel() == 1)
plt.scatter(X[index_1, 0], X[index_1, 1], marker='o', color='r', label='1', s=15)
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper right')
plt.show()

y = y.reshape((-1, 1))  # 经常需要reshape,建议多用，避免不必要的bug
# add the x0=1
one = np.ones((X.shape[0], 1))
X_train = np.hstack((one, X))
classify = logistic()
loss = classify.train(X_train, y)
print(classify.W)

plt.plot(loss)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

# 可视化一下决策边界
label = np.array(y)
index_0 = np.where(label.ravel() == 0)
plt.scatter(X[index_0, 0], X[index_0, 1], marker='x', color='b', label='0', s=15)
index_1 = np.where(label.ravel() == 1)
plt.scatter(X[index_1, 0], X[index_1, 1], marker='o', color='r', label='1', s=15)

# show the decision boundary
x1 = np.arange(4, 7.5, 0.5)
x2 = (- classify.W[0] - classify.W[1] * x1) / classify.W[2]
plt.plot(x1, x2, color='black')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper right')
plt.show()
