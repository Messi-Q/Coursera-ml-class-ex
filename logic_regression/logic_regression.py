# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# load file
def load_data(filename):
    data = []
    file = open(filename)
    for line in file.readlines():
        linearr = line.strip().split(',')
        col_num = len(linearr)
        temp = []
        for i in range(col_num):
            temp.append(float(linearr[i]))
        data.append(temp)
    return np.array(data)


data = load_data('ex2data1.txt')
print(data.shape)
print(data[:5])

X = data[:, :-1]
y = data[:, -1:]
print(X.shape)
print(y.shape)
print(X[:5])
print(y[:5])

# 可视化数据集
label0 = np.where(y.ravel() == 0)  # 能够返回符合某一条件的下标的函数是np.where()，返回符合条件的下标；
# ravel函数将多维数组降为一维，仍返回array数组，元素以列排列
plt.scatter(X[label0, 0], X[label0, 1], marker='x', color='red', label='not admitted')
label1 = np.where(y.ravel() == 1)
plt.scatter(X[label1, 0], X[label1, 1], marker='o', color='blue',
            label='admitted')  # x[:,1]获取第二列作为一维数组，x[:,0]获取第一列作为一维数组
plt.xlabel('Exam1 score')
plt.ylabel('Exam2 score')
plt.legend(loc='upper right')  # 显示图例位置
plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def out(x, w):
    return sigmoid(np.dot(x, w))


# compute the cost
def compute_cost(X_train, y_train, theta):
    m = X_train.shape[0]
    J = 0
    theta = theta.reshape(-1, 1)
    grad = np.zeros((X_train.shape[1], 1))  # np.zeros((2, 1))
    h = out(X_train, theta)
    J = -1 * np.sum(y_train * np.log(h) + (1 - y_train) * np.log((1 - h))) / m
    grad = X_train.T.dot((h - y_train)) / m
    grad = grad.ravel()
    return J, grad


# test the grad
m = X.shape[0]
one = np.ones((m, 1))
X = np.hstack((one, data[:, :-1]))  # 最后一列
w = np.zeros((X.shape[1], 1))

cost, grad = compute_cost(X, y, w)
print('compute with w=[0,0,0]')
print('Expected cost (approx):0.693...')
print(cost)
print('Expected gradients (approx):[-0.1,-12,-11]')
print(grad)

cost1, grad1 = compute_cost(X, y, np.array([[-24], [0.2], [0.2]]))
print('compute with w=[-24,0.2,0.2]')
print('Expected cost (approx):0.218....')
print(cost1)
print('Expected gradients (approx): [0.04,2.566,0.646]')
print(grad1)

# 使用高级优化算法
params = np.zeros((X.shape[1], 1)).ravel()
args = (X, y)


def f(params, *args):
    X_train, y_train = args
    m, n = X_train.shape
    J = 0
    theta = params.reshape((n,1))
    h = out(X_train, theta)
    J = -1 * np.sum(y_train * np.log(h) + (1 - y_train) * np.log((1 - h))) / m
    return J


def gradf(params, *args):
    X_train, y_train = args
    m, n = X_train.shape
    theta = params.reshape(-1, 1)
    h = out(X_train, theta)
    grad = np.zeros((X_train.shape[1], 1))
    grad = X_train.T.dot((h - y_train)) / m
    g = grad.ravel()
    return g


res = optimize.fmin_cg(f, x0=params, fprime=gradf, args=args, maxiter=500)
print(res)

# 可视化线性的决策边界
label = np.array(y)
index_0 = np.where(label.ravel() == 0)
plt.scatter(X[index_0, 1], X[index_0, 2], marker='x', color='b', label='not admitted', s=15)
index_1 = np.where(label.ravel() == 1)
plt.scatter(X[index_1, 1], X[index_1, 2], marker='o', color='r', label='admitted', s=15)
x1 = np.arange(20, 100, 0.5)
x2 = (-res[0] - res[1] * x1) / res[2]
plt.plot(x1, x2, color='black')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper right')
plt.show()


# 预测函数
def predict(X, theta):
    h = out(X, theta)
    y_pred = np.where(h >= 0.5, 1.0, 0)
    return y_pred


# test
prob = out(np.array([[1, 45, 85]]), res)
print("For a student with scores 45 and 85, we predict an admission ")
print("Expected value: 0.775 +/- 0.002")
print(prob)

p = predict(X, res)
print("Expected accuracy (approx): 89.0")
print(np.mean(p == y.ravel()))
