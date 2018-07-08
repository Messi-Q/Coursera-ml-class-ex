# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# load data from file
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


# 打印出来看一下数据集
data = load_data('ex1data2.txt')
print(data.shape)
print(data[:5])

X = data[:, :-1]  # 第一列所有数据
y = data[:, -1:]  # 第二列所有数据
print(X.shape)
print(y.shape)
print(X[:5])
print(y[:5])


# 定义一下特征缩放函数，因为每个特征的取值范围不同，且差异很大
def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    mu = np.mean(X, axis=0)  # 按列计算的均值
    sigma = np.std(X, axis=0)  # 按列计算的标准差
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


X_norm, mu, sigma = featureNormalize(data[:, :-1])
num_train = X.shape[0]
one = np.ones((num_train, 1))
X = np.hstack((one, X_norm))  # 在水平方向上平铺
W = np.zeros((X.shape[1], 1))


# 计算代价函数
def compute_cost(X_test, y_test, theta):
    num_X = X_test.shape[0]
    cost = 0.5 * np.sum(np.square(X_test.dot(theta) - y_test)) / num_X
    return cost


# 计算梯度下降
def gradient_descent(X_test, y_test, theta, alpha=0.005, iters=1500):
    J_history = []
    num_X = X_test.shape[0]
    for i in range(iters):
        theta = theta - alpha * X_test.T.dot(X_test.dot(theta) - y_test) / num_X
        cost = compute_cost(X_test, y_test, theta)
        J_history.append(cost)
    return theta, J_history


# 测试结果
print('run gradient descent')
theta, J_history = gradient_descent(X, y, W)
print('Theta computed from gradient descent: ', theta)
# 绘制代价函数曲线
plt.plot(J_history, color='b')
plt.xlabel('iters')
plt.ylabel('J(theta)')
plt.show()

# predict
X_t = ([[1650, 3]] - mu) / sigma
X_test = np.hstack((np.ones((1, 1)), X_t))
predict = X_test.dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent)')
print(predict)

# 直接使用公式求解最佳theta(正规方程),不用梯度下降法
XX = data[:, :-1]
yy = data[:, -1:]
m = XX.shape[0]

one = np.ones((m, 1))
XX = np.hstack((one, data[:, :-1]))
print(XX.shape)
print(yy.shape)


def normalEquation(X_train, y_train):
    # w = np.zeros((X_train.shape[0], 1))
    w = ((np.linalg.pinv(X_train.T.dot(X_train))).dot(X_train.T)).dot(y_train)
    return w


w = normalEquation(XX, yy)
print('Theta computed from the normal equations:')
print(w)

# predict price
price = np.dot(np.array([[1, 1650, 3]]), w)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations)')
print(price)
