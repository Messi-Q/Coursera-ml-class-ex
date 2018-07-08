# -*- coding:utf-8 -*-
# 绘制学习曲线，学会误差分析,进行模型诊断和选择
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.io as sio

data = sio.loadmat('ex5data1')
X = data['X']  # 训练集
y = data['y']
Xval = data['Xval']  # 验证集
yval = data['yval']
Xtest = data['Xtest']  # 测试集
ytest = data['ytest']
print(X.shape, Xval.shape, Xtest.shape)

plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('change in water level(x)')
plt.ylabel('water flowing out of the dam(y)')
plt.show()


# 正则化的线性回归
def linearRegCostFunction(X, y, theta, reg):
    m, n = X.shape
    J = 0
    theta = theta.reshape((n, 1))
    grad = np.zeros_like(theta)
    theta1 = theta[1:, :]
    J = 0.5 * np.sum((X.dot(theta) - y) ** 2) / m + reg * 0.5 * np.sum(theta1 ** 2) / m
    grad = X.T.dot(X.dot(theta) - y) / m
    grad[1:, :] += reg * theta1 / m
    return J, grad


m = X.shape[0]
XX = np.hstack((np.ones((m, 1)), X))  # add the x0 = 1
XXval = np.hstack((np.ones((Xval.shape[0], 1)), Xval))
XXtest = np.hstack((np.ones((Xtest.shape[0], 1)), Xtest))

# 测试正确性,这里给theta0和theta1的值为1，1
init_theta = np.array([[1], [1]])
J, grad = linearRegCostFunction(XX, y, init_theta, 1.0)
print('Cost at theta = [1 ; 1]: (this value should be about 303.993192)')
print(J)
print('Gradient at theta = [1 ; 1](this value should be about [-15.303016; 598.250744])')
print(grad)


def f(params, *args):
    X, y, reg = args
    m, n = X.shape
    J = 0
    theta = params.reshape((n, 1))
    theta1 = theta[1:, :]
    J = 0.5 * np.sum((X.dot(theta) - y) ** 2) / m + reg * 0.5 * np.sum(theta1 ** 2) / m
    return J


def gradf(params, *args):
    X, y, reg = args
    m, n = X.shape
    theta = params.reshape((n, 1))
    grad = np.zeros_like(theta)
    theta1 = theta[1:, :]
    grad = X.T.dot(X.dot(theta) - y) / m
    grad[1:, :] += reg * theta1 / m
    g = grad.ravel()
    return g


# Train linear regression with lambda = 0
def train(X, y, reg):
    args = (X, y, reg)
    inital_theta = np.zeros((X.shape[1], 1))
    params = inital_theta.ravel()
    res = optimize.fmin_cg(f, x0=params, fprime=gradf, args=args, maxiter=500)
    return res


res = train(XX, y, 0)
print('result=', res)
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.plot(X, XX.dot(res), '-')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()


# 定义学习曲线，观察数据集大小m的变化对训练误差和验证误差的影响
def learnCurve(X, y, Xval, yval, reg):
    m = X.shape[0]
    err_train = []
    err_val = []
    for i in range(m):
        best_theta = train(X[:i + 1, :], y[0:i + 1], reg)
        err_t, g1 = linearRegCostFunction(X[0:i + 1, :], y[0:i + 1], best_theta, reg)
        err_v, g2 = linearRegCostFunction(Xval, yval, best_theta, reg)
        err_train.append(err_t)
        err_val.append(err_v)
    return err_train, err_val


err_train, err_val = learnCurve(XX, y, XXval, yval, 0)
print(err_train[:5])
print(err_val[:5])
# 绘制一下这个模型的学习曲线,从图形看出，模型属于“高偏差”
plt.plot(err_train, 'b', linestyle='-', label='err_train')
plt.plot(err_val, 'r', linestyle='-', label='err_val')
plt.xlabel('Number of training examples(m)')
plt.ylabel('error')
plt.title('Learning curve for linear regression')
plt.legend(loc='upper right')
plt.show()


# 解决高偏差，增加多项式或高次项特征
def ployFeatures(X, p=8):
    X_ploy = np.zeros((X.shape[0], p))
    for i in range(p):
        X_ploy[:, i] = X.T ** (i + 1)
    return X_ploy


# 有高次项，进行特征均值归一化和特征缩放
def featureNormalize(x):
    mu = np.mean(x, axis=0)
    xx = x - mu
    sigma = np.std(x, axis=0)  # 计算标准差
    x_norm = xx / sigma  # 0均值标准化，0均值归一化方法将原始数据集归一化为均值为0、方差1的数据集，
                         # 该种归一化方式要求原始数据的分布可以近似为高斯分布
    return x_norm, mu, sigma


X_ploy = ployFeatures(X, p=8)
X_ploy, mu, sigma = featureNormalize(X_ploy)
X_ploy = np.hstack((np.ones((X.shape[0], 1)), X_ploy))
print(mu)
print(sigma)
print(X_ploy[1, :])

X_ploy_test = ployFeatures(Xtest, p=8)
X_ploy_test = (X_ploy_test - mu) / mu
X_ploy_test = np.hstack((np.ones((Xtest.shape[0], 1)), X_ploy_test))

X_ploy_val = ployFeatures(Xval, p=8)
X_ploy_val = (X_ploy_val - mu) / mu
X_ploy_val = np.hstack((np.ones((Xval.shape[0], 1)), X_ploy_val))

print(X_ploy.shape, X_ploy_test.shape, X_ploy_val.shape)
res1 = train(X_ploy, y, 0)
print(res1)


# 可视化拟合的曲线
def plotFit(mu, sigma, theta, p):
    x = np.linspace(-50, 50).reshape(-1, 1)
    x_ploy = ployFeatures(x, p)
    x_ploy = x_ploy - mu
    x_ploy = x_ploy / sigma
    x_ploy = np.hstack((np.ones((x.shape[0], 1)), x_ploy))
    plt.plot(x, x_ploy, '--', color='black')


# 发现过拟合，需要选择合适的正则化参数reg
plt.plot(X, y, markersize=10, linewidth=1.5)
plotFit(mu, sigma, res1, p=8)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

err_train, err_val = learnCurve(X_ploy, y, X_ploy_val, yval, 0)
print(err_train[:5])
print(err_val[:5])
# 绘制这个模型的学习曲线,从图形看出，模型属于“高方差”
plt.plot(err_train, 'b', linestyle='-', label='err_train')
plt.plot(err_val, 'r', linestyle='-', label='err_val')
plt.xlabel('Number of training examples(m)')
plt.ylabel('error')
plt.title('Learning curve for linear regression')
plt.legend(loc='upper right')
plt.show()


# 解决高方差，需要交叉验证来决定，由交叉验证决定正则化参数reg
def validationCurve(X, y, Xval, yval):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    err_train = []
    err_val = []
    for reg in lambda_vec:  # 选择合适的reg
        best_theta = train(X, y, reg)
        err_t, g1 = linearRegCostFunction(X, y, best_theta, reg)
        err_v, g2 = linearRegCostFunction(Xval, yval, best_theta, reg)
        err_train.append(err_t)
        err_val.append(err_v)
    return lambda_vec, err_train, err_val


lambda_vec, err_train, err_val = validationCurve(X_ploy, y, X_ploy_val, yval)
plt.plot(lambda_vec, err_train, 'b', linestyle='-', label='err_train')
plt.plot(lambda_vec, err_val, 'r', linestyle='-', label='err_val')
plt.xlabel('the lamda(reg)')
plt.ylabel('error')
plt.title('Learning curve for linear regression')
plt.legend(loc='upper right')
plt.show()
# 打印对比err_train和err_val
print('reg\t  err_train\t  err_val')
for i in range(len(lambda_vec)):
    print('%f\t%f\t%f' % (lambda_vec[i], err_train[i], err_val[i]))

# 当reg=0.3,结果会平滑一点，不会太过拟合
res1 = train(X_ploy, y, reg=0.3)
print(res1)
# 可视化拟合曲面,发现更加平滑，不会过拟合
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plotFit(mu, sigma, res1, p=8)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()
