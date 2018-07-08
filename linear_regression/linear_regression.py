# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
data = load_data('ex1data1.txt')
print(data.shape)
print(data[:5])

X = data[:, :-1]  # 第一列所有数据
y = data[:, -1:]  # 第二列所有数据
print(X.shape)
print(y.shape)
print(X[:5])
print(y[:5])
# 可视化数据
plt.scatter(X.ravel(), y.ravel(), color='r', marker='x')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# 计算代价函数
num_train = X.shape[0]
one = np.ones((num_train, 1))
X = np.hstack((one, data[:, :-1]))
W = np.zeros((2, 1))
print(X.shape)
print(W)


# 定义计算代价函数，并测试是否正确
def compute_cost(X_test, y_test, theta):
    num_X = X_test.shape[0]  # shape[0]是读取矩阵第一维度的长度
    cost = 0.5 * np.sum(np.square(X_test.dot(theta) - y_test)) / num_X
    return cost


cost_1 = compute_cost(X, y, W)
print('cost =%f,with W =[0,0]' % (cost_1))  # %是格式符号
print('Expected cost value (approx) 32.07')
cost_2 = compute_cost(X, y, np.array([[-1], [2]]))  # 是一个二维数组，每个数组中有1个元素，2行1列，此时W =[-1,2]
print('cost =%f,with W =[-1,2]' % (cost_2))
print('Expected cost value (approx) 54.24')


# 定义梯度下降函数，更新参数theta，并测试是否正确
def gradient_descent(X_test, y_test, theta, alpha=0.01, iters=1500):
    J_history = []
    num_X = X_test.shape[0]
    for i in range(iters):
        theta = theta - alpha * X_test.T.dot(X_test.dot(theta) - y_test) / num_X
        cost = compute_cost(X_test, y_test, theta)
        J_history.append(cost)
    return theta, J_history


theta, J_history = gradient_descent(X, y, np.array([[0.001], [0.001]]))
print(theta)
print('Expected theta values (approx) W = [-3.6303,1.1664] ')

# predict
predict1 = np.array([[1, 3.5]]).dot(theta)  # 相当于h(x)  x*theta（x0=1,x1=3.5w）
predict2 = np.array([[1, 7]]).dot(theta)
print(predict1 * 10000, predict2 * 10000)

# plot the result可视化一下回归的曲线
plt.subplot(211)
plt.scatter(X[:, 1], y, color='r', marker='x')
plt.xlabel('X')
plt.ylabel('y')

plt.plot(X[:, 1], X.dot(theta), '-', color='black')
# 可视化一下cost变化曲线
plt.subplot(212)  # 分成2x1，占用第二个，即第二行
plt.plot(J_history)
plt.xlabel('iters')
plt.ylabel('cost')
plt.show()

# 绘制3D图像
size = 100
theta0Vals = np.linspace(-10, 10, size)  # 前两个参数分别是数列的开头与结尾。第三个参数，表示数列的元素个数
theta1Vals = np.linspace(-1, 4, size)  # theta0  theta1取值，size取100
JVals = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        col = np.array([[theta0Vals[i]], [theta1Vals[j]]]).reshape(-1, 1)  # 不知道z的shape属性，想让z变成只有一列，通过`z.reshape(-1,1)，Numpy自动计算出有多少行
        JVals[i, j] = compute_cost(X, y, col)

theta0Vals, theta1Vals = np.meshgrid(theta0Vals, theta1Vals)  # 产生一个以向量x为行，向量y为列的矩阵，X、Y必定是行数、列数相等的，且X、Y的行数都
                                                              # 等于输入参数y中元素的总个数，X、Y的列数都等于输入参数x中元素总个数；形成网格
JVals = JVals.T
print(JVals.shape, JVals[0, 0], JVals[1, 1])

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(theta0Vals, theta1Vals, JVals)
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('J(theta)')
plt.show()

# 绘制曲线轮廓
contourFig = plt.figure()
ax = contourFig.add_subplot(111)
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')

CS = ax.contour(theta0Vals, theta1Vals, JVals, np.logspace(-2, 3, 20))  # 绘制等高线  logspace创建等比数列数组，默认底数10；从10的-2次方到10的3次方
plt.clabel(CS, inline=1, fontsize=10)  # inline 控制画标签，移除标签下的线
# 绘制最优解
ax.plot(theta[0, 0], theta[1, 0], 'rx', markersize=10, linewidth=2)
plt.savefig('fig.png',bbox_inches='tight') # 将图像保存为fig.png
plt.show()
