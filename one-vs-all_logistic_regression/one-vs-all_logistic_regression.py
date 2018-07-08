# -*- coding:utf-8 -*-
from sklearn.datasets import load_digits
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# 这个数据集中并没有图片，而是经过提取得到的手写数字特征和标记，就免去了我们的提取数据的麻烦，
# 但是在实际的应用中是需要我们对图片中的数据进行提取
digits = load_digits()  # 1797*64 共有1797个数据，0 ~9中的某个，均为8*8像素，故有64列，每列代表其像素的灰度值（0~255之间）
print(digits.keys())  # 属性查看
data = digits.data  # 训练集
target = digits.target  # 标记，特征向量对应的标记，每一个元素都是自然数是0-9的数字，label
print(data.shape)
print(target.shape)
print('the images 15 is', target[15])
# 随机选50个数据进行识别
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
num_classes = len(classes)
samples_per_class = 5  # 每个类别采样个数

for y, cla in enumerate(classes):  # 既要遍历索引又要遍历元素，对列表元素位置和元素进行循环，y表示元素位置（0,num_class），cla元素本身
    idxs = np.flatnonzero(target == y)  # 找出标签中y类的位置 (该函数输入一个矩阵，返回扁平化后矩阵中非零元素的位置（index）,这里用来返回某个特定元素的位置)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)  # # 从上述找到的位置中随机选出我们所需的5个样本，没有重复出现的数值
    for i, idx in enumerate(idxs):  # 对所选的样本的位置和样本所对应的图片在训练集中的位置进行循环
        plt_idx = i * num_classes + y + 1  # 在子图中所占位置的计算
        plt.subplot(samples_per_class, num_classes, plt_idx)  # 说明要画的子图的编号，subplot绘制多个子图
        plt.imshow(digits.images[idx].astype('uint8'))  # 画图,为图像提供了特殊的数据类型uint8(8位无符号整数),存入一个8位矩阵
        plt.axis('off')
        if i == 0:
            plt.title(cla)
plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def out(x, w):
    return sigmoid(np.dot(x, w))


# 优化算法，稍微改写了cost函数（代价函数）和gradient函数,分别为f(),gradf()
def f(params, *args):
    X_train, y_train, reg = args  # reg相当于lamba ，正则化参数
    m, n = X_train.shape
    J = 0
    theta = params.reshape((n, 1))
    h = out(X_train, theta)
    theta_1 = theta[1:, :]
    J = -1 * np.sum(y_train * np.log(h) + (1 - y_train) * np.log((1 - h))) / m + 0.5 * reg * theta_1.T.dot(theta_1) / m
    return J


def gradf(params, *args):
    X_train, y_train, reg = args
    m, n = X_train.shape
    theta = params.reshape(-1, 1)
    h = out(X_train, theta)
    theta_1 = theta[1:, :]
    grad = X_train.T.dot((h - y_train)) / m
    grad[1:, :] += reg * theta_1 / m
    g = grad.ravel()
    return g


def oneVsAll(x, y, num_class, reg):
    m, n = x.shape
    thetas = np.zeros((n, num_class))  # 调用onevsall函数时，此时thetas=65*10
    for i in range(num_class):
        params = np.zeros((x.shape[1], 1)).ravel()
        args = (x, y == i, reg)
        # 使用高级优化算法（fmin_cg）去训练
        res = optimize.fmin_cg(f, x0=params, fprime=gradf, args=args, maxiter=500)
        thetas[:, i] = res  # 就是取所有行的第几个数据
    return thetas


X = data
y = target
y = y.reshape((-1, 1))
X_mean = np.mean(X, axis=0)  # 对各列求均值，返回 1* n 矩阵 # 1*64
X -= X_mean
m = X.shape[0]
X = np.hstack((np.ones((m, 1)), X))
# hstack会将多个value(value_one, value_two)的相同维度的数值组合在一起，并以同value同样的数据结构返回numpy数组
print(X.shape)
thetas = oneVsAll(X, y, 10, 1.0)
print(thetas.shape)


def predict(x, thetas):
    h = out(x, thetas)
    print(h.shape)
    a = sigmoid(h)
    print(a.shape)
    pred = np.argmax(a, axis=1)  # 找出每行的最大值（即概率）,返回列号即预测的数字。返回后，实际组成的数列是按照扁平的形式排列
    return pred


y_pred = predict(X, thetas)
print(y_pred.shape)
print("train accuracy is :", np.mean(y.ravel() == y_pred))  # 首先这两个矩阵进行比较，返回一个bool矩阵，当对应的数字一致，返回True(或1)，不同False（0）
                                                              #  形成bool矩阵后，通过mean函数，求[0,1,1,0,1,0,1......]的均值，即准确率
# 随机选出几个图片，测试一下
m, n = data.shape
example_size = 10
example_index = np.random.choice(m, (example_size))  # 随机选出几个图片
print(example_index)
for i, idx in enumerate(example_index):
    print("%d example is number %d,we predict it as %d" % (i, target[idx], predict(X[idx, :].reshape(1, -1), thetas)))

# 从mat文件导入数据，这里采用前向神经网络，不再用原有的数据集
weight = sio.loadmat('ex3weights')  # 5000*400
Theta1 = weight["Theta1"]
Theta2 = weight["Theta2"]

data = sio.loadmat('ex3data1')
X = data["X"]
y = data["y"]
print(X.shape)
print(y.shape)

m = X.shape[0]
X1 = np.hstack((np.ones((m, 1)), X))
print(X1.shape)


# 模拟实现前向传播算法
def nn_predict(x, theta1, theta2):
    m = x.shape[0]
    a1 = sigmoid(x.dot(theta1.T))
    a1 = np.hstack((np.ones((m, 1)), a1))
    a2 = sigmoid(a1.dot(theta2.T))
    print(a2.shape)
    pred = np.argmax(a2, axis=1) + 1  # 这里的数字是1-10，而不是0-9，这里的a2相当于h（x）
    return pred


pred = nn_predict(X1, Theta1, Theta2)
print(pred[-10:])
print(y[-10:])
print("train accuracy is :", np.mean(y == pred.reshape(-1, 1)) * 100)

# 随机选择样本，测试一下
m, n = X1.shape
example_size = 10
example_index = np.random.choice(m, example_size)
print(example_index)
for i, idx in enumerate(example_index):
    print("%d example is number %d,we predict it as %d" % (i, y[idx], nn_predict(X1[idx, :].reshape(1, -1), Theta1, Theta2)))
