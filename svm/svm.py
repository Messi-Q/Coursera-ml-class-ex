import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm

# data1
raw_data = sio.loadmat('ex6data1.mat')  # 原始数据
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])  # X是51*2的二维数组，columns 是表头（列的名称）
data['y'] = raw_data['y']

positive = data[data['y'].isin([1])]  # 如果值是一个数据框，然后索引和列标签必须匹配。返回bool值
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
ax.set_title('Dataset1')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend(loc='upper right')
plt.show()


# 定义一个函数画决策边界
def plot_decision_boundary(pred_func, X, y, gap):
    x_min, x_max = X[:, 0].min() - gap, X[:, 0].max() + gap
    y_min, y_max = X[:, 1].min() - gap, X[:, 1].max() + gap
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])  # np.c_函数, 使一维扁平矩阵，按列组合,这里相当于2列
    z = z.reshape(xx.shape)
    ax.contour(xx, yy, z, )  # 绘制等高线


# C越大，方差越大，容易过拟合，C越小，偏差越大，容易欠拟合
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
svc.fit(data[['X1', 'X2']], data['y'])  # 用于训练SVM，只需要给出数据集X和X对应的标签y即可。
acc_train = svc.score(data[['X1', 'X2']], data['y'])
print("the accuracy with C=1 is : ", acc_train)

svc2 = svm.LinearSVC(C=1000, loss='hinge', max_iter=1000)
svc2.fit(data[['X1', 'X2']], data['y'])
acc2_train = svc2.score(data[['X1', 'X2']], data['y'])
print("the accuracy with C=1000 is : ", acc2_train)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
ax.set_title('SVM (C=1) Decision Confidence')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend(loc='upper right')
plot_decision_boundary(lambda x: svc.predict(x), raw_data['X'], data['y'], 0.1)  # 画出C=1时的决策边界
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
ax.set_title('SVM (C=1000) Decision Confidence')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend(loc='upper right')
plot_decision_boundary(lambda x: svc2.predict(x), raw_data['X'], data['y'], 0.1)  # 画出C=1000时的决策边界
plt.show()


# 定义高斯核函数来画决策边界
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))


x1 = np.array([1.0, 2.0, 1.0])
x2 = np.array([0.0, 4.0, -1.0])
sigma = 2
print("gaussian_kernel is : ", gaussian_kernel(x1, x2, sigma))

# data2
raw_data = sio.loadmat('ex6data2')
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']

positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
ax.set_title('Dataset2')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.legend(loc='upper right')
plt.show()

svc3 = svm.SVC(C=100, gamma=10, probability=True)
svc3.fit(data[['X1', 'X2']], data['y'])
acc3_train = svc3.score(data[['X1', 'X2']], data['y'])
print("the accuracy with C=100 is : ", acc3_train)

# data['Probability'] = svc3.predict_proba(data[['X1', 'X2']])[:, 0]  # 分类概率

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
# ax.scatter(data['X1'], data['X2'], s=50, c=data['Probability'], cmap='Reds')
plot_decision_boundary(lambda x: svc3.predict(x), raw_data['X'], data['y'], 0.01)  # 画出c=100决策边界
ax.set_title('SVM (C=100) Decision Confidence')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend(loc='upper right')
plt.show()

# data3, find best params
raw_data = sio.loadmat('ex6data3.mat')
X = raw_data['X']
Xval = raw_data['Xval']
y = raw_data['y'].ravel()
yval = raw_data['yval'].ravel()

label0 = np.where(y == 0)
plt.scatter(X[label0, 0], X[label0, 1], s=50, marker='o', color='r', label='0')
label1 = np.where(y == 1)
plt.scatter(X[label1, 0], X[label1, 1], s=50, marker='+', color='b', label='1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper right')
plt.title("Dataset3")
plt.show()

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
best_score = 0
best_params = {'C': None, 'gamma': None}

for C in C_values:
    for gamma in gamma_values:
        svc=svm.SVC(C=C,gamma=gamma)
        svc.fit(X,y)
        score=svc.score(Xval,yval)

        if score>best_score:
            best_score=score
            best_params['C']=C
            best_params['gamma']=gamma

print("best_accuracy is : ", best_score)
print("best_params  are : ", best_params)


# 邮件分类器
spam_train=sio.loadmat('spamTrain')
spam_test=sio.loadmat('spamTest')
X=spam_train['X']
Xtest=spam_test['Xtest']
y=spam_train['y'].ravel()
ytest=spam_test['ytest'].ravel()
print(X.shape,y.shape,Xtest.shape,ytest.shape)

svc=svm.SVC()
svc.fit(X,y)
print('Training accuracy = {0}%'.format(np.round(svc.score(X, y) * 100, 2)))
print(' Test  accuracy   = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))
