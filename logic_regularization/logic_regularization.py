# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def load_data(filename):
    data=[]
    file=open(filename)
    for line in file.readlines():
        linearr=line.strip().split(',')
        col_num=len(linearr)
        temp=[]
        for i in range(col_num):
            temp.append(float(linearr[i]))
        data.append(temp)
    return np.array(data)


data=load_data('ex2data2.txt')
print(data.shape)
print(data[:5])
X=data[:,:-1]
y=data[:,-1:]
print(X.shape)
print(y.shape)
print(X[:5])
print(y[:5])

label0=np.where(y.ravel()==0)
plt.scatter(X[label0,0],X[label0,1],marker='x',color='r',label=0)
label1=np.where(y.ravel()==1)
plt.scatter(X[label1,0],X[label1,1],marker='o',color='b',label=1)
plt.xlabel('0')
plt.ylabel('1')
plt.show()


# 添加多项式特征，例如x1*x2等
def mapFeature(X1,X2):
    degree=6
    out=np.ones((X1.shape[0],1))
    end=0
    for i in np.arange(1,degree+1,1):
        for j in np.arange(0,i+1,1):
            temp=X1**(i-j)*X2**(j)
            out=np.hstack((out,temp))
    return out


# regularized logistic
def sigmoid(x):
    return 1/(1+np.exp(-x))


def out(x,w):
    return sigmoid(np.dot(x,w))


def cost_reg(theta,XX,yy,reg):
    m=XX.shape[0]
    J=0
    grad=np.zeros((XX.shape[1],1))
    h=out(XX,theta)
    theta_1=theta[1:,:]
    J=-1* np.sum(yy*np.log(h) + (1-yy)*np.log((1-h))) / m + 0.5 * reg * theta_1.T.dot(theta_1) / m
    grad=XX.T.dot((h-yy))/m
    grad[1:,:]+=reg*theta_1/m
    return J,grad

X1=data[:,0:1]
X2=data[:,1:2]
print(X1.shape,X2.shape)
X_map=mapFeature(X1,X2)
print(X_map)
initial_theta=np.zeros((X_map.shape[1],1))
reg=1
# 测试
cost,grad = cost_reg(initial_theta,X_map,y,reg)
print('Cost at initial theta (zeros)=0.693')
print(cost)
print('Excepted: 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')
print(grad[:5])

test_theta = np.ones((X_map.shape[1],1))
cost1,grad1 =cost_reg(test_theta,X_map,y,10)
print('Excepted: cost=3.16')
print(cost1)


# 实现batch gradient decent
def bgd(X_train,y_train,theta,alpha=0.1,iters=5000,reg=1):
    J_history=[]
    for i in range(iters):
        cost,grad=cost_reg(theta,X_train,y_train,reg)
        theta=theta-alpha*grad
        J_history.append(float(cost))
        if i%200==0:
            print('iter=%d,cost=%f' % (i,cost))
    return theta,J_history


W = 0.001*np.random.randn(X_map.shape[1],1).reshape((-1,1))
theta,J_history = bgd(X_map,y,W)
plt.plot(J_history)
plt.xlabel('iters')
plt.ylabel('J_cost')
plt.show()


# plot the boundary
poly = PolynomialFeatures(6)
x1Min = X[:, 0].min()
x1Max = X[:, 0].max()
x2Min = X[:, 1].min()
x2Max = X[:, 1].max()
xx1, xx2 = np.meshgrid(np.linspace(x1Min, x1Max),np.linspace(x2Min, x2Max))
h1 = poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta)
h2 = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta)) #boundary
h1 = h1.reshape(xx1.shape)
h2 = h2.reshape(xx1.shape)
plt.contour(xx1, xx2, h1, [0.5], colors='b', linewidth=.5)
plt.contour(xx1, xx2, h2, [0.5], colors='black', linewidth=.5)
plt.show()


# 预测函数
def predict(X,theta):
    h=out(X,theta)
    y_predict=np.where(h>=0.5,1.0,0)
    return y_predict


p=predict(X_map,theta)
print("Expected accuracy (with lambda = 1): 83.1 (approx)")
print(np.mean(p==y))


