# -*- coding:utf-8 -*-
# 本文是BP神经网络算法，实现手写字体识别
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

digits=load_digits()
print(digits.keys())
data=digits.data
target=digits.target
print(data.shape)
print(target.shape)
print("the image 15 is",target[15])
plt.gray()
plt.matshow(digits.images[15])
plt.show()

classes=['0','1','2','3','4','5','6','7','8','9']
num_classes=len(classes)
sample_per_class=1
for y,cla in enumerate(classes):
    idxs=np.flatnonzero(target==y)
    idxs=np.random.choice(idxs,sample_per_class,replace=False)
    for i,idx in enumerate(idxs):
        plt_idx=i*num_classes+y+1
        plt.subplot(sample_per_class,num_classes,plt_idx)
        plt.imshow(digits.images[idx].astype('uint8'))
        plt.axis('off')
        if i==0:
            plt.title(cla)
plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid的导数 = x * (1 - x)
def dsigmoid(x):
    return x * (1 - x)


# 一个神经网络的类，3层（只有一个隐含层），对于各层单元数量，输入输出由提供的数据与目标分类决定，隐含层有部分经验公式
class NeuralNetwork(object):
    def __init__(self,input_size,hidden_size,output_size):
        self.W1=0.01*np.random.randn(input_size,hidden_size)  # np.random.randn()从标准正态分布中返回一个或多个样本值
        self.b1=np.zeros(hidden_size)
        self.W2=0.01*np.random.randn(hidden_size,output_size)
        self.b2=np.zeros(output_size)

    def loss(self,X,y,reg=0.01):  #  y 目标（期望）输出 ，reg正则化系数
        num_train,num_feature=X.shape
        # forward前向传播 简单的3层神经网络
        a1=X
        a2=sigmoid(a1.dot(self.W1)+self.b1)  # hidden layer:N*H
        a3=sigmoid(a2.dot(self.W2)+self.b2)  # output layer:N*C
        loss=-np.sum(y*np.log(a3)+(1-y)*np.log((1-a3)))/num_train
        loss+=0.5*reg * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2)) / num_train  # 正则化后的神经网络代价函数J（theta）
        # BP反向传播过程
        error3=a3-y
        dw2=a2.T.dot(error3)+reg*self.W2
        db2=np.sum(error3,axis=0)

        error2=error3.dot(self.W2.T)*dsigmoid(a2)
        dw1=a1.T.dot(error2)+reg*self.W1
        db1=np.sum(error2,axis=0)

        dw1/=num_train
        dw2/=num_train
        db1/=num_train
        db2/=num_train

        return loss,dw1,dw2,db1,db2

    # 预测一下，多分类的预测，取得最大值的下标为分类结果
    def predict(self,X_test):
        a2=sigmoid(X_test.dot(self.W1)+self.b1)
        a3=sigmoid(a2.dot(self.W2)+self.b2)
        y_pred=np.argmax(a3,axis=1)
        return y_pred

    # 训练过程,并且每500次，统计一下train_acc和val_acc
    def train(self,X,y,y_train,X_val,y_val,learn_rate=0.01,num_iters=10000):
        batch_size=150
        num_train=X.shape[0]
        loss_list=[]
        accuracy_train=[]
        accuracy_val=[]

        for i in range(num_iters):
            batch_index=np.random.choice(num_train,batch_size,replace=True)  # 生成的随机数中可以有重复的数值
            X_batch=X[batch_index]
            y_batch=y[batch_index]
            y_train_batch=y_train[batch_index]

            loss,dw1,dw2,db1,db2=self.loss(X_batch,y_batch)
            loss_list.append(loss)

            # update the weight
            self.W1 += -learn_rate * dw1
            self.W2 += -learn_rate * dw2
            self.b1 += -learn_rate * db1
            self.b2 += -learn_rate * db2

            if i%500==0:
                print("i=%d,loss=%f" % (i, loss))
                # record the train accuracy and validation accuracy
                train_acc=np.mean(y_train_batch==self.predict(X_batch))
                val_acc=np.mean(y_val==self.predict(X_val))
                accuracy_train.append(train_acc)
                accuracy_val.append(val_acc)
        return loss_list,accuracy_train,accuracy_val


# 定义一下数值梯度，用于梯度检查
def eval_numerical_gradient(f, x, verbose=True, h=0.00001):  # 计算数值梯度，h 步长
    fx=f(x)
    grad=np.zeros_like(x)  # 将x中的每个元素变为0，并赋值到grad
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])  # op_flags=['readwrite']有读写权限
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index  # 返回元素的索引
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore
        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope  斜率
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension，继续下一个元素
    return grad


# 定义一下比较函数，用于两个梯度的对比（前面的dw和梯度检验的grad）
def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# 模拟一个神经网络，进行梯度检查，判断模型正确性
input_size = 4
hidden_size = 10
output_size = 3
num_train = 5


def init_toy_model():
    np.random.seed(0)  # 函数可以保证生成的随机数具有可预测性。可预测性是指相同的种子（seed值）所产生的随机数是相同的
    return NeuralNetwork(input_size, hidden_size, output_size)


def init_toy_data():
    np.random.seed(1)
    XX = 10 * np.random.randn(num_train, input_size)  # 返回样本，具有标准正态分布
    yy = np.array([0, 1, 2, 2, 1])   # 为了对应num_train = 5,也就是5个标签
    return XX, yy


net = init_toy_model()
XX, yy = init_toy_data()
yy_label = LabelBinarizer().fit_transform(yy)   # 每个实例数据显示5个标签，这里可以debug看一下

loss, dw1, dw2, db1, db2 = net.loss(XX, yy_label)  # dw1(dw1_)=4*10 ,dw2(dw2_)=10*3,db1(db_1)=1*10;db2(db_2)=1*3
print("dw1=%s" % dw1)
f = lambda W: net.loss(XX, yy_label)[0]  # 定义了一个匿名函数，w是入口参数，所以相当于f(w) = loss
dw1_ = eval_numerical_gradient(f, net.W1, verbose=False)
dw2_ = eval_numerical_gradient(f, net.W2, verbose=False)
db1_ = eval_numerical_gradient(f, net.b1, verbose=False)
db2_ = eval_numerical_gradient(f, net.b2, verbose=False)

print('%s max relative error: %e' % ('W1', rel_error(dw1, dw1_)))
print('%s max relative error: %e' % ('W2', rel_error(dw2, dw2_)))
print('%s max relative error: %e' % ('b1', rel_error(db1, db1_)))
print('%s max relative error: %e' % ('b2', rel_error(db2, db2_)))


X = data
y = target
X_mean = np.mean(X, axis=0)  # 数据的每一维的均值=0
X -= X_mean
# 分隔数据集，分为train，val（交叉验证集）,test集合
X_data, X_test, y_data, y_test = train_test_split(X, y, test_size=0.2)   # 测试集占总样本百分比
# split the train and tne validation
X_train = X_data[0:1000]   # X_data是全部数据集的80%，1437*64，这里取前1000行，1000*64，训练
y_train = y_data[0:1000]   # 1*1000
X_val = X_data[1000:-1]  # 436*64
y_val = y_data[1000:-1]
print(X_train.shape, X_val.shape, X_test.shape)
# 进行标签二值化，1-->[0,1,0,0,0,0,0,0,0,0]   5=[0,0,0,0,0,1,0,0,0,0]
y_train_label = LabelBinarizer().fit_transform(y_train)
classify = NeuralNetwork(X.shape[1], 100, 10)  # 创建一个NeuralNetwork类的实例（类实例化）
# 数据准备完毕，开始训练
print('start')
loss_list, accuracy_train, accuracy_val = classify.train(X_train, y_train_label, y_train, X_val, y_val)
print('end')

# 可视化训练结果
plt.subplot(211)
plt.plot(loss_list)
plt.title('train loss')
plt.xlabel('iters')
plt.ylabel('loss')

plt.subplot(212)
plt.plot(accuracy_train, label='train_acc', color='red')
plt.plot(accuracy_val, label='val_acc', color='black')
plt.xlabel('iters')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()

# 预测并看看正确率
y_pred = classify.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("the accuracy is ", accuracy)

# 随机挑选样本
m, n = data.shape
example_size = 10
example_index = np.random.choice(m, example_size)
print(example_index)
for i, idx in enumerate(example_index):
    print("%d example is number %d,we predict it as %d"  % (i, target[idx], classify.predict(X[idx, :].reshape(1, -1))))

# 从mat文件导入数据
weight = sio.loadmat('ex4weights')  # 5000*400
Theta1 = weight["Theta1"]
Theta2 = weight["Theta2"]

data = sio.loadmat('ex4data1')
X = data["X"]
y = data["y"]
print(X.shape[1])
print(X.shape[0])
print(y.shape)

m = X.shape[0]
X1 = np.hstack((np.ones((m, 1)), X))
print(X1.shape)
