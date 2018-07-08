# -*- coding:utf-8 -*-
import numpy as np


def load_data(filename):
    data = []
    file = open(filename)
    for line in file.readlines():
        linearr = line.strip().split(',')  # 去除空白和逗号“,”strip 剥离
        col_num = len(linearr)
        temp = []
        for i in range(col_num):
            temp.append(float(linearr[i]))
        data.append(temp)
    return np.array(data)


data = load_data('ex1data1.txt')
X = data[:, :-1]  # 取不是最后一列的前面的列
y = data[:, -1:]
data_1 = data[1:, :]  # 从所有行的第2行开始，取所有列
print(data.shape)
print(X.shape)
print(X[:5])
print(y.shape)
print(y[:5])
print(data_1.shape)
print(data_1[:5])
