# -*- coding: utf-8 -*-

from math import e
from numpy import loadtxt, where, zeros, reshape, log, transpose, array
import matplotlib.pylab as pt

# load the dataset

data = loadtxt('./data1.txt', delimiter=',')
# 0:2 不包括2
X = data[:, 0:2]

y = data[:, 2]

yes = where(y == 1)
no = where(y == 0)

# ........作图分析相关数据...........#

# fig = pt.figure()
#
# pt.scatter(X[yes, 0], X[yes, 1], marker='o', c='b')
# pt.scatter(X[no, 0], X[no, 1], marker='x', c='r')
# pt.xlabel('Feature 1')
# pt.ylabel('Feature 2')
# pt.legend(['yes', 'no'])
# pt.show()


# .............建立模型................#

m=0

def sigmoid_func(para):
    den = 1.0 + e ** (-1.0 * para)
    gz = 1.0 / den
    return gz


def cost(func_para, X, y):
    m = X.shape[0]

    func_para = reshape(func_para, (len(func_para), 1))
    # transpose()求转置    dot() 矩阵点乘
    J = (1. / m) * (-transpose(y).dot(log(sigmoid_func(X.dot(func_para)))) - transpose(1 - y).dot(log(1 - sigmoid_func(X.dot(func_para)))))
    print J
    grad = transpose((1. / m) * transpose(sigmoid_func(X.dot(func_para)) - y).dot(X))

    return J[0][0]  # grad


def compute_grad(func_para, X, y):

    func_para.shape = (1, 3)
    grad = zeros(3)
    #  func_para.T 是func_para的转置
    h = sigmoid_func(X.dot(func_para.T))
    delta = h - y
    l = grad.size
    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = (1.0 / m) * sumdelta * -1
    func_para.shape = (3,)
    return grad

#....................对训练数据做一个预测............#

def predict(func_para, X):
    m, n = X.shape
    p = zeros(shape=(m,1))
    h = sigmoid_func(X.dot(func_para.T))
    # 对数据结果进行划分
    for it in range(0, h.shape[0]):
        if h[it]>0.5:
            p[it,0]=1
        else:
            p[it,0]=0
    return p

#计算精度
p = predict(array(zeros(2)),X)

print'Train Accuracy is : %f' % ((y[where(p == y)].size / float(y.size))*100.0)