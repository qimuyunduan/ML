# -*- coding: utf-8 -*-

from math import e


from numpy import loadtxt, where, zeros, reshape, log, transpose, array, ones, append, linspace
import matplotlib.pylab as pt
from scipy.optimize import fmin_bfgs

data = loadtxt('./data2.txt',delimiter=',')

X = data[:,0:2]
y = data[:,2]

pos = where(y==1)
neg = where(y==0)

#.................作图分析相关数据...................#


# fig = pt.figure()
# fig.set_alpha(alpha=0.4)
#
# pt.scatter(X[pos,0],X[pos,1],marker='o',c='b')
# pt.scatter(X[neg,0],X[neg,1],marker='x',c='r')
# pt.title("data figure")
# pt.xlabel(u'feature one')
# pt.ylabel(u'feature two')
# pt.legend(['1','0'])
# pt.show()



#.............建立相应模型.................#


def sigmoid_func(X):

    den = 1.0 + e ** (-1.0 * X)
    gz = 1.0 / den
    return gz


def map_feature(x1, x2):

    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    mapped_fea = ones(shape=(x1[:, 0].size, 1))

    m, n = mapped_fea.shape

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            mapped_fea = append(out, r, axis=1)
    return mapped_fea


mapped_fea = map_feature(X[:, 0], X[:, 1])


def cost_func(theta, X, y, l):

    h = sigmoid_func(X.dot(theta))
    thetaR = theta[1:, 0]
    J = (1.0 / m) * ((-y.T.dot(log(h))) - ((1 - y.T).dot(log(1.0 - h)))) \
         (l / (2.0 * m)) * (thetaR.T.dot(thetaR))
    delta = h - y
    sum_delta = delta.T.dot(X[:, 1])
    grad1 = (1.0 / m) * sum_delta
    XR = X[:, 1:X.shape[1]]
    sum_delta = delta.T.dot(XR)
    grad = (1.0 / m) * (sum_delta + l * thetaR)
    out = zeros(shape=(grad.shape[0], grad.shape[1] + 1))
    out[:, 0] = grad1
    out[:, 1:] = grad
    return J.flatten(), out.T.flatten()


m, n = X.shape
y.shape = (m, 1)
it = map_feature(X[:, 0], X[:, 1])

# Initialize theta parameters
initial_theta = zeros(shape=(it.shape[1], 1))
# Use regularization and set parameter lambda to 1
l = 1
# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = cost_func(initial_theta, it, y, l)


def decorated_cost(theta):

    return cost_func(theta, it, y, l)


print fmin_bfgs(decorated_cost, initial_theta, maxfun=500)

# Plot Boundary

u = linspace(-1, 1.5, 50)
v = linspace(-1, 1.5, 50)

z = zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i, j] = (map_feature(array(u[i]), array(v[j])).dot(array(theta)))
z = z.T
pt.contour(u, v, z)
pt.title('lambda = %f' % l)
pt.xlabel('Microchip Test 1')
pt.ylabel('Microchip Test 2')
pt.legend(['y = 1', 'y = 0', 'Decision boundary'])
pt.show()


def predict(theta, X):
    '''''Predict whether the label
    is 0 or 1 using learned logistic
    regression parameters '''
    m, n = X.shape
    p = zeros(shape=(m, 1))
    h = sigmoid_func(X.dot(theta.T))
    for it in range(0, h.shape[0]):
        if h[it] > 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0
    return p


#  Compute accuracy on our training set

p = predict(array(theta), it)
print'Train Accuracy: %f' % ((y[where(p == y)].size / float(y.size)) * 100.0)
