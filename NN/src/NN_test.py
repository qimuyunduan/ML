#coding=utf-8
import mnist_loader
import network
#将数据分为:训练集,交叉验证集,测试集
training_data,validation_data,test_data=mnist_loader.load_data_wrapper()
#生成神经网络对象,三层,每层节点依次为(784,30,10)
net=network.Network([784,100,10])
#用(mini-batch)梯度下降法训练神经网络（权重与偏移），并生成测试结果。
#训练回合数=30, 用于随机梯度下降法的最小样本数=10，学习率=3.0
net.SGD(training_data,30,10,3,test_data=test_data)