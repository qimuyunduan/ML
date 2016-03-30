# -*- coding: utf-8 -*-
import sys, math, random, collections
def shuffle(inFile):
    '''
        简单的乱序操作，用于生成训练集和测试集
    '''
    textLines = [line.strip() for line in open(inFile)]
    print "正在准备训练和测试数据，请稍后..."
    random.shuffle(textLines)
    num = len(textLines)
    trainText = textLines[:3*num/5]
    testText = textLines[3*num/5:]
    print "准备训练和测试数据准备完毕，下一步..."
    return trainText, testText

#总共有9种新闻类别，我们给每个类别一个编号
lables = ['A','B','C','D','E','F','G','H','I']