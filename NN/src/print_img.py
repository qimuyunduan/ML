#coding=utf-8
# #以读取mnist.pkl.gz为例
import cPickle, gzip
f = gzip.open('../data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
print test_set