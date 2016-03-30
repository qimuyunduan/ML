
#coding=utf-8
from sklearn import datasets
iris = datasets.load_iris()

print iris.data[:5]
#用贝叶斯分类器建模
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#填充数据进行拟合
result_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
right_num = (iris.target == result_pred).sum()
print("Total testing num :%d , naive bayes accuracy :%f" %(iris.data.shape[0], float(right_num)/iris.data.shape[0]))

