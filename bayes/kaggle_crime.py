# -*- coding: utf-8 -*-
import pandas as pd

train_data = pd.read_csv('./train.csv',parse_dates = ['Dates'])
test_data = pd.read_csv('./test.csv',parse_dates = ['Dates'])


#特征因子化,文本型特征转化为数字型特征
from sklearn import preprocessing
#用LabelEncoder 对不同的犯罪类型编号(把文本分类标签数字化)
labelCrime = preprocessing.LabelEncoder()
crime = labelCrime.fit_transform(train_data.Category)
#对星期几,街区,小时...因子化
days = pd.get_dummies(train_data.DayOfWeek)
#发生区域
district = pd.get_dummies(train_data.PdDistrict)
#获取Dates属性中的hour
hour = train_data.Dates.dt.hour

hour = pd.get_dummies(hour)

#组合特征
train_data = pd.concat([hour,days,district],axis =1)
#print train_data
train_data['crime'] = crime


#对测试数据做同样的处理
#属性因子化
days = pd.get_dummies(test_data.DayOfWeek)
district = pd.get_dummies(test_data.PdDistrict)
hour = test_data.Dates.dt.hour
hour = pd.get_dummies(hour)
test_data = pd.concat([hour,days,district],axis=1)



import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import time

#取星期几和街区作为分类器输入特征
features = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday',
            'BAYVIEW','CENTRAL', 'INGLESIDE', 'MISSION',
 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
#分割训练集 train_size 指定训练集占的比例 0.6
train,test = train_test_split(train_data,train_size=.60)

#建模
model = BernoulliNB()
startTime = time.time()
model.fit(train[features],train['crime'])
#建模完成时间
costTime = time.time()-startTime
#测试结果
predicted = np.array(model.predict_proba(test[features]))
loss=log_loss(test['crime'],predicted)
print "朴素贝叶斯建模耗时 %f 秒" %(costTime)
print "朴素贝叶斯log损失为 %f" %(loss)

#逻辑回归建模
model = LogisticRegression(C=.01)
start= time.time()
model.fit(train[features], train['crime'])
costTime = time.time() - start
predicted = np.array(model.predict_proba(test[features]))
loss=log_loss(test['crime'], predicted)
print "逻辑回归建模耗时 %f 秒" %(costTime)
print "逻辑回归log损失为 %f" %(loss)


print "添加犯罪小时特征后....."
# 添加犯罪的小时时间点作为特征
features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
'Wednesday', 'INGLESIDE', 'MISSION','BAYVIEW', 'CENTRAL',
'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']

hourFea = [x for x in range(0,24)]
features = features + hourFea

# 分割训练集(3/5)和测试集(2/5)
train,test = train_test_split(train_data,train_size=.60)
#建模
model = BernoulliNB()
startTime = time.time()
model.fit(train[features],train['crime'])
#建模完成时间
costTime = time.time()-startTime
#测试结果
predicted = np.array(model.predict_proba(test[features]))
loss=log_loss(test['crime'],predicted)
print "朴素贝叶斯建模耗时 %f 秒" %(costTime)
print "朴素贝叶斯log损失为 %f" %(loss)

#逻辑回归建模
model = LogisticRegression(C=.01)
start= time.time()
model.fit(train[features], train['crime'])
costTime = time.time() - start
predicted = np.array(model.predict_proba(test[features]))
loss=log_loss(test['crime'], predicted)
print "逻辑回归建模耗时 %f 秒" %(costTime)
print "逻辑回归log损失为 %f" %(loss)