# -*- coding: utf-8 -*-
#正则表达式
import re
#html 标签处理
from bs4 import BeautifulSoup
import pandas as pd

def wordlist(review):
    '''
    把评论转成次序列
    '''
    #去掉html标签
    text = BeautifulSoup(review,"lxml").get_text()
    #用正则表达式取出相关内容
    text = re.sub("[^a-zA-Z]"," ",text)
    #将文字转成小写,并转成词list
    words = text.lower().split()
    return words
train = pd.read_csv('./labeledTrainData.tsv',header=0,delimiter='\t',quoting=3)
test  = pd.read_csv('./testData.tsv',header=0,delimiter='\t',quoting=3)
#情感标签
print train
result = train['sentiment']
train_data = []
for i in xrange(0,len(train['review'])):
    train_data.append(" ".join(wordlist(train['review'][i])))
test_data  = []
for i in xrange(0,len(test['review'])):
    test_data.append(" ".join(wordlist(train['review'][i])))
#特征处理
#?如何把文本抽取为数值型特征
from sklearn.feature_extraction.text import TfidfVectorizer as TF

#初始化TF对象,去停用词,加2元语言模型
tf = TF(min_df=3,max_features=None,strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,2),use_idf=1,
        sublinear_tf=1,smooth_idf=1,stop_words='english')

#将训练集和测试集合并,进行向量化操作
data = train_data+test_data
len_train = len(train_data)

#填充数据,词到向量的过程
tf.fit(data)
data = tf.transform(data)
print data
#恢复成训练集和测试集两部分
v_train = data[:len_train]
v_test  = data[len_train:]

#建模 多项式贝叶斯模型
from sklearn.naive_bayes import MultinomialNB as MNB

model= MNB()
#填充数据,训练模型
model.fit(v_train,result)

MNB(alpha=1.0,class_prior=None,fit_prior=True)

from sklearn.cross_validation import cross_val_score
import numpy as np
print "多项式贝叶斯分类器10折交叉验证得分: ",np.mean(cross_val_score(model,v_train,result,cv=10,scoring='roc_auc'))

#建模 逻辑回归
from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV

#set parameter for Gridsearch
grid_value = {'C':[30]}
modelLR = GridSearchCV(LR(penalty='l2',dual=True,random_state=0),grid_value,scoring='roc_auc',cv=10)
#填充数据,模型训练
modelLR.fit(v_train,result)

#10折交叉验证
GridSearchCV(cv=10, estimator=LR(C=1.0, class_weight=None, dual=True,
        fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=0, tol=0.0001),
        fit_params={}, iid=True, n_jobs=1,param_grid={'C': [30]}, pre_dispatch='2*n_jobs', refit=True,scoring='roc_auc', verbose=0)
print "逻辑回归10折交叉验证得分: ", modelLR.grid_scores_









