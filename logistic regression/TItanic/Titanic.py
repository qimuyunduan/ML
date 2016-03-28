# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:23:14 2016

@author: qimuyunduan
"""

import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as pt
from sklearn.ensemble import  RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.learning_curve import  learning_curve
from sklearn.ensemble import BaggingRegressor

data_train = pd.read_csv("./train.csv")
#data_train.info() #数据相关信息
#print data_train.describe()# 数据详细信息

# 1   ............作图分析相关数据...............#
#
# fig = pt.figure()
# fig.set(alpha=0.3) #设定图表的alpha
#
# pt.subplot2grid((2,3),(0,0))
# data_train.Survived.value_counts().plot(kind="bar") #柱狀图
# pt.title(u"1 is survived")
# pt.ylabel(u"person number")
#
# pt.subplot2grid((2,3),(0,1))
# data_train.Pclass.value_counts().plot(kind="bar")#柱状图
# pt.title(u"person class")
# pt.ylabel(u"person number")
#
#
# pt.subplot2grid((2,3),(0,2))
# pt.scatter(data_train.Survived,data_train.Age)
# pt.ylabel(u"age")
# pt.grid(b=True,which='major',axis='y')
# pt.title(u"age & survived")
#
# pt.subplot2grid((2,3),(1,0),colspan=2)
# data_train.Age[data_train.Pclass==1].plot(kind='kde')
# data_train.Age[data_train.Pclass==2].plot(kind='kde')
# data_train.Age[data_train.Pclass==3].plot(kind='kde')
# pt.xlabel(u"age")
# pt.ylabel(u"person density")
# pt.title(u"age & pclass")
# pt.legend((u'fisrt class',u'second class',u'three class'),loc='best')
#
#
# pt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# pt.title(u'percount & boardgate')
# pt.ylabel(u'person numbers')
# pt.show()
#
#
#
# #...............获救情况与乘客等级的关系................#
#
# fig2 = pt.figure()
# fig2.set(alpha=0.2)
#
# died = data_train.Pclass[data_train.Survived==0].value_counts()
# alive = data_train.Pclass[data_train.Survived==1].value_counts()
# df=pd.DataFrame({u'survived':alive,u'died':died})
# df.plot(kind='bar',stacked=True)
# pt.title(u"Survived & pclass")
# pt.xlabel(u'passenger & pclass')
# pt.ylabel(u'number')
# pt.show()
#
#
# #...............获救情况与性别的关系..................#
#
# fig3 = pt.figure()
# fig3.set(alpha=0.2)
#
# died=data_train.Pclass[data_train.Survived==0].value_counts()
# alive = data_train.Pclass[data_train.Survived==1].value_counts()
# df=pd.DataFrame({u'survived':alive,u'died':died})
# df.plot(kind='bar',stacked=True)
# pt.title(u'Survived & age')
# pt.xlabel(u'passenger class')
# pt.ylabel(u'numbers')
# pt.show()
#
# #................获救情况与性别和舱位级别...............#
#
# fig4 = pt.figure()
# fig4.set(alpha=0.4)
# pt.title(u'survived & Age and Pclass')
#
# ax1 = fig4.add_subplot(141)
# data_train.Survived[data_train.Sex=='female'][data_train.Pclass!=3].value_counts().plot(kind='bar',label='female highclass',color='#FA2479')
# ax1.set_xticklabels([u'survived',u'died'],rotation=0)
# ax1.legend([u'female highclass'],loc='best')
#
# ax2 = fig4.add_subplot(142,sharey=ax1)
# data_train.Survived[data_train.Sex=='female'][data_train.Pclass==3].value_counts().plot(kind='bar',label='female lowclass',color='pink')
# ax2.set_xticklabels([u'survived',u'died'],rotation=0)
# pt.legend([u'female lowlass'],loc='best')
#
# ax3 = fig4.add_subplot(143,sharey=ax1)
# data_train.Survived[data_train.Sex=='male'][data_train.Pclass!=3].value_counts().plot(kind='bar',label='male highclass',color='#FA2479')
# ax3.set_xticklabels([u'survived',u'died'],rotation=0)
# pt.legend([u'male highclass'],loc='best')
#
# ax4 = fig4.add_subplot(144,sharey=ax1)
# data_train.Survived[data_train.Sex=='male'][data_train.Pclass==3].value_counts().plot(kind='bar',label='male lowclass',color='pink')
# ax4.set_xticklabels([u'survived',u'died'],rotation=0)
# pt.legend([u'male lowclass'],loc='best')
#
# pt.show()
#
# #...............获救人数与登船港口的关系.................#
#
#
# fig5 = pt.figure()
# fig5.set(alpha=0.5)
#
# alive = data_train.Embarked[data_train.Survived==1].value_counts()
# died  = data_train.Embarked[data_train.Survived==0].value_counts()
# df    = pd.DataFrame({u'survived':alive,u'died':died})
# df.plot(kind='bar',stacked='True')
# pt.title(u'survived & Embarked')
# pt.xlabel(u'Embarked')
# pt.ylabel(u'number')
# pt.show()
#
#
# #.................获救情况与堂兄,弟妹的关系................#
# #
# fig6 = pt.figure()
# fig6.set_alpha(0.3)
# g=data_train.groupby(['SibSp','Survived'])
# pd.DataFrame(g.count()['PassengerId']).plot(kind='bar',stacked='True')
# pt.ylabel(u'number')
# pt.title(u'survived numbers')
# pt.show()
#
# # 看一下Cabin中204个乘客的相关信息
# #print data_train.Cabin.value_counts()
#
#
# #.................看Cabin有无值进行分类,再看其中的获救情况......#
#
# fig7 = pt.figure()
# fig7.set(alpha='0.3')
#
# alived = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# died   = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df=pd.DataFrame({u'have':alived,u'no':died})
# df.plot(kind='bar',stacked='True')
# pt.title(u'survived & cabin ')
# pt.xlabel(u'cabin null or not null')
# pt.ylabel(u'numbers')
# pt.show()



# 2 .............数据预处理.................................#

#使用RandomForestClassifier 填补缺失的属性.................#



def fill_missing_age(df):
    #把已有的数值型特征取出来丢进Random Forest Regressor 中
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    #print age_df

    #把乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # print "known_age......."
    # print known_age
    # print "unknown age ........"
    # print unknown_age

    # 目标年龄
    y=known_age[:,0]

    # 特征属性值
    x=known_age[:,1:]

    #fit 到RandomForestRegressor之中
    RFR=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    RFR.fit(x,y)

    #用得到的模型进行未知年龄结果预测
    predictedAge= RFR.predict(unknown_age[:,1::])

    #用预测的结果填补原缺失数据
    df.loc[(df.Age.isnull()),'Age']=predictedAge
    return df,RFR

def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin']='Yes'
    df.loc[(df.Cabin.isnull()),'Cabin']='No'
    return df

data_train,RFR = fill_missing_age(data_train)
# print "data_train after fill age....."
# print data_train
data_train = set_cabin_type(data_train)
#print "data_train after set cabin_type......"
#print data_train

#..................特征因子化..................#

flatten_cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
flatten_embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
flatten_sex = pd.get_dummies(data_train['Sex'],prefix='Sex')
flatten_pclass = pd.get_dummies(data_train['Pclass'],prefix='Pclass')

df=pd.concat([data_train,flatten_cabin,flatten_embarked,flatten_sex,flatten_pclass],axis=1)
df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)

#..................将age和Fare特征规范化..........#

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'],age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'],fare_scale_param)



# 3.....................建立模型....................#

#取出我们想要的属性
wanted_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = wanted_df.as_matrix()
#print train_np

# y Survived结果
y=train_np[:,0]

# x 特征属性值

x=train_np[:,1:]

# fit 到RandomForestRegressor之中
model=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
model.fit(x,y)
#print model


#...................导入测试集合进行预处理.................#

data_test = pd.read_csv("./test.csv")
data_test.loc[(data_test.Fare.isnull(),'Fare')]=0

# fill value of missing attributes and flatten attributes

temp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
null_age = temp_df[data_test.Age.isnull()].as_matrix()

X = null_age[:,1:]

predictedAge = RFR.predict(X)
data_test.loc[(data_test.Age.isnull()),'Age']=predictedAge

data_test = set_cabin_type(data_test)
flatten_cabin = pd.get_dummies(data_test['Cabin'],prefix='Cabin')
flatten_pclass = pd.get_dummies(data_test['Pclass'],prefix='Pclass')
flatten_sex = pd.get_dummies(data_test['Sex'],prefix='Sex')
flatten_embarked = pd.get_dummies(data_test['Embarked'],prefix='Embarked')

df_test = pd.concat([data_test,flatten_cabin,flatten_embarked,flatten_sex,flatten_pclass],axis=1)
df_test.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace = True)
df_test['Age_scaled']=scaler.fit_transform(df_test['Age'],age_scale_param)
df_test['Fare_scaled']=scaler.fit_transform(df_test['Fare'],fare_scale_param)
#print df_test

# 预测取出结果

#取出相关属性
# test=df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = model.predict(test)
# #print predictions
# result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32 )})
#
# result.to_csv("./test_result.csv",index=False)


# 4......................属性观察和系统优化.................#

#print pd.DataFrame({"columns":list(wanted_df.columns)[1:],"coef":list(model.coef_.T)})





# 5.........................交叉验证................#

model = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.as_matrix()[:,1:]

Y = all_data.as_matrix()[:,0]
#print cross_validation.cross_val_score(model,X,Y,cv=5)


#将原始训练数据进行分割---- 训练数据:CV数据 = 8:2

splited_train,splited_cv = cross_validation.train_test_split(df,test_size=0.2,random_state=0)
train_df = splited_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

#生成模型
model = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
model.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])

# 对splited_cv数据进行预测

cv_df = splited_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions = model.predict(cv_df.as_matrix()[:,1:])

origin_data = pd.read_csv('./train.csv')

#查看判断错误的数据
bad_cases = origin_data.loc[origin_data['PassengerId'].isin(splited_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
#print bad_cases



# 5 ..................学习曲线..............#

# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
# def plot_learning_curve(estimator, title, x, y, ylimit=None, cv=None, n_jobs=1,train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
#
#     """
#     画出data在某模型上的learning curve.
#     参数解释
#     ----------
#     estimator : 你用的分类器。
#     title : 表格的标题。
#     x : 输入的feature，numpy类型
#     y : 输入的target vector
#     ylimit : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
#     cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
#     n_jobs : 并行的的任务数(默认1)
#     """
#     train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
#
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#
#     if plot:
#         pt.figure()
#         pt.title(title)
#         if ylimit is not None:
#             pt.ylim(ylimit)
#         pt.xlabel(u"training size")
#         pt.ylabel(u"scores")
#         pt.grid()
#         pt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
#                          alpha=0.1, color="b")
#         pt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
#                          alpha=0.1, color="r")
#         pt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"scores on data_train")
#         pt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"scores on data_cv")
#
#         pt.legend(loc="best")
#         pt.draw()
#         pt.show()
#     midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
#     diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
#     return midpoint, diff
#
# plot_learning_curve(model, u"learning  curve", X, Y)


# 6...................模型融合.....................#


train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

x = train_np[:, 1:]


# fit 到BaggingRegressor之中
model = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_model = BaggingRegressor(model,n_estimators=20,max_samples=0.8,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=-1)
bagging_model.fit(x,y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = bagging_model.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32 )})
result.to_csv('./result.csv',index=False)


