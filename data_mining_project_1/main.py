# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:47:23 2019

@author: Dv00
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
import time as tm

RANDOM_SEED = 106

time_begin = tm.time()

data_train = pd.read_csv("C:/Users/Dv00/Desktop/new1datamining2019spring/trainSet.csv")
data_test = pd.read_csv("C:/Users/Dv00/Desktop/new1datamining2019spring/test set.csv")

#data_train = pd.read_csv("../../new1datamining2019spring/trainSet.csv")
#data_test = pd.read_csv("../../new1datamining2019spring/test set.csv")

#data_train['F9'] = np.log1p(data_train['F9'])
#data_test['F9'] = np.log1p(data_test['F9'])

x = data_train.iloc[:, 0:-1]
y = data_train.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 438)

# 使用随机森林

rfc = RandomForestClassifier(n_estimators=2000, oob_score = True, min_samples_split=190, min_samples_leaf=8,
                             criterion="gini", n_jobs=-1, random_state = 321)

rfc.fit(x_train, y_train)

rfc_y_predict = rfc.predict(x_test)

#获取特征的重要性
#importances = rfc.feature_importances_
#indices = np.argsort(importances)[::-1]
#cols_name = data_train.columns[:-1]
#for f in range(x_train.shape[1]):
#    print("%2d) %-*s %f" % (f + 1,30,cols_name[indices[f]],importances[indices[f]]))

print(rfc.score(x_test, y_test))
 
print(classification_report(y_test, rfc_y_predict,digits=4))

# x_test = pd.concat([data_test.iloc[:, :10], data_test.iloc[:, 11:]], axis = 1)
# result = rfc.predict(data_test)
# pd.DataFrame(result,columns=['Predicted'],index=list(range(1,len(result) + 1))).to_csv('result.csv')

print(tm.time() - time_begin)