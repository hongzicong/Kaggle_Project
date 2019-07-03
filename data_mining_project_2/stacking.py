# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:47:30 2019

@author: Dv00
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

RANDOM_SEED = 106

time_begin = tm.time()

#data_train = pd.read_csv("C:/Users/Dv00/Desktop/dm2019springproj2/train.csv")
#data_test = pd.read_csv("C:/Users/Dv00/Desktop/dm2019springproj2/test.csv")

data_train_1 = pd.read_csv("../../dm2019springproj2/train1.csv", header=None)
data_train_2 = pd.read_csv("../../dm2019springproj2/train2.csv", header=None)
data_train_3 = pd.read_csv("../../dm2019springproj2/train3.csv", header=None)
data_train_4 = pd.read_csv("../../dm2019springproj2/train4.csv", header=None)
data_train_5 = pd.read_csv("../../dm2019springproj2/train5.csv", header=None)

data_test_1 = pd.read_csv("../../dm2019springproj2/test1.csv", header=None)
data_test_2 = pd.read_csv("../../dm2019springproj2/test2.csv", header=None)
data_test_3 = pd.read_csv("../../dm2019springproj2/test3.csv", header=None)
data_test_4 = pd.read_csv("../../dm2019springproj2/test4.csv", header=None)
data_test_5 = pd.read_csv("../../dm2019springproj2/test5.csv", header=None)
data_test_6 = pd.read_csv("../../dm2019springproj2/test6.csv", header=None)

label_1 = pd.read_csv("../../dm2019springproj2/label1.csv", header=None)
label_2 = pd.read_csv("../../dm2019springproj2/label2.csv", header=None)
label_3 = pd.read_csv("../../dm2019springproj2/label3.csv", header=None)
label_4 = pd.read_csv("../../dm2019springproj2/label4.csv", header=None)
label_5 = pd.read_csv("../../dm2019springproj2/label5.csv", header=None)

x = pd.concat([data_train_1, data_train_2, data_train_3, data_train_4, data_train_5], ignore_index=True)
y = pd.concat([label_1, label_2, label_3, label_4, label_5], ignore_index=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = RANDOM_SEED)

### 第一层模型
clfs = [GBDT(n_estimators=100),
       RF(n_estimators=100),
       ET(n_estimators=100),
       ADA(n_estimators=100)
]
X_train_stack  = np.zeros((x_train.shape[0], len(clfs)))
X_test_stack = np.zeros((x_test.shape[0], len(clfs)))

skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1)
for i, clf in enumerate(clfs):
    X_stack_test_n = np.zeros((x_test.shape[0], 6))
    for j,(train_index,test_index) in enumerate(skf.split(x_train, y_train)):
        tr_x = x_train[train_index]
        tr_y = y_train[train_index]
        clf.fit(tr_x, tr_y)
        #生成stacking训练数据集
        X_train_stack [test_index, i] = clf.predict_proba(x_train[test_index])[:,1]
        X_stack_test_n[:,j] = clf.predict_proba(x_test)[:,1]
    #生成stacking测试数据集
    X_test_stack[:,i] = X_stack_test_n.mean(axis=1) 

###第二层模型LR
clf_second = LogisticRegression(solver="lbfgs")
clf_second.fit(X_train_stack, y_train)
pred = clf_second.predict_proba(X_test_stack)[:,1]

predictions = [round(value) for value in pred]

score = r2_score(y_test, predictions)
print("R2 Score: %.3f%%" % (score * 100.0))