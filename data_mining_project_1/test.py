# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:47:23 2019

@author: Dv00
"""
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

from sklearn.grid_search import GridSearchCV

import time as tm
import random

RANDOM_SEED = 120

time_begin = tm.time()

data_train = pd.read_csv("C:/Users/Dv00/Desktop/new1datamining2019spring/trainSet.csv")
data_test = pd.read_csv("C:/Users/Dv00/Desktop/new1datamining2019spring/test set.csv")

#data_train = pd.read_csv("../../new1datamining2019spring/trainSet.csv")
#data_test = pd.read_csv("../../new1datamining2019spring/test set.csv")


x = data_train.iloc[:, 0:-1]
y = data_train.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 106)

param_test = {'min_samples_split':[i for i in range(100,200,10)], 'min_samples_leaf':[i for i in range(1,10,2)]}

gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 50, criterion="gini", n_jobs=-1 ,
                                                           oob_score=True, random_state=105), param_grid = param_test, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(x, y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

print(tm.time() - time_begin)