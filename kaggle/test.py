# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:47:23 2019

@author: Dv00
"""

import pandas as pd
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.grid_search import GridSearchCV

data_train = pd.read_csv("C:/Users/Dv00/Desktop/new1datamining2019spring/trainSet.csv")
data_test = pd.read_csv("C:/Users/Dv00/Desktop/new1datamining2019spring/test set.csv")

x = data_train.iloc[:, 0:-1]
y = data_train.iloc[:, -1]

param_test2 = {'min_samples_split':list(range(2,102,10))}

gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=50, oob_score = True),
   param_grid = param_test2, scoring='roc_auc', iid=False, cv=5)
 
gsearch2.fit(x, y)
print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
