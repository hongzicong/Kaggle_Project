# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:27:08 2019

@author: Dv00
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
from catboost import CatBoostClassifier

data_train = pd.read_csv("C:/Users/Dv00/Desktop/new1datamining2019spring/trainSet.csv")
data_test = pd.read_csv("C:/Users/Dv00/Desktop/new1datamining2019spring/test set.csv")

x = data_train.iloc[:, 0:-1]
y = data_train.iloc[:, -1]

X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25)

model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.5, loss_function='Logloss',
                            logging_level='Verbose')

model.fit(X_train,y_train,eval_set=(X_validation, y_validation),plot=True)