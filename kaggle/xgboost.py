# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:30:28 2019

@author: Dv00
"""

import pandas as pd
import time as tm

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

time_begin = tm.time()

data_train = pd.read_csv("C:/Users/Dv00/Desktop/new1datamining2019spring/trainSet.csv")
data_test = pd.read_csv("C:/Users/Dv00/Desktop/new1datamining2019spring/test set.csv")

x = data_train.iloc[:, 0:-1]
y = data_train.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 106)

params = {
    'n_estimators': 500,
    'learning_rate': 0.15,
    'max_depth': 18,
    'seed': 1000,
    'nthread': -1,
}

model = XGBClassifier(**params)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print(classification_report(y_test, predictions,digits=4))

print("Time used: %f" % (tm.time() - time_begin))

#result = model.predict(data_test)
#pd.DataFrame(result,columns=['Predicted'],index=list(range(1,len(result) + 1))).to_csv('result_xgb.csv')