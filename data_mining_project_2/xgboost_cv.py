# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:09:07 2019

@author: Dv00
"""

import pandas as pd
import time as tm

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score

time_begin = tm.time()

RANDOM_SEED = 106

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

params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': 6,
    'learning_rate': 0.2,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree':0.8,
    'seed': RANDOM_SEED,
    'n_jobs': -1,
}

xgb = XGBRegressor()

xgb_grid = GridSearchCV(xgb, params, cv = 2, n_jobs = -1, verbose=True)

xgb_grid.fit(x_train, y_train)

print(xgb_grid.best_score_)

print(xgb_grid.best_params_)

print(r2_score(y_test, xgb_grid.best_estimator_.predict(x_test)))

print("Time used: %f" % (tm.time() - time_begin))