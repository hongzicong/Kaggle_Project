# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:31:08 2019

@author: DELL-PC
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation, metrics
import time as tm

RANDOM_SEED = 106

time_begin = tm.time()

#data_train = pd.read_csv("C:/Users/Dv00/Desktop/dm2019springproj2/train.csv")
#data_test = pd.read_csv("C:/Users/Dv00/Desktop/dm2019springproj2/test.csv")

data_train_1 = pd.read_csv("../../dm2019springproj2/train1.csv")
data_train_2 = pd.read_csv("../../dm2019springproj2/train2.csv")
data_train_3 = pd.read_csv("../../dm2019springproj2/train3.csv")
data_train_4 = pd.read_csv("../../dm2019springproj2/train4.csv")
data_train_5 = pd.read_csv("../../dm2019springproj2/train5.csv")

data_test_1 = pd.read_csv("../../dm2019springproj2/test1.csv")
data_test_2 = pd.read_csv("../../dm2019springproj2/test2.csv")
data_test_3 = pd.read_csv("../../dm2019springproj2/test3.csv")
data_test_4 = pd.read_csv("../../dm2019springproj2/test4.csv")
data_test_5 = pd.read_csv("../../dm2019springproj2/test5.csv")

label_1 = pd.read_csv("../../dm2019springproj2/label1.csv")
label_2 = pd.read_csv("../../dm2019springproj2/label2.csv")
label_3 = pd.read_csv("../../dm2019springproj2/label3.csv")
label_4 = pd.read_csv("../../dm2019springproj2/label4.csv")
label_5 = pd.read_csv("../../dm2019springproj2/label5.csv")

x = pd.concat([data_train_1, data_train_2, data_train_3, data_train_4, data_train_5], ignore_index=True)
y = pd.concat([label_1, label_2, label_3, label_4, label_5], ignore_index=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = RANDOM_SEED)

# 使用随机森林

rfc = RandomForestRegressor(n_estimators=50, oob_score = True, 
                             n_jobs=-1, random_state = RANDOM_SEED)

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
# pd.Series(result).to_csv('result.csv')

print(tm.time() - time_begin)