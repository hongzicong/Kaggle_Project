# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:30:11 2019

@author: Dv00
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
import time as tm
from sklearn.externals import joblib

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

# 使用GBDT

gbdt = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, min_samples_split=140, min_samples_leaf=110,
                                 max_depth=10, random_state = RANDOM_SEED)

gbdt.fit(x_train, y_train)

gbdt_y_predict = gbdt.predict(x_test)

#获取特征的重要性
#importances = rfc.feature_importances_
#indices = np.argsort(importances)[::-1]
#cols_name = data_train.columns[:-1]
#for f in range(x_train.shape[1]):
#    print("%2d) %-*s %f" % (f + 1,30,cols_name[indices[f]],importances[indices[f]]))

predictions = [round(value) for value in gbdt_y_predict]

score = r2_score(y_test, predictions)
print("R2 Score: %.3f%%" % (score * 100.0))

data_test = pd.concat([data_test_1, data_test_2, data_test_3, data_test_4, data_test_5, data_test_6], ignore_index=True)
result = gbdt.predict(data_test)
pd.DataFrame(result,columns=['Predicted'],index=list(range(1,len(result) + 1))).to_csv('../../dm2019springproj2/result_gdbt.csv')

#model_save_path = "./model_save/"
#save_path_name = model_save_path + "rf_" + "train_model.m"
#joblib.dump(rfc, save_path_name)

print(tm.time() - time_begin)