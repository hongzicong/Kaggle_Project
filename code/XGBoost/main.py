# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:36:20 2019

@author: Dv00
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

RANDOM_STATE = 1

def change_age(x):
    if x < 16:
        return 1
    else:
        return 0


def change_family(x):
    if x == 1 or x == 2 or x == 3:
        return 1
    else:
        return 0


def change_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    elif x == 'Q':
        return 2


def pro_train_data(datas):
    datas['family'] = datas.SibSp + datas.Parch
    datas.loc[[65, 159, 176, 709], 'Age'] = 15
    age = datas.Age.map(change_age)
    datas['Age'] = age
    datas['family'] = datas.family.map(change_family)
    cabin_datas = datas.Cabin.replace(np.nan, 0)
    datas['Cabin'] = np.where(cabin_datas == 0, 0, 1)
    datas.Embarked.fillna('C', inplace=True)
    datas['Embarked'] = datas.Embarked.map(change_embarked)
    datas['Sex'] = np.where(datas['Sex'] == 'female', 1, 0)
    to_drop = ['PassengerId', 'Survived', 'Name', 'Ticket', 'SibSp',
               'Parch']
    target = datas.Survived
    datas_handle = datas.drop(to_drop, axis=1)
    return datas_handle, target


def pro_test_data(test_datas):
    test_datas.loc[(test_datas.Fare.isnull()), 'Fare'] = test_datas[
        (test_datas.Embarked == 'S') & (test_datas.Pclass == 3)].Fare.median()
    test_datas['family'] = test_datas.SibSp + test_datas.Parch
    test_datas.loc[[244, 344, 417], 'Age'] = 15
    age = test_datas.Age.map(change_age)
    test_datas['Age'] = age
    test_datas['family'] = test_datas.family.map(change_family)
    cabin_datas = test_datas.Cabin.replace(np.nan, 0)
    test_datas['Cabin'] = np.where(cabin_datas == 0, 0, 1)
    test_datas.Embarked.fillna('S', inplace=True)
    test_datas['Embarked'] = test_datas.Embarked.map(change_embarked)
    test_datas['Sex'] = np.where(test_datas['Sex'] == 'female', 1, 0)
    to_drop = ['PassengerId', 'Name', 'Ticket', 'SibSp',
               'Parch']
    new_test_data_handle = test_datas.drop(to_drop, axis=1)

    return new_test_data_handle

train_data = pd.read_csv('../../data/train.csv')
test_data = pd.read_csv('../../data/test.csv')
pro_datas, target = pro_train_data(train_data)
pre_datas = pro_test_data(test_data)

X_train, X_test, y_train, y_test = train_test_split(pro_datas, target, test_size=0.20, 
                                                    random_state=RANDOM_STATE)

xgcmodel = XGBClassifier(n_estimators=48, max_depth=4,
                    n_jobs=-1, random_state=RANDOM_STATE)
xgcmodel.fit(X_train, y_train)

train_predictions = xgcmodel.predict(X_test)
print(classification_report(y_test, train_predictions, digits=4))

predictions = pd.DataFrame(xgcmodel.predict(pre_datas), columns= ['Survived'])

predictions = pd.concat([test_data['PassengerId'], predictions], axis=1, join='inner')

predictions.to_csv('predictions.csv' , index=False)