# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 09:17:05 2019
@author: Dv00
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import lightgbm as lgb

RANDOM_STATE = 1

train, test = pd.read_csv('../../data/train.csv'), pd.read_csv('../../data/test.csv')

missing=train.columns[train.isnull().any()].tolist()

def fill_missing_age(rows):
    Age = rows[0]
    Pclass = rows[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train['isMaster'], test['isMaster'] = [(df.Name.str.split().str[1] == 'Master.').astype('int') for df in [train, test]]

sex_train = pd.get_dummies(train['Sex'],drop_first=True)
embark_train = pd.get_dummies(train['Embarked'],drop_first=True)
train = train.drop(['Sex', 'Embarked', 'Name', 'Ticket'],axis=1)
train = pd.concat([train, sex_train, embark_train],axis=1)

train['Age'] = train[['Age','Pclass']].apply(fill_missing_age, axis=1)

train.drop('Cabin',axis=1,inplace=True)

train.dropna(inplace=True)

sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embark_test = pd.get_dummies(test['Embarked'],drop_first=True)
test = test.drop(['Sex', 'Embarked', 'Name','Ticket'],axis=1)
test = pd.concat([test, sex_test, embark_test],axis=1)

test['Age'] = test[['Age','Pclass']].apply(fill_missing_age, axis=1)

test.drop('Cabin',axis=1,inplace=True)

test['Fare'].fillna(test['Fare'].mean(), inplace=True)

X, y = train.drop('Survived',axis=1), train['Survived']
kf = KFold(n_splits=5, shuffle=True, random_state=4590)

predictions = []
pt = []
scores = 0

for trn_idx, val_idx in kf.split(X, y):
    dartmodel = lgb.LGBMClassifier(
        boosting_type='dart',
        num_leaves=30,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=0.001,
        min_child_samples=1,
        subsample=0.9,
        subsample_freq=20,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        reg_lambda=0.01,
        random_state=None,
        n_jobs=10,
        silent=True,
        importance_type='split')
    dartmodel.fit(X.iloc[trn_idx], y.iloc[trn_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], eval_names='val', eval_metric=None, verbose=False)
    score = accuracy_score(dartmodel.predict(X.iloc[val_idx]), y.iloc[val_idx])
    print(score)
    scores += score
    print(dartmodel.best_iteration_)
    predictions += [dartmodel.predict(test)]

print(scores / 5)

predictions = (np.mean(predictions, axis=0) > 0.5) * 1

predictions = pd.DataFrame(predictions, columns= ['Survived'])
predictions = pd.concat([test['PassengerId'], predictions], axis=1, join='inner')

predictions.to_csv('predictions.csv' , index=False)