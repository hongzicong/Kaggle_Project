# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 09:20:08 2019
@author: Dv00
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import warnings

plt.style.use("ggplot")
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

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

"""
To fill the gap that the voting method has not been used, we equally average the best 
predictions of the models we have tried: logistic regression, GBDT, random forest, 
DART and XGBoost, where the prediction result is set to 1 if the average is greater 
than 0.5 otherwise 0. We test it in the Kaggle platform and the score is 0.79425.
It outperforms its constituent models except XGBoost.
"""

lr = pd.read_csv('../LR/predictions.csv') # 0.74641
rf = pd.read_csv('../RF/predictions.csv') # 0.76555
gbdt = pd.read_csv('../GBDT/predictions.csv') # 0.76076
dart = pd.read_csv('../DART/predictions.csv') # 0.78947
xgb = pd.read_csv('../XGBoost/predictions.csv') # 0.81818

predictions = pd.DataFrame(((lr.Survived + rf.Survived + gbdt.Survived + dart.Survived + xgb.Survived) / 5 > 0.5) * 1, columns=['Survived'])
predictions = pd.concat([test['PassengerId'], predictions], axis=1, join='inner')

predictions.to_csv('predictions_equal.csv' , index=False)

"""
In our view, the prominent strength of XGBoost may be encumbered by the other inferior 
models. We can assign them different voting weights based on the Kaggle scores of their 
respective best predictions. However, this adaptable approach gets a unchanged result, 
whose Kaggle score is still 0.79425.
"""

total_score = 0.74641 + 0.76555 + 0.76076 + 0.78947 + 0.81818

predictions = ((0.74641 / total_score * lr.Survived + \
            0.76555 / total_score * rf.Survived + \
            0.76076 / total_score * gbdt.Survived + \
            0.78947 / total_score * dart.Survived + \
            0.81818 / total_score * xgb.Survived) > 0.5) * 1

predictions = pd.DataFrame(predictions, columns= ['Survived'])
predictions = pd.concat([test['PassengerId'], predictions], axis=1, join='inner')

predictions.to_csv('predictions_weighted.csv' , index=False)

"""
We keep trying and create a classification model, whose input is the best predictions of 
the five models above while output is the ensemble prediction. Their predictions of the 
original training data are used as training data while the original training labels are 
still used as training labels here. We apply XGBoost and DART to run the model yet get 
unexpectedly worse results, the Kaggle scores of which are 0.75598 and 0.74162, 
respectively. Nonetheless, we plot their feature importances contrasting with the 
Kaggle scores of the respective best predictions of the constituent models, which make 
sense (the ranking is consistent).
"""


# logistic regression prediction of the original training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
pred_lr = logmodel.predict(train.drop('Survived', axis=1))

# random forest prediction of the original training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
rfmodel = RandomForestClassifier(criterion='gini', 
                                 n_estimators=100,
                                 min_samples_split=9,
                                 min_samples_leaf=5,
                                 oob_score=True,
                                 random_state=RANDOM_STATE,
                                 n_jobs=-1)
rfmodel.fit(X_train, y_train)
pred_rf = rfmodel.predict(train.drop('Survived', axis=1))

# GBDT prediction of the original training data
gbdtmodel = GradientBoostingClassifier(n_estimators=300,
                                       learning_rate = 0.04,
                                       max_depth=3,
                                       min_samples_leaf=1,
                                       random_state=RANDOM_STATE)
gbdtmodel.fit(X_train, y_train)
pred_gbdt = gbdtmodel.predict(train.drop('Survived', axis=1))

# DART prediction of the original training data
pred_dart = []
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
    dartmodel.fit(X.iloc[trn_idx], y.iloc[trn_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])])
    pred_dart += [dartmodel.predict(X)]
pred_dart = (np.mean(pred_dart, axis=0) > 0.5) * 1

# XGBoost prediction of the original training data
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

def clf(datas, label, pre_datas):
    x_train, x_test, y_train, y_test = train_test_split(datas, label, random_state=8)
    xgc = XGBClassifier()
    xgc_param = {
        'n_estimators': range(30, 50, 2),
        'max_depth': range(2, 7, 1)
    }
    gc = GridSearchCV(xgc, param_grid=xgc_param, cv=5)
    gc.fit(x_train, y_train)
    return gc

train_data, test_data = pd.read_csv('../../data/train.csv'), pd.read_csv('../../data/test.csv')
pro_datas, target = pro_train_data(train_data)
pre_datas = pro_test_data(test_data)
pre_y = clf(pro_datas, target, pre_datas)
pred_xgb = pre_y.predict(pro_datas)

# combine data
ensemble_train = pd.DataFrame(np.vstack([pred_lr, pred_rf, pred_gbdt, pred_dart, 
                                        pred_xgb]).T, columns=['lr', 'rf', 'gbdt', 'dart', 'xgb'])
ensemble_test = pd.DataFrame(np.vstack([lr.Survived, rf.Survived, gbdt.Survived, dart.Survived,
                                    xgb.Survived]).T, columns=['lr', 'rf', 'gbdt', 'dart', 'xgb'])

# apply XGBoost to run the model
x_train, x_test, y_train, y_test = train_test_split(ensemble_train, y, random_state=8)
xgc = XGBClassifier()
xgc_param = {
    'n_estimators': range(30, 50, 2),
    'max_depth': range(2, 7, 1)
}
gc = GridSearchCV(xgc, param_grid=xgc_param, cv=5)
gc.fit(x_train, y_train)

predictions = pd.DataFrame(gc.predict(ensemble_test), columns= ['Survived'])
predictions = pd.concat([test['PassengerId'], predictions], axis=1, join='inner')
predictions.to_csv('predictions_xgb.csv' , index=False)

print(gc.best_params_)
# {'n_estimators': 30, 'max_depth': 3}

# plot feature importances
import xgboost
xgc1 = XGBClassifier(n_estimators=30, max_depth=3)
xgc1.fit(x_train, y_train)
xgboost.plot_importance(xgc1)
# [11, 30, 30, 54, 78]

# apply DART to run the model
predictions = []
for trn_idx, val_idx in kf.split(ensemble_train, y):
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
    dartmodel.fit(ensemble_train.iloc[trn_idx], y.iloc[trn_idx], eval_set=[(ensemble_train.iloc[val_idx], y.iloc[val_idx])])
    predictions += [dartmodel.predict(blend_test)]

predictions = (np.mean(predictions, axis=0) > 0.5) * 1
predictions = pd.DataFrame(predictions, columns= ['Survived'])
predictions = pd.concat([test['PassengerId'], predictions], axis=1, join='inner')
predictions.to_csv('predictions_dart.csv' , index=False)

# plot feature importances of the last model
lgb.plot_importance(dartmodel)
# [33, 101, 196, 203, 285]

# plot feature importances contrasting with the Kaggle scores of the respective best 
# predictions of the constituent models
x = np.arange(5)
a = [74.641, 76.076, 76.555, 78.947, 81.818]
b = [11, 30, 30, 54, 78]
c = np.array([33, 101, 196, 203, 285]) / 3

total_height, n = 0.8, 3
height = total_height / n
x = x - (total_height - height) / 2

plt.barh(x + 2 * height, a,  height=height, label='Public Score')
plt.barh(x + height, b, height=height, label='XGBoost Importance')
plt.barh(x, c, height=height, label=r'DART Importance ($\times 3$)')
plt.yticks(range(5), ['lr', 'gbdt', 'rf', 'dart', 'xgb'])
plt.legend()