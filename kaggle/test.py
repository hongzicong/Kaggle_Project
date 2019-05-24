# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:47:23 2019

@author: Dv00
"""


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

data_train = pd.read_csv("../../new1datamining2019spring/trainSet.csv")
data_test = pd.read_csv("../../new1datamining2019spring/test set.csv")


x = data_train.iloc[:, 0:-1]
y = data_train.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Set the parameters by cross-validation
parameter_space = {
    "n_estimators": [10, 15, 20],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [2, 4, 6],
}
 
#scores = ['precision', 'recall', 'roc_auc']
scores = ['roc_auc']
 
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
 
    clf = RandomForestClassifier(random_state=14)
    grid = GridSearchCV(clf, parameter_space, cv=5, scoring='%s' % score)
    #scoring='%s_macro' % score：precision_macro、recall_macro是用于multiclass/multilabel任务的
    grid.fit(x_train, y_train)
 
    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    bclf = grid.best_estimator_
    bclf.fit(x_train, y_train)
    y_true = y_test
    y_pred = bclf.predict(x_test)
    y_pred_pro = bclf.predict_proba(x_test)
    y_scores = pd.DataFrame(y_pred_pro, columns=bclf.classes_.tolist())[1].values
    print(classification_report(y_true, y_pred))
    auc_value = roc_auc_score(y_true, y_scores)