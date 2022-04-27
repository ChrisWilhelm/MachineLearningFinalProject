"""
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np
import pandas as pd

from ReadData import read_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class RandomForest():

    def __init__(self, num_estimators=100, criterion="entropy", max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.classifier = RandomForestClassifier(n_estimators=num_estimators, criterion=criterion,
                                                 max_depth=max_depth, min_samples_split=min_samples_split,
                                                 min_samples_leaf=min_samples_leaf, random_state=553)

    def fit(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)

    def predict(self, X):
        return self.classifier.predict(X)


def modeling(data):
    np_dataset = data.to_numpy()
    num_col = np.shape(np_dataset)[1]
    X = np_dataset[:, :num_col - 1]
    y = np_dataset[:, num_col - 1]
    num_folds = KFold(n_splits=10, shuffle=True, random_state=553)
    results = list()
    for train_x, test_x in num_folds.split(X):
        X_train, X_test = X[train_x, :], X[test_x, :]
        Y_train, Y_test = y[train_x], y[test_x]
        # best params found n_estimators = 20, max_depth = 15 min_samples_split = 3, criterion='entropy'
        # variables = dict()
        # variables['n_estimators'] = [5, 10, 15, 20]
        # variables['max_depth'] = [5, 10, 15]
        # variables['min_samples_split'] = [2, 3, 4]
        # variables['criterion'] = ['gini', 'entropy']
        # model = RandomForestClassifier(random_state=553)
        # best_hyper = GridSearchCV(model, variables, scoring='accuracy')
        # result = best_hyper.fit(X_train, Y_train)
        # best_model = result.best_estimator_
        # Y_pred = best_model.predict(X_test)
        # acc = accuracy_score(Y_test, Y_pred)
        # results.append(acc)
        # print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, best_hyper.best_score_, best_hyper.best_params_))
        model = RandomForest(num_estimators=20, max_depth=15, min_samples_split=3, criterion='entropy')
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        results.append(acc)
    return results


def main():
    dem_bio_data, biomarker_data, replicated_biomarker_data = read_data()
    print(np.mean(modeling(dem_bio_data)))
    print(np.mean(modeling(biomarker_data)))
    print(np.mean(modeling(replicated_biomarker_data)))
    """
    test accuracy for the 3 data types
    0.9653208669783255
    0.9543075708821565
    0.9042104304535243
    """


if __name__ == '__main__':
    main()
