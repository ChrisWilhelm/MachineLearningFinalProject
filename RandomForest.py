"""
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np
import pandas as pd

from ReadData import read_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class RandomForest():

    def __init__(self, num_estimators=100, criterion="entropy", max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.classifier = RandomForestClassifier(num_estimators, criterion=criterion,
                                                 max_depth=max_depth, min_samples_split=min_samples_split,
                                                 min_samples_leaf=min_samples_leaf)

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
        X_train, X_test = X[train_x,:], X[test_x,:]
        Y_train, Y_test = y[train_x], y[test_x]
        model = RandomForest()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        results.append(accuracy_score(Y_test, Y_pred))
    return results


def main():
    dem_bio_data, biomarker_data, replicated_biomarker_data = read_data()
    print(np.mean(modeling(dem_bio_data)))
    print(np.mean(modeling(biomarker_data)))
    print(np.mean(modeling(replicated_biomarker_data)))


if __name__ == '__main__':
    main()
