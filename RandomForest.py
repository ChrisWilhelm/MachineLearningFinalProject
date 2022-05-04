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
    num_rows = np.shape(np_dataset)[0] - 100
    X = np_dataset[:num_rows, :num_col - 1]
    y = np_dataset[:num_rows, num_col - 1]
    num_folds = KFold(n_splits=10, shuffle=True, random_state=553)
    best_model = -1
    best_model_acc = 0
    results = list()
    for train_x, test_x in num_folds.split(X):
        print('Next Set:')
        X_train, X_test = X[train_x, :], X[test_x, :]
        Y_train, Y_test = y[train_x], y[test_x]
        numElements = X_train.shape[0]
        numLastIter = numElements % 50
        model = RandomForest(num_estimators=15, max_depth=5, min_samples_split=2, criterion='entropy')
        for i in range(1, (numElements - numLastIter)//100):
            model.fit(X_train[:i*100], Y_train[:i*100])
            print('Iteration: ' + str(i) + ' is ' + str(accuracy_score(Y_train[:i*100], model.predict(X_train[:i*100]))))
        model.fit(X_train, Y_train)
        print('Total Training Accuracy: ' + str(accuracy_score(Y_train, model.predict(X_train))))
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
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print('Test Accuracy for the set: ' + str(acc))
        if acc > best_model_acc:
            best_model = model
            best_model_acc = acc
        results.append(acc)
    return best_model, best_model_acc

def accuracy(y, y_hat):
    specificity = 0
    total_spec = 0
    sensitivity = 0
    total_sens = 0
    if y.shape == y_hat.shape:
        for i in range(y.shape[0]):
            if y[i] == 0:
                total_spec += 1
                if y_hat[i] == y[i]:
                    specificity += 1
            else:
                total_sens += 1
                if y_hat[i] == y[i]:
                    sensitivity += 1
    print("Specificity: ", specificity/total_spec, ", n = ", total_spec)
    print("Sensitivity: ", sensitivity/total_sens, ", n = ", total_sens)


def main():
    dem_bio_data, biomarker_data, replicated_biomarker_data = read_data()
    dem_bio = dem_bio_data.to_numpy()
    num_col = np.shape(dem_bio)[1]
    X_dem_bio = dem_bio[:, :num_col - 1]
    y_dem_bio = dem_bio[:, num_col - 1]
    X_dem_bio_test = X_dem_bio[:-100]
    y_dem_bio_test = y_dem_bio[:-100]
    bio = biomarker_data.to_numpy()
    num_col = np.shape(bio)[1]
    X_bio = bio[:, :num_col - 1]
    y_bio = bio[:, num_col - 1]
    X_bio_test = X_bio[:-100]
    y_bio_test = y_bio[:-100]
    rep = replicated_biomarker_data.to_numpy()
    num_col = np.shape(rep)[1]
    X_rep = rep[:, :num_col - 1]
    y_rep = rep[:, num_col - 1]
    X_rep_test = X_rep[:-100]
    y_rep_test = y_rep[:-100]
    dem_model, dem_acc = modeling(dem_bio_data)
    bio_model, bio_acc = modeling(biomarker_data)
    replicated_model, rep_acc = modeling(replicated_biomarker_data)
    Y_pred = dem_model.predict(X_dem_bio_test)
    print(dem_acc)
    print(accuracy_score(y_dem_bio_test, Y_pred))
    accuracy(y_dem_bio_test, Y_pred)
    Y_pred = bio_model.predict(X_bio_test)
    print(bio_acc)
    print(accuracy_score(y_bio_test, Y_pred))
    accuracy(y_bio_test, Y_pred)
    Y_pred = replicated_model.predict(X_rep_test)
    print(rep_acc)
    print(accuracy_score(y_rep_test, Y_pred))
    accuracy(y_rep_test, Y_pred)
    # print(dem_res)
    # print(bio_res)
    # print(replicated_res)
    """
    test accuracy for the 3 data types
    0.9638905066977286
    0.9621432731508445
    0.9196272568433314
    """


if __name__ == '__main__':
    main()
