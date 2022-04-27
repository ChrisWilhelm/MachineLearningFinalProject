"""
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np
import pandas as pd

from ReadData import read_data
from sklearn.ensemble import RandomForestClassifier

class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self):
        raise NotImplementedError()

    def fit(self, *, X, iterations):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class RandomForest(Model):

    def __init__(self, num_estimators=100, criterion="entropy", max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Args:
            nfeatures: size of feature space (only needed for _fix_test_feats)
            lambda0: A float giving the default value for lambda
        """
        # TODO: Initializations etc. go here.
        self.classifier = RandomForestClassifier(num_estimators, criterion=criterion,
                                                 max_depth=max_depth, min_samples_split=min_samples_split,
                                                 min_samples_leaf=min_samples_leaf)

    def fit(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)

    def predict(self, X):
        return self.classifier.predict(X)


def main():
    dem_bio_data, biomarker_data, replicated_biomarker_data = read_data()
    dem_bio_data_dev = dem_bio_data
    dem_bio_data_test = []
    model = RandomForest()
    # select random X% for training and Y% for pred where X + Y = 100
    model.fit(dem_bio_data[dem_bio_data.columns[:-1]], dem_bio_data[dem_bio_data.columns[dem_bio_data.columns.length - 1]])


if __name__ == '__main__':
    main()
