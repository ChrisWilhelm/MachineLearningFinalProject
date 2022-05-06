#Logistic Regression Model. 10-fold Cross Validation

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

import utils

def logistic_regression(dataset):
	X_train, X_dev, _, y_train, y_dev, _ = dataset

	num_folds = KFold(n_splits=10, shuffle=True)
	lr_model = LogisticRegression(penalty='l2')
	cval_score = cross_val_score(lr_model, X_train, y_train, cv=num_folds).mean()
	lr_model.fit(X_train, y_train)
	y_pred = lr_model.predict(X_dev)

	accuracy, specificity, sensitivity = utils.acc_spec_sens(y_pred, y_dev)
	return cval_score, accuracy, specificity, sensitivity

if __name__ == "__main__":
	demographic_biomarker_data, biomarker_data, replicated_biomarker_data = utils.read_data()
	biomarker_score = logistic_regression(biomarker_data)
	replicated_score = logistic_regression(replicated_biomarker_data)
	demographic_biomarker_data_score = logistic_regression(demographic_biomarker_data)

	print(biomarker_score)
	print(replicated_score)
	print(demographic_biomarker_data_score)
