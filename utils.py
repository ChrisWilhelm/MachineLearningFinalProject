import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# splits the dataset into
# test set : last 100 rows of the dataset
# train set: (1-dev_percentage) percent of the remaining data
# dev set  : dev_percentage percent of the remaining data
def train_test_dev_split(dataset, dev_percentage=0.2):
	assert 0.0 <= dev_percentage <= 1.0
	dataset = dataset.to_numpy()
	X_train, X_dev, y_train, y_dev = train_test_split(dataset[:-100, :-1], dataset[:-100, -1], test_size=dev_percentage)
	X_test = dataset[-100:, :-1]
	y_test = dataset[-100:, -1]

	return X_train, X_dev, X_test, y_train, y_dev, y_test

def read_data():
	sample_data = pd.read_csv('dataset/Consolidated_CancerSEEK_Data.csv')
	sample_data = sample_data.drop(["Patient ID #", "Sample ID #", "Tumor type", "Race", "AJCC Stage"], axis=1)

	demographic_biomarker_data = sample_data.drop(["CancerSEEK Logistic Regression Score", "CancerSEEK Test Result"], axis=1)
	biomarker_data = demographic_biomarker_data.drop(["Age", "Sex"], axis=1)
	replicated_biomarker_data = biomarker_data[["Ω score", "CA-125 (U/ml)", "CA19-9 (U/ml)", "CEA (pg/ml)", "HGF (pg/ml)",
												"Myeloperoxidase (ng/ml)", "OPN (pg/ml)", "Prolactin (pg/ml)", "TIMP-1 (pg/ml)",
												"Ground Truth"]]


	return train_test_dev_split(demographic_biomarker_data), \
		   train_test_dev_split(biomarker_data), \
		   train_test_dev_split(replicated_biomarker_data)

def acc_spec_sens(y_hat, y):
	assert y_hat.shape == y.shape

	y_non_zero = np.count_nonzero(y)
	y_plus_y_hat = y + y_hat

	accuracy = 1.0 - (np.count_nonzero(y_plus_y_hat == 1.0) /  y.shape[0])
	specificity =     np.count_nonzero(y_plus_y_hat == 0.0) / (y.shape[0] - y_non_zero)
	sensitivity =     np.count_nonzero(y_plus_y_hat == 2.0) / y_non_zero

	return accuracy, specificity, sensitivity


def visualize_data(df):
    sns.heatmap(df.corr(),
                xticklabels=df.corr().columns.values,
                yticklabels=df.corr().columns.values)
    plt.show()
