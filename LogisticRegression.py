#Logistic Regression Model. 10-fold Cross Validation


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_data():
    sample_data = pd.read_csv('dataset/Consolidated_CancerSEEK_Data.csv')
    sample_data = sample_data.drop(["Patient ID #", "Sample ID #", "Tumor type", "Race", "AJCC Stage"], axis=1)
    demographic_biomarker_data = sample_data.drop(["CancerSEEK Logistic Regression Score", "CancerSEEK Test Result"], axis=1)
    biomarker_data = demographic_biomarker_data.drop(["Age", "Sex"], axis=1)
    replicated_biomarker_data = biomarker_data[["Ω score", "CA-125 (U/ml)", "CA19-9 (U/ml)", "CEA (pg/ml)", "HGF (pg/ml)",
                                                "Myeloperoxidase (ng/ml)", "OPN (pg/ml)", "Prolactin (pg/ml)", "TIMP-1 (pg/ml)",
                                                "Ground Truth"]]
    return demographic_biomarker_data, biomarker_data, replicated_biomarker_data

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


def logistic_regression(dataset):
    np_dataset = dataset.to_numpy()
    num_col = np.shape(np_dataset)[1]
    num_rows = np.shape(np_dataset)[0]
    X = np_dataset[:num_rows - 100, :num_col - 1]
    y = np_dataset[:num_rows - 100, num_col - 1]
    X_test = np_dataset[:-100, :num_col-1]
    Y_test = np_dataset[:-100, num_col - 1]
    num_folds = KFold(n_splits=10, shuffle=True, random_state=553)
    lr_model = LogisticRegression(penalty='l2')
    score = cross_val_score(lr_model, X, y, cv=num_folds).mean()
    lr_model.fit(X, y)
    Y_pred = lr_model.predict(X_test)
    accuracy(Y_test, Y_pred)
    return score

demographic_biomarker_data, biomarker_data, replicated_biomarker_data = read_data()
biomarker_score = logistic_regression(biomarker_data)
replicated_score = logistic_regression(replicated_biomarker_data)
demographic_biomarker_data_score = logistic_regression(demographic_biomarker_data)

print(biomarker_score, replicated_score, demographic_biomarker_data_score)
