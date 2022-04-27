#Logistic Regression Model. 10-fold Cross Validation

import scipy
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


def logistic_regression(dataset):
    np_dataset = dataset.to_numpy()
    print(np_dataset)


demographic_biomarker_data, biomarker_data, replicated_biomarker_data = read_data()
logistic_regression(biomarker_data)