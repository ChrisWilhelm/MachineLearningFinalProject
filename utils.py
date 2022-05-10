import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# splits the dataset into
# test set : last 100 rows of the dataset
# train set: (1-dev_percentage) percent of the remaining data
# dev set  : dev_percentage percent of the remaining data
def train_test_dev_split(dataset, dev_percentage=0.2):
    assert 0.0 <= dev_percentage <= 1.0
    dataset = dataset.to_numpy()
    X_train, X_dev, y_train, y_dev = train_test_split(dataset[:-200, :-1], dataset[:-200, -1], test_size=dev_percentage)
    X_test = dataset[-200:, :-1]
    y_test = dataset[-200:, -1]

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def read_data():
    sample_data = pd.read_csv('dataset/Consolidated_CancerSEEK_Data.csv')
    cols = ['Patient ID #', 'Age', 'Sex', 'Race', 'Tumor type', 'AJCC Stage', 'Ω score', 'AFP (pg/ml)', \
            'Angiopoietin-2 (pg/ml)', 'AXL (pg/ml)', 'CA-125 (U/ml)', 'CA 15-3 (U/ml)', 'CA19-9 (U/ml)', 'CD44 (ng/ml)', \
            'CEA (pg/ml)', 'CYFRA 21-1 (pg/ml)', 'DKK1 (ng/ml)', 'Endoglin (pg/ml)', 'FGF2 (pg/ml)', 'Follistatin (pg/ml)', \
            'Galectin-3 (ng/ml)', 'G-CSF (pg/ml)', 'GDF15 (ng/ml)', 'HE4 (pg/ml)', 'HGF (pg/ml)', 'IL-6 (pg/ml)', \
            'IL-8 (pg/ml)', 'Kallikrein-6 (pg/ml)', 'Leptin (pg/ml)', 'Mesothelin (ng/ml)', 'Midkine (pg/ml)', \
            'Myeloperoxidase (ng/ml)', 'NSE (ng/ml)', 'OPG (ng/ml)', 'OPN (pg/ml)', 'PAR (pg/ml)', 'Prolactin (pg/ml)', \
            'sEGFR (pg/ml)', 'sFas (pg/ml)', 'SHBG (nM)', 'sHER2/sEGFR2/sErbB2 (pg/ml)', 'sPECAM-1 (pg/ml)', 'TGFa (pg/ml)', \
            'Thrombospondin-2 (pg/ml)', 'TIMP-1 (pg/ml)', 'TIMP-2 (pg/ml)', 'CancerSEEK Logistic Regression Score', \
            'CancerSEEK Test Result', 'Sample ID #', 'Ground Truth']
    sample_data = sample_data[cols]
    # print(sample_data.columns.tolist())
    sample_data = sample_data.drop(["Patient ID #", "Sample ID #", "Tumor type", "Race", "AJCC Stage"], axis=1)
    demographic_biomarker_data = sample_data.drop(["CancerSEEK Logistic Regression Score", "CancerSEEK Test Result"],axis=1)
    biomarker_data = demographic_biomarker_data.drop(["Age", "Sex"], axis=1)
    replicated_biomarker_data = biomarker_data[
        ["Ω score", "CA-125 (U/ml)", "CA19-9 (U/ml)", "CEA (pg/ml)", "HGF (pg/ml)",
         "Myeloperoxidase (ng/ml)", "OPN (pg/ml)", "Prolactin (pg/ml)", "TIMP-1 (pg/ml)", "Ground Truth"]]

    return train_test_dev_split(demographic_biomarker_data), \
           train_test_dev_split(biomarker_data), \
           train_test_dev_split(replicated_biomarker_data)


def confusion_matrix(true, preds):
    cf = np.zeros((2, 2))
    for i in range(len(true)):
        tval = true[i]
        pval = preds[i]
        if tval == 0: # true negative
            if tval == pval:
                cf[0, 0] += 1
            else:
                cf[0, 1] += 1
        else: # true positive
            if tval == pval:
                cf[1, 1] += 1
            else:
                cf[1, 0] += 1
    return cf


def plot_confusion_matrix(true, preds, model_type, classification, dataset_type):
    plt.clf()
    cf_matrix = confusion_matrix(true, preds)
    # plotting confusion matrix
    group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    ax = plt.axes()
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', ax=ax)
    axis_title = classification + ', Model ' + model_type + ', Data ' + dataset_type
    ax.set_title(axis_title)
    fname = './graphs/' + model_type + '_cfmatrix_' + classification + '_dataset_' + dataset_type
    plt.savefig(fname)


def analyze_cancer_type(ypred, ytest, sample_ids, model_type, dataset_type):
    assert len(ypred) == len(ytest) == len(sample_ids)
    dataset = pd.read_csv('dataset/Consolidated_CancerSEEK_Data.csv')
    # for each cancer type, there are two arrays for negative and positive classifications for y_hat
    lung = []
    colorectum = []
    esophagus = []
    breast = []
    pancreas = []
    liver = []
    ovary = []
    stomach = []
    normal = []
    for i in range(len(ypred)):
        y_hat = ypred[i]
        y_test = ytest[i]
        sample_id = sample_ids[i]
        ct = dataset.loc[dataset['Sample ID #'] == sample_id, 'Tumor type']
        cancer_type = ct.iloc[0]
        if cancer_type == 'Colorectum':
            #true positive classification
            if y_hat == y_test:
                colorectum.append(1)
            #false negative classification
            else:
                colorectum.append(0)
        elif cancer_type == 'Normal':
            # true negative classification
            if y_hat == y_test:
                normal.append(0)
            # false positive classification
            else:
                normal.append(1)
        elif cancer_type == 'Lung':
            # true positive classification
            if y_hat == y_test:
                lung.append(1)
            # false negative classification
            else:
                lung.append(0)
        elif cancer_type == 'Breast':
            # true positive classification
            if y_hat == y_test:
                breast.append(1)
            # false negative classification
            else:
                breast.append(0)
        elif cancer_type == 'Stomach':
            # true positive classification
            if y_hat == y_test:
                stomach.append(1)
            # false negative classification
            else:
                stomach.append(0)
        elif cancer_type == 'Ovary':
            # true positive classification
            if y_hat == y_test:
                ovary.append(1)
            # false negative classification
            else:
                ovary.append(0)
        elif cancer_type == 'Liver':
            # true positive classification
            if y_hat == y_test:
                liver.append(1)
            # false negative classification
            else:
                liver.append(0)
        elif cancer_type == 'Esophagus':
            # true positive classification
            if y_hat == y_test:
                esophagus.append(1)
            # false negative classification
            else:
                esophagus.append(0)
        elif cancer_type == 'Pancreas':
            # true positive classification
            if y_hat == y_test:
                pancreas.append(1)
            # false negative classification
            else:
                pancreas.append(0)
        else:
            print("Erroneous Sample Number")
    true_normal = np.zeros(len(normal))
    plot_confusion_matrix(true_normal, normal, model_type, 'Normal', dataset_type)
    true_colorectum = np.ones(len(colorectum))
    plot_confusion_matrix(true_colorectum, colorectum, model_type, 'Colorectum', dataset_type)
    true_stomach = np.ones(len(stomach))
    plot_confusion_matrix(true_stomach, stomach, model_type, 'Stomach', dataset_type)
    true_liver = np.ones(len(liver))
    plot_confusion_matrix(true_liver, liver, model_type, 'Liver', dataset_type)
    true_esophagus = np.ones(len(esophagus))
    plot_confusion_matrix(true_esophagus, esophagus, model_type, 'Esophagus', dataset_type)
    true_ovary = np.ones(len(ovary))
    plot_confusion_matrix(true_ovary, ovary, model_type, 'Ovary', dataset_type)
    true_pancreas = np.ones(len(pancreas))
    plot_confusion_matrix(true_pancreas, pancreas, model_type, 'Pancreas', dataset_type)
    true_breast = np.ones(len(breast))
    plot_confusion_matrix(true_breast, breast, model_type, 'Breast', dataset_type)
    true_lung = np.ones(len(lung))
    plot_confusion_matrix(true_lung, lung, model_type, 'Lung', dataset_type)


def train_test_dev_split_cancer(dataset, dev_percentage=0.2):
    assert 0.0 <= dev_percentage <= 1.0
    dataset = dataset.to_numpy()
    col_length = dataset.shape[1]
    X_train, X_dev, y_train, y_dev = train_test_split(dataset[:-200, 0:col_length - 2], dataset[:-200, -1], \
                                     test_size=dev_percentage)
    X_test = dataset[-200:, :col_length - 2]
    y_test = dataset[-200:, -2:]

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def read_data_cancertype():
    sample_data = pd.read_csv('dataset/Consolidated_CancerSEEK_Data.csv')
    cols = ['Patient ID #', 'Age', 'Sex', 'Race', 'Tumor type', 'AJCC Stage', 'Ω score', 'AFP (pg/ml)', \
            'Angiopoietin-2 (pg/ml)', 'AXL (pg/ml)', 'CA-125 (U/ml)', 'CA 15-3 (U/ml)', 'CA19-9 (U/ml)', 'CD44 (ng/ml)', \
            'CEA (pg/ml)', 'CYFRA 21-1 (pg/ml)', 'DKK1 (ng/ml)', 'Endoglin (pg/ml)', 'FGF2 (pg/ml)', 'Follistatin (pg/ml)', \
            'Galectin-3 (ng/ml)', 'G-CSF (pg/ml)', 'GDF15 (ng/ml)', 'HE4 (pg/ml)', 'HGF (pg/ml)', 'IL-6 (pg/ml)', \
            'IL-8 (pg/ml)', 'Kallikrein-6 (pg/ml)', 'Leptin (pg/ml)', 'Mesothelin (ng/ml)', 'Midkine (pg/ml)', \
            'Myeloperoxidase (ng/ml)', 'NSE (ng/ml)', 'OPG (ng/ml)', 'OPN (pg/ml)', 'PAR (pg/ml)', 'Prolactin (pg/ml)', \
            'sEGFR (pg/ml)', 'sFas (pg/ml)', 'SHBG (nM)', 'sHER2/sEGFR2/sErbB2 (pg/ml)', 'sPECAM-1 (pg/ml)', 'TGFa (pg/ml)', \
            'Thrombospondin-2 (pg/ml)', 'TIMP-1 (pg/ml)', 'TIMP-2 (pg/ml)', 'CancerSEEK Logistic Regression Score', \
            'CancerSEEK Test Result', 'Sample ID #', 'Ground Truth']
    sample_data = sample_data[cols]
    sample_data = sample_data.drop(["Patient ID #", "Tumor type", "Race", "AJCC Stage"], axis=1)
    demographic_biomarker_data = sample_data.drop(["CancerSEEK Logistic Regression Score", "CancerSEEK Test Result"],axis=1)
    biomarker_data = demographic_biomarker_data.drop(["Age", "Sex"], axis=1)
    replicated_biomarker_data = biomarker_data[
        ["Ω score", "CA-125 (U/ml)", "CA19-9 (U/ml)", "CEA (pg/ml)", "HGF (pg/ml)",
         "Myeloperoxidase (ng/ml)", "OPN (pg/ml)", "Prolactin (pg/ml)", "TIMP-1 (pg/ml)", "Sample ID #", "Ground Truth"]]

    return train_test_dev_split_cancer(demographic_biomarker_data), \
           train_test_dev_split_cancer(biomarker_data), \
           train_test_dev_split_cancer(replicated_biomarker_data)


def acc_spec_sens(y_hat, y):
    assert y_hat.shape == y.shape

    y_non_zero = np.count_nonzero(y)
    y_plus_y_hat = y + y_hat

    accuracy = 1.0 - (np.count_nonzero(y_plus_y_hat == 1.0) / y.shape[0])
    specificity = np.count_nonzero(y_plus_y_hat == 0.0) / (y.shape[0] - y_non_zero)
    sensitivity = np.count_nonzero(y_plus_y_hat == 2.0) / y_non_zero

    return accuracy, specificity, sensitivity


def visualize_data(df):
    sns.heatmap(df.corr(),
                xticklabels=df.corr().columns.values,
                yticklabels=df.corr().columns.values)
    plt.show()


# plots the roc curve and returns the roc auc score
def plot_roc(y_pred_probs, y_actual, model_type, dataset_type, fname):
    y_pred_fpr, y_pred_tpr, _ = roc_curve(y_actual, y_pred_probs)
    plt.plot(y_pred_fpr, y_pred_tpr, label=dataset_type)
    plt.title(model_type + ' ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(fname)

    return roc_auc_score(y_actual, y_pred_probs)


"""
demographic_biomarker_data, biomarker_data, replicated_biomarker_data = read_data_cancertype()
X_train, X_dev, X_test, y_train, y_dev, y_test = demographic_biomarker_data
print(y_dev.astype(np.float32))

demographic_biomarker_data, biomarker_data, replicated_biomarker_data = read_data()
X_train, X_dev, X_test, y_train, y_dev, y_test = demographic_biomarker_data
print(y_dev)
"""
