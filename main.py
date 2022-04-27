import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def read_data():
    sample_data = pd.read_csv('dataset/Consolidated_CancerSEEK_Data.csv')
    sample_data.fillna(0)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    demographic_biomarker_data = sample_data.drop(["CancerSEEK Logistic Regression Score", "CancerSEEK Test Result"], axis=1)
    biomarker_data = demographic_biomarker_data.drop(["Age", "Sex"], axis=1)
    replicated_biomarker_data = biomarker_data[["Ω score", "CA-125 (U/ml)", "CA19-9 (U/ml)", "CEA (pg/ml)", "HGF (pg/ml)",
                                                "Myeloperoxidase (ng/ml)", "OPN (pg/ml)", "Prolactin (pg/ml)", "TIMP-1 (pg/ml)",
                                                "Ground Truth"]]
    return demographic_biomarker_data, biomarker_data, replicated_biomarker_data


def visualize_data(df):
    sns.heatmap(df.corr(),
                xticklabels=df.corr().columns.values,
                yticklabels=df.corr().columns.values)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    demographic_biomarker_data, biomarker_data, replicated_biomarker_data = read_data()
    print(demographic_biomarker_data.describe())
    print(biomarker_data.describe())
