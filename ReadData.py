import pandas as pd

from utils import train_test_dev_split

def read_data():
    sample_data = pd.read_csv('dataset/Consolidated_CancerSEEK_Data.csv')
    colorectum_starter = sample_data[sample_data["Tumor type"].isin(["Colorectum", "Normal"])]
    colorectum_final = colorectum_starter.drop(["Patient ID #", "Sample ID #", "Tumor type", "Race",
                                                "AJCC Stage", "CancerSEEK Logistic Regression Score",
                                                "CancerSEEK Test Result"], axis=1)
    colorectum_final = colorectum_final.sample(frac=1, random_state=553)
    sample_data = sample_data.drop(["Patient ID #", "Sample ID #", "Tumor type", "Race", "AJCC Stage"], axis=1)
    demographic_biomarker_data = sample_data.drop(["CancerSEEK Logistic Regression Score", "CancerSEEK Test Result"], axis=1)
    biomarker_data = demographic_biomarker_data.drop(["Age", "Sex"], axis=1)
    replicated_biomarker_data = sample_data[
        ["Ω score", "CA-125 (U/ml)", "CA19-9 (U/ml)", "CEA (pg/ml)", "HGF (pg/ml)",
         "Myeloperoxidase (ng/ml)", "OPN (pg/ml)", "Prolactin (pg/ml)", "TIMP-1 (pg/ml)",
         "Ground Truth"]]
    return demographic_biomarker_data, biomarker_data, replicated_biomarker_data, colorectum_final

def read_data2():
	sample_data = pd.read_csv('dataset/Consolidated_CancerSEEK_Data.csv')
	colorectum_starter = sample_data[sample_data["Tumor type"].isin(["Colorectum", "Normal"])]
	colorectum_final = colorectum_starter.drop(["Patient ID #", "Sample ID #", "Tumor type", "Race",
												"AJCC Stage", "CancerSEEK Logistic Regression Score",
												"CancerSEEK Test Result"], axis=1)
	colorectum_final = colorectum_final.sample(frac=1, random_state=553)
	sample_data = sample_data.drop(["Patient ID #", "Sample ID #", "Tumor type", "Race", "AJCC Stage"], axis=1)
	demographic_biomarker_data = sample_data.drop(["CancerSEEK Logistic Regression Score", "CancerSEEK Test Result"], axis=1)
	biomarker_data = demographic_biomarker_data.drop(["Age", "Sex"], axis=1)
	replicated_biomarker_data = sample_data[
		["Ω score", "CA-125 (U/ml)", "CA19-9 (U/ml)", "CEA (pg/ml)", "HGF (pg/ml)",
		 "Myeloperoxidase (ng/ml)", "OPN (pg/ml)", "Prolactin (pg/ml)", "TIMP-1 (pg/ml)",
		 "Ground Truth"]]
	return train_test_dev_split(demographic_biomarker_data), \
		   train_test_dev_split(biomarker_data), \
		   train_test_dev_split(replicated_biomarker_data), \
		   train_test_dev_split(colorectum_final)
