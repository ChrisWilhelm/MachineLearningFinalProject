import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def read_data():
    sample_data = pd.read_csv('dataset/Consolidated_CancerSEEK_Data.csv')
    print(sample_data)


def visualize_data(X, y):
    df = pd.DataFrame(data=X, columns=['Omega', 'CA-125', 'CA19-9', 'CEA', 'HGF', 'Myleoperoxidase',
                                       'OPN', 'Prolactin', 'TIMP 1'])
    sns.heatmap(df.corr(),
                xticklabels=df.corr().columns.values,
                yticklabels=df.corr().columns.values)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    read_data()