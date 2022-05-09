import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import utils
import pandas as pd


class Net(nn.Module):
    def __init__(self, input_shape, hidden_layer):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden_layer)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_layer, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y


def predict(model, data):
    preds = []
    for n in range(np.shape(data)[0]):
        x = torch.from_numpy(data[n].astype(np.float32))
        output = model(x).detach().numpy()
        preds.append(output[0])
    return np.array(preds)


def train_epoch(model, opt, criterion, X, Y, epochs=4000, batch_size=50):
    model.train()
    losses = []
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    for i in range(epochs):
        for beg_i in range(0, X.shape[0], batch_size):
            x_batch = X[beg_i:beg_i + batch_size, :]
            y_batch = Y[beg_i:beg_i + batch_size, :]
            x_batch = Variable(x_batch)
            y_batch = Variable(y_batch)

            opt.zero_grad()
            # (1) Forward
            y_hat = model(x_batch)
            # (2) Compute diff
            loss = criterion(y_hat, y_batch)
            # (3) Compute gradients
            loss.backward()
            # (4) update weights
            opt.step()
            losses.append(loss.data.numpy())
		# if i%100 == 0:
        # print("Epoch: ", i, "- ",loss.data.numpy())
    return model


def run_network(dataset, learning_rate, epochs, hidden_layer, dataset_type):
    X_train, X_dev, X_test, y_train, y_dev, y_test = dataset
    sc = StandardScaler()

    # briefly combine the Xs so we can run fit_transform on them
    combined = np.concatenate((X_train, X_dev, X_test))
    combined = sc.fit_transform(combined)

    # re-separate the combined Xs
    X_train = combined[:X_train.shape[0]]
    X_dev = combined[X_train.shape[0]:X_train.shape[0] + X_dev.shape[0]]
    X_test = combined[X_train.shape[0] + X_dev.shape[0]:X_train.shape[0] + X_dev.shape[0] + X_test.shape[0]]

    y_train = y_train[:, None]

    net = Net(X_train.shape[1], hidden_layer)
    opt = optim.Adam(net.parameters(), learning_rate, betas=(0.9, 0.999))
    criterion = nn.BCELoss()
    model = train_epoch(net, opt, criterion, X_train, y_train, epochs)

    # predictions
    output = predict(model, X_dev)
    y_dev = y_dev.astype(np.float32)
    roc_auc_score = utils.plot_roc(output, y_dev, 'Neural Net', dataset_type, './graphs/nn_roc')
    print("Dataset: ", dataset_type)
    print("ROC, AUC score: ", roc_auc_score)
    y_hat_rint = np.rint(output)
    y_hat = []
    cutoff = .99999
    for i in range(len(output)):
        if output[i] >= cutoff:
            y_hat.append(1)
        else:
            y_hat.append(0)
    y_hat = np.array(y_hat)
    accuracy, specificity, sensitivity = utils.acc_spec_sens(y_hat, y_dev)
    print("Accuracy = ", accuracy, ", Specificity = ", specificity, ", Sensitivity = ", sensitivity)
    return model, X_test, y_test


def run_testdata(model, X, y, dataset_type):
    print("TEST DATA: Dataset ", dataset_type)
    # predictions
    output = predict(model, X)
    y_test = y[:, 1].astype(np.float32)
    sample_ids = y[:, 0]
    roc_auc_score = utils.plot_roc(output, y_test, 'Neural Net', dataset_type, './graphs/nn_roc')
    print("ROC, AUC score: ", roc_auc_score)
    y_hat = []
    cutoff = .99999
    for i in range(len(output)):
        if output[i] >= cutoff:
            y_hat.append(1)
        else:
            y_hat.append(0)
    y_hat = np.array(y_hat)
    accuracy, specificity, sensitivity = utils.acc_spec_sens(y_hat, y_test)
    print("Accuracy = ", accuracy, ", Specificity = ", specificity, ", Sensitivity = ", sensitivity)
    analyze_cancer_type(y_hat, y_test, sample_ids)


if __name__ == "__main__":
    EPOCHS = 400
    LR = .005
    NUM_HIDDEN_LAYERS = 100
    demographic_biomarker_data, biomarker_data, replicated_biomarker_data = utils.read_data_cancertype()
    model, X_test, y_test = run_network(demographic_biomarker_data, LR, EPOCHS, NUM_HIDDEN_LAYERS,
                                        'Demographic Biomarker')
    run_testdata(model, X_test, y_test, 'Demographic Biomarker')
    model, X_test, y_test = run_network(biomarker_data, LR, EPOCHS, NUM_HIDDEN_LAYERS, 'Biomarker')
    run_testdata(model, X_test, y_test, 'Biomarker')
    model, X_test, y_test = run_network(replicated_biomarker_data, LR, EPOCHS, NUM_HIDDEN_LAYERS, 'Replicated')
    run_testdata(model, X_test, y_test, 'Replicated')
"""


entire_dataset = pd.read_csv('dataset/Consolidated_CancerSEEK_Data.csv')
for i in range(entire_dataset.shape[0]):
    sample = entire_dataset.iloc[i, 1]
    ct = get_cancer_type(sample, entire_dataset)
    print(ct)



"""