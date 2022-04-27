import inline as inline
import matplotlib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import numpy as np

def read_data():
    sample_data = pd.read_csv('dataset/Consolidated_CancerSEEK_Data.csv')
    sample_data = sample_data.drop(["Patient ID #", "Sample ID #", "Tumor type", "Race", "AJCC Stage"], axis=1)
    demographic_biomarker_data = sample_data.drop(["CancerSEEK Logistic Regression Score", "CancerSEEK Test Result"],
                                                  axis=1)
    biomarker_data = demographic_biomarker_data.drop(["Age", "Sex"], axis=1)
    replicated_biomarker_data = biomarker_data[
        ["Ω score", "CA-125 (U/ml)", "CA19-9 (U/ml)", "CEA (pg/ml)", "HGF (pg/ml)",
         "Myeloperoxidase (ng/ml)", "OPN (pg/ml)", "Prolactin (pg/ml)", "TIMP-1 (pg/ml)",
         "Ground Truth"]]
    return demographic_biomarker_data, biomarker_data, replicated_biomarker_data

class Net(nn.Module):
    def __init__(self, input_shape, hidden_layer):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden_layer)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.1)
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


def train_epoch(model, opt, criterion, X, Y, batch_size=50):
    model.train()
    losses = []
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
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
        print(loss.data.numpy())
    return losses


def run_network(dataset, hidden_layer):
    dataset = dataset.to_numpy()
    sc = StandardScaler()
    X_prenorm = dataset[:, :-1]
    X = sc.fit_transform(X_prenorm)
    y = dataset[:, -1]
    y = y[:, None]

    net = Net(X.shape[1], hidden_layer)
    opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.BCELoss()
    e_losses = []
    num_epochs = 20
    for e in range(num_epochs):
        e_losses += train_epoch(net, opt, criterion, X, y)


layer = 100
demographic_biomarker_data, biomarker_data, replicated_biomarker_data = read_data()
run_network(demographic_biomarker_data, layer)