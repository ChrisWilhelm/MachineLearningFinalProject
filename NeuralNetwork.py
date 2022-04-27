import os
from _ast import arguments
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.accuracies import approx_train_acc_and_loss, dev_acc_and_loss

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


class BestModel(torch.nn.Module):
    def __init__(self, input_shape, hidden_layer):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def run_network(dataset, epochs=4000, learning_rate=200, batch_size=.001):
    # scaling data and splitting into train/dev/test
    dataset = dataset.to_numpy()
    sc = StandardScaler()
    X_prenorm = dataset[:, :-1]
    X = sc.fit_transform(X_prenorm)
    y = dataset[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)

    # model directory
    MODEL_SAVE_DIR = "/MachineLearningFinalProject"

    # creating NN model
    input_shape = np.shape(X)[1]
    hidden_layer = input_shape * 2
    model = BestModel(input_shape, hidden_layer)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training model
    losses = []
    accuracy = []
    for i in range(epochs):
        for j, (x_train, y_train) in enumerate(trainloader):
            # calculate output
            output = model(x_train)

            # calculate loss
            loss = loss_fn(output, y_train.reshape(-1, 1))

            # accuracy
            predicted = model(torch.tensor(x, dtype=torch.float32))
            acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 50 == 0:
            losses.append(loss)
            accur.append(acc)
            print("epoch {}\tloss : {}\t accuracy : {}".format(i, loss, acc))
    """
    for step in range(epochs):  # LR: 0.001, bs: 200, epochs: 4000
        i = np.random.choice(input_shape, size=batch_size, replace=False)
        x = torch.from_numpy(X_train[i].astype(np.float32))
        y = torch.from_numpy(y_train[i].astype(np.int))
        # Forward pass: Get logits for x
        logits = model(x)
        # Compute loss
        loss = F.cross_entropy(logits, y)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log model performance every 100 epochs
        if step % 100 == 0:
            train_acc, train_loss = approx_train_acc_and_loss(model, X_train, y_train)
            dev_acc, dev_loss = dev_acc_and_loss(model, X_dev, y_dev)
            step_metrics = {
                'step': step,
                'train_loss': loss.item(),
                'train_acc': train_acc,
                'dev_loss': dev_loss,
                'dev_acc': dev_acc
            }
            print(f"On step {step}:\tTrain loss {train_loss}\t|\tDev acc is {dev_acc}")
    """
    model_savepath = os.path.join(MODEL_SAVE_DIR, "bestmodel.pt")
    print("Training completed, saving model at {model_savepath}")
    torch.save(model, model_savepath)

    # testing model
    PREDICTIONS_FILE = "predictions.txt"
    model = model
    predictions = []
    for test_case in X_test:
        x = torch.from_numpy(test_case.astype(np.float32))
        x = x.view(1, -1)
        logits = model(x)
        pred = torch.max(logits, 1)[1]
        predictions.append(pred.item())
    print(f"Storing predictions in {PREDICTIONS_FILE}")
    predictions = np.array(predictions)
    np.savetxt(PREDICTIONS_FILE, predictions, fmt="%d")


demographic_biomarker_data, biomarker_data, replicated_biomarker_data = read_data()
run_network(demographic_biomarker_data)

