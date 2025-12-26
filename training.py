import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from acnn import *
from utils import *
import matplotlib.pyplot as plt
import random
import yaml

"""
    Train the model for one epoch.
    
    Args:
        model: the neural network model to train
        device: 'cuda' or 'cpu'
        train_loader: DataLoader for training data
        optimizer: optimizer object (e.g., Adam)
        epoch: current epoch number
        lambda_l2: L2 regularization coefficient
 """
def train(model, device, train_loader, optimizer, epoch, lambda_l2):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        
        # Calculate L2 regularization
        l2_reg = sum(torch.norm(w, p=2) for w in model.parameters())
        loss += lambda_l2 * l2_reg
        
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data.x),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))

"""
Make predictions on a dataset using the model.
    
Args:
model: trained model
device: 'cuda' or 'cpu'
loader: DataLoader for evaluation data
    
Returns:
total_labels: truth labels
total_preds: predicted values
"""
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

# -------------------------------
# Load configuration from YAML
# -------------------------------
with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

TRAIN_BATCH_SIZE = config["train_batch_size"]
TEST_BATCH_SIZE = config["test_batch_size"]
LR = config["learning_rate"]
LOG_INTERVAL = config["log_interval"]
NUM_EPOCHS = config["num_epochs"]
LAMBDA_L2 = config["lambda_l2"]

# -------------------------------
# Select dataset and model based on command line arguments
# -------------------------------
datasets = [['davis', 'kiba'][int(sys.argv[1])]]
modeling = [cnn][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    processed_data_file_val = 'data/processed/' + dataset + '_val.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset + '_train')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test')
        val_data = TestbedDataset(root='data', dataset=dataset + '_val')
            
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = './model_' + model_st + '_' + dataset + '.model'
        result_file_name = './result_' + model_st + '_' + dataset + '.csv'
            
        val_mse_values = []
        epoch_values = []
        # -------------------------------
        # Training loop
        # -------------------------------    
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1, LAMBDA_L2)
            G, P = predicting(model, device, val_loader)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
    
            val_mse = ret[1]
            val_mse_values.append(val_mse)
            epoch_values.append(epoch + 1)

            # Save the best model if validation MSE improves
            if ret[1] < best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret)))
                best_epoch = epoch + 1
                best_mse = ret[1]
                best_ci = ret[-1]
                print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)
            else:
                print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)
