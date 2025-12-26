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
import csv
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

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
# Choose dataset and model from command line arguments
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

# -------------------------------
# Prepare dataset paths and DataLoaders
# -------------------------------
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

"""
Make predictions using the trained model and save results to CSV.
    
Args:
model: trained PyTorch model
device: device to run model on ('cpu' or 'cuda')
loader: DataLoader for dataset to predict
output_csv_path: path to save predictions as CSV
    
Returns:
total_labels: true labels 
predictions: predicted values 
"""
def predicting(model, device, loader, output_csv_path):
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
    
    predictions = total_preds.numpy().flatten()

    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Prediction'])
        csv_writer.writerows(map(lambda x: [x], predictions))

    return total_labels.numpy().flatten(), predictions

"""
    Run prediction on a dataset, save CSV, and compute evaluation metrics.
    
    Args:
        loader: DataLoader for dataset
        name: name used for output CSV and logging
"""
def pred(loader, name):
    output_csv_path = f'./{name}_pred.csv'
    true_labels, predicted_labels = predicting(model, device, loader, output_csv_path)
    true_data = pd.read_csv(f'./data/{name}.csv')
    true_affinities = true_data['affinity'].values
    predicted_data = pd.read_csv(f'./{name}_pred.csv')
    predicted_affinities = predicted_data['Prediction'].values
    mse = mean_squared_error(true_affinities, predicted_affinities)
    print(f'Mean Squared Error (MSE): {mse}')
    correlation_coefficient, _ = pearsonr(true_affinities, predicted_affinities)
    n = len(true_affinities)
    standard_error = 1.96 / np.sqrt(n - 3)
    print(name)
    print(f'Pearson Correlation Coefficient: {correlation_coefficient}')

device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model = modeling().to(device)
model_file_name = './model_' + model_st + '_' + dataset + '.model'
model.load_state_dict(torch.load(model_file_name, map_location=device))
pred(test_loader, f'{datasets[0]}_test')