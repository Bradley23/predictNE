"""Predicts global NE release patterns from widefield Ca2+ 
imaging data acquired from 12 allen atlas regions.
author - Brad Rauscher (February, 2026)"""

#%% ############### import packages ###############
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from models.lstm import predictNE
from train import train
from datasets.createDataset import createDataset
from datasets.loader import load
from torch.utils.data import Dataset, DataLoader
from metrics import predict_NE_from_timecourse, corr_rmse

#%% ############### load datasets ###############

print('Loading datasets...')
print('\tLoading Ca data...')
Ca = load('Ca')
print('\tLoading NE data...')
NE = load('NE')

print('\tSuccessfully loaded all data!!')

#%% ############### intialize parameters ###############

print('Initializing parameters...')

N = Ca.shape[0]

#%% determine indices for each training session

num_lo = 5

test_indices = []
train_indices = []

i = 0
while i < N:
    test_batch = []
    for b in range(num_lo):
        if i < N:
            test_batch.append(i)
        i += 1
    test_indices.append(test_batch)

S = len(test_indices)

for i in range(S):
    train_indices.append([x for x in range(N) if x not in test_indices[i]])

#%% ############### create train, val, test datasets ###############

l5o_correlation = []
l5o_rmse = []

for i in range(S):

    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    np.random.seed(23)

    print(f'Creating training, validation, and test datasets for session {i}...')

    Ca_train = Ca[train_indices[i]]
    Ca_test = Ca[test_indices[i]]

    NE_train = NE[train_indices[i]]
    NE_test = NE[test_indices[i]]

    model, optimizer, training_info = train(Ca_train, NE_train, Ca_test, NE_test)

    print("Making predictions on example timecourse...")

    device = torch.device("cuda")

    predicted_ne = predict_NE_from_timecourse(Ca_test, model, device)
    correlation, rmse = corr_rmse(NE_test, predicted_ne)
    
    print(f"Prediction quality:")
    print(f"  Correlation: {np.mean(correlation):.4f}")
    print(f"  RMSE: {np.mean(rmse):.4f}")

    l5o_correlation.append(np.mean(correlation))
    l5o_rmse.append(np.mean(rmse))

np.save('l5o_results.npy', {
    'correlation': np.array(l5o_correlation),
    'rmse': np.array(l5o_rmse)},
    allow_pickle=True)
    