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
from models.GNN import GNN_NE, fully_connected_edge_index
from train import train
from datasets.createDataset import createDataset
from datasets.loader import load, sort_trials
from torch.utils.data import Dataset, DataLoader
from metrics import predict_NE_from_timecourse, corr_rmse
from datasets.processing import calc_connectivity, window_clip, bpf
from diagnostics import Report

#%% ############### load datasets ###############

print('Loading datasets...')
print('\tLoading Ca data...')
Ca = load('Ca')
print('\tLoading HbT data...')
HbT = load('HbT')
print('\tLoading NE data...')
NE = load('NE')

print('\tSuccessfully loaded all data!!')

#%% signal processing

Ca = np.concatenate((Ca, HbT), axis=2)
NE = bpf(NE, freq=[0, 0.5], fs=10)

#%% ############### intialize parameters ###############

print('Initializing parameters...')

N = Ca.shape[0]

#%% determine indices for each training session

test_indices, train_indices = sort_trials('lomo')
S = len(test_indices)

#%% ############### create train, val, test datasets ###############

l5o_correlation = []
l5o_rmse = []

save_title = 'checkpoints/lomo_LSTM_HbT_Ca_lpNE'

report = Report(save_title + '.pdf')
report.add_title_page(title='lomo_LSTM_Ca', info=
                      {'model': 'LSTM',
                       'predictor': 'Ca',
                       'predicted': 'low-pass NE',
                       'epochs': 10})

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

    report.save_fig(report.plot_loss(training_info['train_loss_history'], training_info['val_loss_history']))

    print("Making predictions on example timecourse...")

    device = torch.device("cuda")

    predicted_ne = predict_NE_from_timecourse(Ca_test, model, device)
    correlation, rmse = corr_rmse(NE_test, predicted_ne)
    
    report.save_fig(report.plot_pred(NE_test, predicted_ne, r=correlation))

    print(f"Prediction quality:")
    print(f"  Correlation: {np.mean(correlation):.4f}")
    print(f"  RMSE: {np.mean(rmse):.4f}")

    l5o_correlation.append(np.mean(correlation))
    l5o_rmse.append(np.mean(rmse))

report.close()

np.save(save_title + '.npy', {
    'correlation': np.array(l5o_correlation),
    'rmse': np.array(l5o_rmse)},
    allow_pickle=True)
    