"""Predicts global NE release patterns from widefield Ca2+ 
imaging data acquired from 12 allen atlas regions.
author - Brad Rauscher (February, 2026)"""

#%% import packages
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import platform
import os
from datetime import datetime
from models.lstm import predictNE

#%% load datasets

print('Loading datasets...')
print('\tLoading Ca data...')
Ca = np.load('datasets/Ca_stack.npy')
print('\tLoading NE data...')
NE = np.load('datasets/NE_stack.npy')

print('\tSuccessfully loaded all data!!')

#%% intialize parameters

print('Initializing parameters...')

N = Ca.shape[0]
torch.manual_seed(23)
torch.cuda.manual_seed_all(23)
np.random.seed(23)

pTrain = 0.75
pVal = 0.15
pTest = 0.10

print(f"\tTraining proportion: {pTrain*100}%")
print(f"\tValidation proportion: {pVal*100}%")
print(f"\tTesting proportion: {pTest*100}%")

#%% create train, val, test datasets

print('Creating training, validation, and test datasets...')

indices = np.random.permutation(N)

idxTrain = indices[:int(pTrain * N)]
idxVal = indices[int(pTrain * N):int((pTrain + pVal) * N)]
idxTest = indices[int((pTrain + pVal) * N):]

Ca_train = Ca[idxTrain]
Ca_val = Ca[idxVal]
Ca_test = Ca[idxTest]

NE_train = NE[idxTrain]
NE_val = NE[idxVal]
NE_test = NE[idxTest]