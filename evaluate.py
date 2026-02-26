#%%
import torch
from models.lstm import predictNE
from metrics import predict_NE_from_timecourse, corr_rmse
import numpy as np
from matplotlib import pyplot as plt

#%% ############### load datasets ###############

print('Loading datasets...')
print('\tLoading Ca data...')
Ca = np.load('datasets/Ca_stack.npy')
print('\tLoading NE data...')
NE = np.load('datasets/NE_stack.npy')

print('\tSuccessfully loaded all data!!')

#%% ############### intialize parameters ###############

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

#%% ############### create train, val, test datasets ###############

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

#%% ############### load model state dictionary ###############

model = predictNE()
device = torch.device("cpu")

model_parameters = torch.load('checkpoints/2026-02-20_10-28-14.pt', map_location=device)

model.load_state_dict(model_parameters['model_state_dict'])

#%% ############### predict NE from new timecourse ###############

print("Making predictions on example timecourse...")

predicted_ne = predict_NE_from_timecourse(Ca_test, model, device)
correlation, rmse = corr_rmse(NE_test, predicted_ne)

print(f"Prediction quality:")
print(f"  Correlation: {np.mean(correlation):.4f}")
print(f"  RMSE: {np.mean(rmse):.4f}")

#%%
# Plot comparison
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(example_ne_true, label='True NE', alpha=0.7)
# plt.plot(predicted_ne, label='Predicted NE', alpha=0.7)
# plt.xlabel('Time')
# plt.ylabel('NE')
# plt.legend()
# plt.title('NE Prediction vs Ground Truth')

# plt.subplot(2, 1, 2)
# plt.plot(example_ca[:, :3])  # Plot first 3 Ca channels as example
# plt.xlabel('Time') 
# plt.ylabel('Ca Signal')
# plt.title('Input Ca Signals (first 3 channels)')

# plt.tight_layout()
# plt.show()