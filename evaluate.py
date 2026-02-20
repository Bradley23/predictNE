#%%
import torch
from models.lstm import predictNE
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

model_parameters = torch.load('checkpoints/2026-02-20_09-38-19.pt', map_location=device)

model.load_state_dict(model_parameters['model_state_dict'])

#%% ############### predict NE from new timecourse ###############

def predict_NE_from_timecourse(ca_timecourse, model, device, seq_len=300):
    """
    Predict NE from a calcium timecourse using the trained model.
    
    Args:
        ca_timecourse (np.ndarray): Input Ca data of shape (T, 12)
        model: Trained PyTorch model
        device: Device to run inference on
        seq_len (int): Sequence length used during training
        
    Returns:
        np.ndarray: Predicted NE timecourse of shape (T,)
    """
    model.eval()
    
    T, n_features = ca_timecourse.shape
    print(f"Input timecourse shape: {ca_timecourse.shape}")
    
    if n_features != 12:
        raise ValueError(f"Expected 12 features, got {n_features}")
    
    if T < seq_len:
        raise ValueError(f"Timecourse too short. Need at least {seq_len} timepoints, got {T}")
    
    # Initialize output array
    ne_predictions = np.zeros(T)
    
    # Convert to tensor
    ca_tensor = torch.tensor(ca_timecourse, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # For the first seq_len points, we can only predict starting from seq_len
        # So we'll fill the first seq_len-1 points with the first prediction
        
        # Get first prediction
        first_seq = ca_tensor[:seq_len].unsqueeze(0)  # Add batch dimension
        first_pred = model(first_seq).squeeze().cpu().numpy()  # Remove batch dim
        
        # Fill first seq_len points
        ne_predictions[:seq_len] = first_pred
        
        # Predict for remaining timepoints using sliding window
        for start_idx in range(1, T - seq_len + 1):
            seq = ca_tensor[start_idx:start_idx + seq_len].unsqueeze(0)
            pred = model(seq).squeeze().cpu().numpy()
            
            # Take the last prediction (most recent timestep)
            ne_predictions[start_idx + seq_len - 1] = pred[-1]
    
    print(f"Generated NE predictions with shape: {ne_predictions.shape}")
    return ne_predictions

# Example usage:
# Select a test timecourse (first test recording)
example_ca = Ca_test[0]  # Shape: (T, 12)
example_ne_true = NE_test[0]  # True NE values for comparison

print("Making predictions on example timecourse...")
predicted_ne = predict_NE_from_timecourse(example_ca, model, device)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(example_ne_true, label='True NE', alpha=0.7)
plt.plot(predicted_ne, label='Predicted NE', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('NE')
plt.legend()
plt.title('NE Prediction vs Ground Truth')

plt.subplot(2, 1, 2)
plt.plot(example_ca[:, :3])  # Plot first 3 Ca channels as example
plt.xlabel('Time') 
plt.ylabel('Ca Signal')
plt.title('Input Ca Signals (first 3 channels)')

plt.tight_layout()
plt.show()

#%% Calculate correlation
correlation = np.corrcoef(example_ne_true.flatten(), predicted_ne.flatten())[0, 1]
rmse = np.mean((example_ne_true - predicted_ne) ** 2) ** 0.5

print(f"Prediction quality:")
print(f"  Correlation: {correlation:.4f}")
print(f"  RMSE: {rmse:.4f}")