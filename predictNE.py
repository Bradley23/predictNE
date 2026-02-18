"""Predicts global NE release patterns from widefield Ca2+ 
imaging data acquired from 12 allen atlas regions.
author - Brad Rauscher (2/2026)"""

#%% import packages
import numpy as np
import scipy.io as sio
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from tqdm import tqdm

#%% load Ca and NE timecourses as np stack and convert to tensors

print('Loading datasets...')
print('\tLoading Ca data...')
Ca = np.load('data/Ca_stack.npy')
print('\tLoading NE data...')
NE = np.load('data/NE_stack.npy')

print('\tSuccessfully loaded all data!!')

#%% intialize parameters

print('Initializing parameters...')

N = Ca.shape[0]
torch.manual_seed(23)

pTrain = 0.75
pVal = 0.15
pTest = 0.10

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

#%% create datasets

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class createDataset(Dataset):
    def __init__(self, X_list, y_list, seq_len):
        self.X_list = X_list
        self.y_list = y_list
        self.seq_len = seq_len
        
        # Precompute (recording_index, start_index)
        self.indices = []
        for rec_idx, X in enumerate(X_list):
            T = X.shape[0]
            for start in range(T - seq_len):
                self.indices.append((rec_idx, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rec_idx, start = self.indices[idx]
        
        X_seq = self.X_list[rec_idx][start:start+self.seq_len]
        y_seq = self.y_list[rec_idx][start:start+self.seq_len]
        
        return (
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32)
        )

seq_len = 300
batch_size = 64  # Increased batch size for better GPU utilization

dataset_train = createDataset(Ca_train, NE_train, seq_len)
dataset_val = createDataset(Ca_val, NE_val, seq_len)
dataset_test = createDataset(Ca_test, NE_test, seq_len)

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

#%% initialize network
class predictNE(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=1):
        super().__init__()
        
        # mabye RNN
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: [batch, seq_len, 12]
        
        out, _ = self.rnn(x)            # out: [batch, seq_len, hidden_size]
        out = self.fc(out)              # [batch, seq_len, 1]
        
        return out

#%% initialize model

print('Connecting to mac gpu...')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if torch.backends.mps.is_available():
    print('\tConnected to mac gpu!!')
else:
    print("\tCouldn't find mac gpu, connected to mac cpu!!")

model = predictNE().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#%% train model

num_epochs = 10

print('Starting model training...')

epoch_pbar = tqdm(range(num_epochs), desc="Training")
for epoch in epoch_pbar:
    
    model.train()
    total_loss = 0
    batch_count = 0
    
    batch_pbar = tqdm(loader_train, desc=f"Epoch {epoch+1}", leave=False)
    for X_batch, y_batch in batch_pbar:
        
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # Update batch progress bar with current loss
        batch_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / batch_count
    epoch_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

#%% save model

print('Training completed!')
print('Saving model...')

# Save the model state dict
torch.save(model.state_dict(), 'predictNE_model.pth')

# Also save the entire model (includes architecture)
torch.save(model, 'predictNE_model_full.pth')

# Save training info
training_info = {
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'seq_len': seq_len,
    'final_loss': avg_loss,
    'model_architecture': str(model),
    'device': str(device)
}

torch.save(training_info, 'training_info.pth')

print(f'Model saved successfully!')
print(f'  - Model weights: predictNE_model.pth')
print(f'  - Full model: predictNE_model_full.pth')
print(f'  - Training info: training_info.pth')

#%% to load

# model = predictNE()
model.load_state_dict(torch.load('predictNE_model.pth'))

# Or load full model
# model = torch.load('predictNE_model_full.pth')

# Load training info
info = torch.load('training_info.pth')

#%% predict NE from new timecourse

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
mse = np.mean((example_ne_true - predicted_ne) ** 2)

print(f"Prediction quality:")
print(f"  Correlation: {correlation:.4f}")
print(f"  MSE: {mse:.4f}")

