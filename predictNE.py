"""Predicts global NE release patterns from widefield Ca2+ 
imaging data acquired from 12 allen atlas regions.
author - Brad Rauscher (2/2026)"""

#%% import packages
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import platform
import os
from datetime import datetime

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
torch.cuda.manual_seed_all(23)
np.random.seed(23)

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
batch_size = 64  # Restore larger batch size - 12GB should handle this easily

print('Creating datasets and data loaders...')
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

print('Detecting available compute device...')

# Detect operating system and set device accordingly
system = platform.system()

if system == "Darwin":  # macOS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.backends.mps.is_available():
        print('\tConnected to Mac GPU (MPS)!!')
    else:
        print("\tCouldn't find Mac GPU, using CPU!!")
elif system == "Linux":  # Linux
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f'\tConnected to NVIDIA GPU: {torch.cuda.get_device_name()}!!')
    else:
        print("\tCouldn't find CUDA GPU, using CPU!!")
else:  # Windows or other systems
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f'\tConnected to CUDA GPU: {torch.cuda.get_device_name()}!!')
    else:
        print(f"\tNo GPU acceleration available on {system}, using CPU!!")

print(f'\tUsing device: {device}')

model = predictNE().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#%% train model

num_epochs = 50

print('Starting model training...')

# Add GPU memory monitoring
if device.type == 'cuda':
    print(f'GPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved')

# Store training history
train_losses = []
val_losses = []

epoch_pbar = tqdm(range(num_epochs), desc="Training")
for epoch in epoch_pbar:
    
    model.train()
    total_loss = 0
    batch_count = 0
    
    batch_pbar = tqdm(loader_train, desc=f"Epoch {epoch+1}", leave=False)
    cudnn_disabled = False  # Track if we've disabled cuDNN for this epoch
    
    for batch_idx, (X_batch, y_batch) in enumerate(batch_pbar):
        
        # Check for NaN values before moving to device
        if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
            print(f"\nWarning: NaN values detected in batch {batch_idx}, skipping...")
            continue
        
        # Move to device and verify
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        # Debug: Print device info for first batch
        if batch_idx == 0 and epoch == 0:
            print(f'\nFirst batch debugging:')
            print(f'\tX_batch device: {X_batch.device}')
            print(f'\ty_batch device: {y_batch.device}')
            print(f'\tX_batch shape: {X_batch.shape}')
            print(f'\ty_batch shape: {y_batch.shape}')
            if device.type == 'cuda':
                print(f'\tGPU memory after batch load: {torch.cuda.memory_allocated()/1e9:.2f} GB')
        
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        # Debug: Check prediction device for first batch
        if batch_idx == 0 and epoch == 0:
            print(f'\ty_pred device: {y_pred.device}')
            print(f'\ty_pred shape: {y_pred.shape}')
        
        # Check for NaN in predictions
        if torch.isnan(y_pred).any():
            print(f"\nWarning: NaN values in predictions at batch {batch_idx}, skipping batch...")
            continue
        
        loss = criterion(y_pred, y_batch)
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"\nWarning: NaN loss detected at batch {batch_idx}, skipping batch...")
            continue
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # Update batch progress bar with current loss
        batch_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Clear cache every 10 batches to prevent memory buildup
        if batch_idx % 10 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Calculate average training loss for this epoch
    avg_train_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    train_losses.append(avg_train_loss)
    
    # Validation evaluation
    model.eval()
    val_total_loss = 0
    val_batch_count = 0
    
    with torch.no_grad():
        for X_val_batch, y_val_batch in loader_val:
            # Check for NaN values
            if torch.isnan(X_val_batch).any() or torch.isnan(y_val_batch).any():
                continue
            
            # Move to device
            X_val_batch = X_val_batch.to(device, non_blocking=True)
            y_val_batch = y_val_batch.to(device, non_blocking=True)
            
            # Forward pass
            y_val_pred = model(X_val_batch)
            
            # Check for NaN in predictions
            if torch.isnan(y_val_pred).any():
                continue
            
            val_loss = criterion(y_val_pred, y_val_batch)
            
            # Check for NaN in loss
            if torch.isnan(val_loss):
                continue
            
            val_total_loss += val_loss.item()
            val_batch_count += 1
    
    # Calculate average validation loss for this epoch
    avg_val_loss = val_total_loss / val_batch_count if val_batch_count > 0 else float('inf')
    val_losses.append(avg_val_loss)
    
    # Update epoch progress bar with both losses
    epoch_pbar.set_postfix({
        "Train Loss": f"{avg_train_loss:.4f}", 
        "Val Loss": f"{avg_val_loss:.4f}"
    })
    
    # Print detailed loss info after every epoch
    print(f"\nEpoch {epoch+1}/{num_epochs}:")
    print(f"  Training Loss: {avg_train_loss:.6f}")
    print(f"  Validation Loss: {avg_val_loss:.6f}")
    if device.type == 'cuda':
        print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

#%% save model

print('Training completed!')
print('Saving model...')

# Create timestamped directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f'models/{timestamp}'
os.makedirs(save_dir, exist_ok=True)

# Save the model state dict
torch.save(model.state_dict(), f'{save_dir}/predictNE_model.pth')

# Also save the entire model (includes architecture)
torch.save(model, f'{save_dir}/predictNE_model_full.pth')

# Save training info
training_info = {
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'seq_len': seq_len,
    'final_train_loss': train_losses[-1] if train_losses else float('inf'),
    'final_val_loss': val_losses[-1] if val_losses else float('inf'),
    'train_loss_history': train_losses,
    'val_loss_history': val_losses,
    'model_architecture': str(model),
    'device': str(device),
    'timestamp': timestamp
}

torch.save(training_info, f'{save_dir}/training_info.pth')

print(f'Model saved successfully in {save_dir}!')
print(f'  - Model weights: {save_dir}/predictNE_model.pth')
print(f'  - Full model: {save_dir}/predictNE_model_full.pth')
print(f'  - Training info: {save_dir}/training_info.pth')

#%% to load

# # model = predictNE()
# model.load_state_dict(torch.load('predictNE_model.pth', map_location=torch.device('cpu')))

# # Or load full model
# # model = torch.load('predictNE_model_full.pth')

# # Load training info
# info = torch.load('training_info.pth', map_location=torch.device('cpu'))

# #%% predict NE from new timecourse

# def predict_NE_from_timecourse(ca_timecourse, model, device, seq_len=300):
#     """
#     Predict NE from a calcium timecourse using the trained model.
    
#     Args:
#         ca_timecourse (np.ndarray): Input Ca data of shape (T, 12)
#         model: Trained PyTorch model
#         device: Device to run inference on
#         seq_len (int): Sequence length used during training
        
#     Returns:
#         np.ndarray: Predicted NE timecourse of shape (T,)
#     """
#     model.eval()
    
#     T, n_features = ca_timecourse.shape
#     print(f"Input timecourse shape: {ca_timecourse.shape}")
    
#     if n_features != 12:
#         raise ValueError(f"Expected 12 features, got {n_features}")
    
#     if T < seq_len:
#         raise ValueError(f"Timecourse too short. Need at least {seq_len} timepoints, got {T}")
    
#     # Initialize output array
#     ne_predictions = np.zeros(T)
    
#     # Convert to tensor
#     ca_tensor = torch.tensor(ca_timecourse, dtype=torch.float32).to(device)
    
#     with torch.no_grad():
#         # For the first seq_len points, we can only predict starting from seq_len
#         # So we'll fill the first seq_len-1 points with the first prediction
        
#         # Get first prediction
#         first_seq = ca_tensor[:seq_len].unsqueeze(0)  # Add batch dimension
#         first_pred = model(first_seq).squeeze().cpu().numpy()  # Remove batch dim
        
#         # Fill first seq_len points
#         ne_predictions[:seq_len] = first_pred
        
#         # Predict for remaining timepoints using sliding window
#         for start_idx in range(1, T - seq_len + 1):
#             seq = ca_tensor[start_idx:start_idx + seq_len].unsqueeze(0)
#             pred = model(seq).squeeze().cpu().numpy()
            
#             # Take the last prediction (most recent timestep)
#             ne_predictions[start_idx + seq_len - 1] = pred[-1]
    
#     print(f"Generated NE predictions with shape: {ne_predictions.shape}")
#     return ne_predictions

# # Example usage:
# # Select a test timecourse (first test recording)
# example_ca = Ca_test[0]  # Shape: (T, 12)
# example_ne_true = NE_test[0]  # True NE values for comparison

# print("Making predictions on example timecourse...")
# predicted_ne = predict_NE_from_timecourse(example_ca, model, device)

# # Plot comparison
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

# #%% Calculate correlation
# correlation = np.corrcoef(example_ne_true.flatten(), predicted_ne.flatten())[0, 1]
# rmse = np.mean((example_ne_true - predicted_ne) ** 2) ** 0.5

# print(f"Prediction quality:")
# print(f"  Correlation: {correlation:.4f}")
# print(f"  RMSE: {rmse:.4f}")

