"""Predicts global NE release patterns from widefield Ca2+ 
imaging data acquired from 12 allen atlas regions.
author - Brad Rauscher (February, 2026)"""

#%% ############### import packages ###############
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import platform
import os
from datetime import datetime
from models.lstm import predictNE
from datasets.createDataset import createDataset
from torch.utils.data import Dataset, DataLoader

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

#%% ############### create datasets ###############

seq_len = 300
batch_size = 64  # Restore larger batch size - 12GB should handle this easily

print('Creating datasets and data loaders...')
dataset_train = createDataset(Ca_train, NE_train, seq_len)
dataset_val = createDataset(Ca_val, NE_val, seq_len)
dataset_test = createDataset(Ca_test, NE_test, seq_len)

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

#%% ############### initialize model ###############

print('Detecting available compute device...')

# Detect operating system and set device accordingly
system = platform.system()

if system == "Darwin":  # macOS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.backends.mps.is_available():
        print('\tConnected to Mac GPU (MPS)!!')
    else:
        print("\tCouldn't find Mac GPU, using CPU!!")
else:  # Linux or Windows
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f'\tConnected to NVIDIA GPU: {torch.cuda.get_device_name()}!!')
    else:
        print("\tCouldn't find CUDA GPU, using CPU!!")

print(f'\tUsing device: {device}')

model = predictNE().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#%% ############### train model ###############

num_epochs = 20

print(f'Starting model training for {num_epochs} epochs...')

# Store training history
train_losses = []
val_losses = []

epoch_pbar = tqdm(range(num_epochs), desc="Training")
for epoch in epoch_pbar:
    
    model.train()
    total_loss = 0
    batch_count = 0
    
    batch_pbar = tqdm(loader_train, desc=f"Epoch {epoch+1}", leave=False)
    
    for batch_idx, (X_batch, y_batch) in enumerate(batch_pbar):
        
        # Move to device and verify
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch)
        
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
            
            # Move to device
            X_val_batch = X_val_batch.to(device, non_blocking=True)
            y_val_batch = y_val_batch.to(device, non_blocking=True)
            
            # Forward pass
            y_val_pred = model(X_val_batch)
            
            val_loss = criterion(y_val_pred, y_val_batch)
            
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
    # print(f"\nEpoch {epoch+1}/{num_epochs}:")
    # print(f"  Training Loss: {avg_train_loss:.6f}")
    # print(f"  Validation Loss: {avg_val_loss:.6f}")

#%% ############### save model ###############

print('Training completed!')
print('Saving model...')

# Create timestamped directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f'checkpoints/{timestamp}.pt'
# os.makedirs(save_dir, exist_ok=True)

# Save the model state dict
# torch.save(model.state_dict(), f'{save_dir}/predictNE_model.pth')

# Also save the entire model (includes architecture)
# torch.save(model, f'{save_dir}/predictNE_model_full.pth')

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

# torch.save(training_info, f'{save_dir}/training_info.pth')

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'training_ingo': training_info
}, save_path)

print(f'Model saved successfully in {save_path}!')