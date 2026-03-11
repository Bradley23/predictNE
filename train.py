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
from models import build_model
from models.lstm import predictNE
from models.GNN import GNN_NE, fully_connected_edge_index
from datasets.createDataset import createDataset
from torch.utils.data import Dataset, DataLoader
from metrics import predict_NE_from_timecourse, corr_rmse
import yaml
from datasets.processing import calc_connectivity, window_clip, bpf
from diagnostics import Report

#%%
def train(Ca_train, NE_train, Ca_val, NE_val):
    #%% ############### Initialize parameters ###############
    seq_len = 300
    batch_size = 64  # Restore larger batch size - 12GB should handle this easily

    criterion = nn.MSELoss()

    NUM_EPOCHS = 10

    LR = 1e-3

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    #%% ############### create datasets ###############

    print('Creating datasets and data loaders...')
    dataset_train = createDataset(Ca_train, NE_train, seq_len)
    dataset_val = createDataset(Ca_val, NE_val, seq_len)
    # dataset_test = createDataset(Ca_test, NE_test, seq_len)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    # loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

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

    model = build_model(config).to(device)

    edge_index = fully_connected_edge_index(12).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    #%% ############### train model ###############

    print(f'Starting model training for {NUM_EPOCHS} epochs...')

    # Store training history
    train_losses = []
    val_losses = []

    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Training")
    for epoch in epoch_pbar:
        
        model.train()
        total_loss = 0
        batch_count = 0
        
        # batch_pbar = tqdm(loader_train, desc=f"Epoch {epoch+1}", leave=False)
        # for batch_idx, (X_batch, y_batch) in enumerate(batch_pbar):
        for (X_batch, y_batch) in loader_train:
            
            # Move to device and verify
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            y_pred = model(X_batch, edge_index)
            
            # loss = criterion(y_pred, y_batch)
            loss_full = criterion(y_pred, y_batch)
            loss_final = criterion(y_pred[:, -1], y_batch[:, -1])

            loss = loss_full + 2.0 * loss_final

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Update batch progress bar with current loss
            # batch_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
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
                y_val_pred = model(X_val_batch, edge_index)

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

    print('Training completed!')

    # store training info
    training_info = {
        'num_epochs': NUM_EPOCHS,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'final_train_loss': train_losses[-1] if train_losses else float('inf'),
        'final_val_loss': val_losses[-1] if val_losses else float('inf'),
        'train_loss_history': train_losses,
        'val_loss_history': val_losses,
        'model_architecture': str(model),
        'device': str(device)
    }

    return model, optimizer, training_info

if __name__ == "__main__":
    #%% ############### load datasets ###############

    print('Loading datasets...')
    print('\tLoading Ca data...')
    Ca = np.load('datasets/Ca_stack.npy')
    print('\tLoading HbT data...')
    HbT = np.load('datasets/HbT_stack.npy')
    print('\tLoading NE data...')
    NE = np.load('datasets/NE_stack.npy')

    print('\tSuccessfully loaded all data!!')

    #%% process data (optional)

    window_size = 100
    eps = 1e-6

    # Ca = bpf(Ca, freq=[0, 0.5], fs=10)
    NE = bpf(NE, freq=[0, 0.5], fs=10)
    conn = calc_connectivity(Ca, window_size=window_size)
    conn = np.arctanh(np.clip(conn, -1 + eps, 1 - eps)) # fisher z transform Pearson's correlation values
    NE = window_clip(NE, window_size=window_size)
    Ca = window_clip(Ca, window_size=window_size)
    HbT = window_clip(HbT, window_size=window_size)

    # Ca = np.concatenate((Ca, HbT), axis=2)
    # Ca = conn

    #%% ############### intialize parameters ###############

    print('Initializing parameters...')

    N = Ca.shape[0]
    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    np.random.seed(24)

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

    model, optimizer, training_info = train(Ca_train, NE_train, Ca_val, NE_val)

    print("Making predictions on example timecourse...")

    device = torch.device("cuda")
    
    edge_index = fully_connected_edge_index(12).to(device)
    predicted_ne = predict_NE_from_timecourse(Ca_test, model, device, 300, edge_index)
    correlation, rmse = corr_rmse(NE_test, predicted_ne)
    
    print(f"Prediction quality:")
    print(f"  Correlation: {np.mean(correlation):.4f}")
    print(f"  RMSE: {np.mean(rmse):.4f}")

    # PLOT RESULTS

    report = Report('checkpoints/LSTM_report.pdf')
    report.plot_loss(training_info['train_loss_history'], training_info['val_loss_history'])
    report.close()

    # print('Saving model...')

    # Create timestamped directory
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save_path = f'checkpoints/{timestamp}.pt'
    
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'training_ingo': training_info
    # }, save_path)

    # print(f'Model saved successfully in {save_path}!')