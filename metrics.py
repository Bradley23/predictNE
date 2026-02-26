import torch
import numpy as np

def predict_NE_from_timecourse(ca_timecourse, model, device, seq_len=300):
    
    N = len(ca_timecourse)
    ne_predictions = []
    
    model.eval()
    T = ca_timecourse[0].shape[0]

    for i in range(N):
        # Initialize output array
        ne_predictions.append(np.zeros(T))
        ca_tensor = torch.tensor(ca_timecourse[i], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            # Get first prediction
            first_seq = ca_tensor[:seq_len].unsqueeze(0)  # Add batch dimension
            first_pred = model(first_seq).squeeze().cpu().numpy()  # Remove batch dim
            
            # Fill first seq_len points
            ne_predictions[i][:seq_len] = first_pred
            
            # Predict for remaining timepoints using sliding window
            for start_idx in range(1, T - seq_len + 1):
                seq = ca_tensor[start_idx:start_idx + seq_len].unsqueeze(0)
                pred = model(seq).squeeze().cpu().numpy()
                
                # Take the last prediction (most recent timestep)
                ne_predictions[i][start_idx + seq_len - 1] = pred[-1]
    
    return ne_predictions

def corr_rmse(real, predicted):
    
    N = len(predicted)

    correlation = []
    rmse = []

    for i in range(N):
        correlation.append(np.corrcoef(real[i].flatten(), predicted[i].flatten())[0, 1])
        rmse.append(np.mean((real[i] - predicted[i]) ** 2) ** 0.5)

    return correlation, rmse