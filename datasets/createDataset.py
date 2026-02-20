import torch
from torch.utils.data import Dataset

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