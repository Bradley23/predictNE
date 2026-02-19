import torch.nn as nn

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