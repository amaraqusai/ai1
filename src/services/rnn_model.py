"""
RNN Model definition.
"""
import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_size: int = 64, num_layers: int = 2, output_dim: int = 1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_dim)
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, seq_len, input_dim)
        out, _ = self.rnn(x)
        # Take the output from the last timestep
        last_out = out[:, -1, :]
        return self.fc(last_out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
