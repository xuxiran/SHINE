
#!/usr/bin/env python3
"""CNN-LSTM baseline adapted to arbitrary neural channel counts and joint targets."""
import torch
import torch.nn.functional as F
from torch import nn

class CNNLSTM(nn.Module):
    def __init__(self, input_channels, kernels=5, kernel_time=16, hidden=8):
        super().__init__()
        self.kernel_time = kernel_time
        self.conv = nn.Conv1d(input_channels, kernels, kernel_time)
        self.lstms = nn.ModuleList([nn.LSTM(1, hidden, batch_first=True) for _ in range(kernels)])
        self.head = nn.Linear(kernels * hidden, 11)
    def forward(self, neural):
        left = (self.kernel_time - 1) // 2
        right = self.kernel_time - 1 - left
        x = self.conv(F.pad(neural, (left, right)))
        streams = [lstm(x[:, i].unsqueeze(-1))[0] for i, lstm in enumerate(self.lstms)]
        output = self.head(torch.cat(streams, dim=-1)).transpose(1, 2)
        return {"envelope": output[:, :1], "mel": output[:, 1:]}

def create_model(input_channels, **_): return CNNLSTM(input_channels)
