
#!/usr/bin/env python3
"""Accou-style dilated convolution baseline with a learned spatial projection."""
from torch import nn

class DilatedConv(nn.Module):
    def __init__(self, input_channels, channels=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, channels, 1), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=3, dilation=3), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=9, dilation=9), nn.ReLU(),
            nn.Conv1d(channels, 11, 1),
        )
    def forward(self, neural):
        output = self.layers(neural)
        return {"envelope": output[:, :1], "mel": output[:, 1:]}

def create_model(input_channels, **_): return DilatedConv(input_channels)
