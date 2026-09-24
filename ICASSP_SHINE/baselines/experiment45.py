
#!/usr/bin/env python3
"""AWaveNet baseline adapted to arbitrary neural channel counts and joint targets."""
import math
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, 2 * channels, 3, padding=dilation, dilation=dilation)
        self.residual = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)
    def forward(self, x):
        value, gate = self.dilated(x).chunk(2, 1)
        h = torch.tanh(value) * torch.sigmoid(gate)
        return (x + self.residual(h)) * math.sqrt(0.5), self.skip(h)

class AWaveNet(nn.Module):
    def __init__(self, input_channels, channels=64, layers=36, cycle=12):
        super().__init__()
        self.input = nn.Conv1d(input_channels, channels, 1)
        self.blocks = nn.ModuleList([ResidualBlock(channels, 2 ** (i % cycle)) for i in range(layers)])
        self.output = nn.Sequential(nn.GELU(), nn.Conv1d(channels, channels, 1), nn.GELU(), nn.Conv1d(channels, 11, 1))
    def forward(self, neural):
        x = self.input(neural); skip = 0
        for block in self.blocks:
            x, current = block(x); skip = skip + current
        output = self.output(skip * math.sqrt(1.0 / len(self.blocks)))
        return {"envelope": output[:, :1], "mel": output[:, 1:]}

def create_model(input_channels, **_): return AWaveNet(input_channels)
